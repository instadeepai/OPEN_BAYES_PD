# pylint: disable=W0212,C0301,E1129,W0612,R0913,R0914,R0917
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from captum.attr import (
    LRP,
    DeepLift,
    DeepLiftShap,
    GuidedGradCam,
    IntegratedGradients,
    Saliency,
)
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule
from omegaconf import DictConfig
from pyro.nn import PyroModule
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_processing.esm_embeddings import extract_batch_embeddings, load_esm_model
from src.explainers.captum_metrics import (
    calculate_auc_ma_for_aggregated_maps,
    calculate_captum_metrics,
)
from src.explainers.model_conversion_utils import (
    from_pyro_to_pytorch_baseline_cnn_scorer,
    sample_weights_from_guide,
)
from src.explainers.plotting_utils import (
    create_statistical_analysis,
    plot_captum_values_cnn_baseline,
)
from src.models.constants import NUMERICAL_STABILITY_CONSTANT
from src.models.core.bayesian_cnn import DeterministicBayesianCNNCounterpart


@dataclass
class ExplainerConfig:
    """Static configuration for explainers (hyperparams and model/guide/cfg)."""

    baseline_size: Optional[int] = None
    model: Optional[torch.nn.Module] = None
    guide: Optional[PyroModule] = None
    cfg: Optional[DictConfig] = None
    device: Optional[torch.device] = None


@dataclass
class ExplainerState:
    """Mutable runtime state for the explainer (model, guide, explainer, etc.)."""

    affinity_predictor: Optional[torch.nn.Module] = None
    explainer: Optional[object] = None
    name: str = "None"
    use_mean_model: bool = False
    needs_baseline: bool = False


class CaptumExplainer:
    """
    Base class for all Captum explainers.

    This class provides the common functionality for all Captum-based explainers:
    - Model conversion from Pyro to PyTorch
    - Generic explanation generation
    - Statistical analysis and plotting
    - Baseline setup for explainers that need it
    """

    # Note: avoid __slots__ here to keep refactor simple and allow attributes like cfg

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """
        Initialize the base Captum explainer.

        Parameters:
        ----------
        config : ExplainerConfig | None
            Configuration for the explainer (epsilon, baseline size, model/guide/cfg).
        """
        self.config = config
        self.state = ExplainerState()
        self.args = {}
        self.kwargs = {}

    def setup_model(
        self,
        weights: dict = None,
    ):
        """Create deterministic counterpart from probabilistic model and guide."""
        # Create a deterministic counterpart from the trained probabilistic model
        # and guide
        self.state.affinity_predictor = from_pyro_to_pytorch_baseline_cnn_scorer(
            self.config.model, self.config.guide, self.config.cfg, weights
        ).to(self.config.device)

    def setup_baseline(
        self, batch_size: int, sequence_length: int, use_padding: bool = True
    ) -> torch.Tensor:
        """
        Setup baseline for explainers that need it.

        Creates random sequences of length 150 with numbers between 2 and 23,
        starting with 0, and passes them through the ESM embedder.

        Parameters:
        ----------
        batch_size : int
            Number of sequences to generate

        Returns:
        -------
        torch.Tensor
            Baseline tensor (random embeddings from random sequences)
        """
        # Create random sequences of length 150 with numbers between 2 and 23,
        # starting with 0
        random_sequences = []
        for _ in range(batch_size):
            # Create sequence starting with 0
            sequence = [0]
            # Add random numbers between 1 and 23, or 1 if you
            # want to use a fully padded baseline or a fully random baseline
            sequence.extend(
                [
                    random.randint(1, 23) if use_padding else 1
                    for _ in range(sequence_length - 1)
                ]
            )
            random_sequences.append(sequence)
        # Convert sequences to strings of numbers
        number_sequences = []
        for seq in random_sequences:
            number_seq = "".join([str(num) for num in seq])
            number_sequences.append(number_seq)

        model = load_esm_model(
            self.config.cfg.preprocessing.esm_model_name,
            self.config.cfg.model.cnn_sequence_length,
            self.config.cfg.model.fitness_predictor_architecture,
        )
        # Create embeddings for random sequences
        with torch.no_grad():
            embeddings = extract_batch_embeddings(model, number_sequences)
            embeddings = embeddings.transpose(1, 2)

        return embeddings

    def explain(
        self,
        data_to_explain: DataLoader,
        prediction: List[float],
        prediction_confidence: List[float],
        cdrs: List[List[int]],
        temp_address: str,
        save_images: bool = False,
        calculate_metrics: bool = True,
        num_model_to_explain: int = 30,
    ) -> tuple[dict, List[dict]]:
        """
        Generate explanations for the given data.

        Parameters:
        ----------
        data_to_explain : DataLoader
            Dataloader for sequences to run predictions on
        prediction : List[float]
            Predicted mean binding probability from baseline model
        prediction_confidence : List[float]
            Predicted standard deviation binding probability
        cdrs : List[List[int]]
            CDR starting and ending positions for 3 CDRs per data point
        temp_address : str
            Folder address to store explanation images
        save_images : bool
            Whether to save explanation images
        calculate_metrics : bool
            Whether to calculate metrics
        num_model_to_explain : int
            Number of models to explain

        Returns:
        -------
        tuple
            - dict: aggregated relevance scores (mean/union/intersection/uai_plus)
            - List[dict]: averaged metrics per sequence
        """
        # Prepare explanation kwargs and check num_model_to_explain
        kwargs = self._prepare_explanation_kwargs()
        num_model_to_explain = self._is_num_model_zero(num_model_to_explain)

        # Generate explanations for multiple models
        results = self._generate_multi_model_explanations(
            data_to_explain, calculate_metrics, num_model_to_explain, kwargs, cdrs
        )
        all_relevance_scores, deterministic_predictions, _all_metrics = results

        # Aggregate relevance scores (mean only if single model; otherwise add unions)
        all_aggregated_scores = self._aggregate_relevance_scores(
            all_relevance_scores,
            percentile_union=self.config.cfg.model.explainer.percentile_union,
            percentile_intersection=self.config.cfg.model.explainer.percentile_inter,
        )
        averaged_deterministic_predictions = self._average_predictions(
            deterministic_predictions
        )

        # Save visualizations if requested
        aggregated_auc_ma = None
        if save_images:
            compute_multi = len(all_relevance_scores) > 1
            # Compute per-sequence AUC/MA for mean/union/intersection/uai_plus
            aggregated_auc_ma = calculate_auc_ma_for_aggregated_maps(
                all_aggregated_scores,
                cdrs,
                sequence_length=int(self.config.cfg.model.cnn_sequence_length),
                include_multi=compute_multi,
            )

            plot_captum_values_cnn_baseline(
                data_to_explain,
                averaged_deterministic_predictions,
                prediction,
                prediction_confidence,
                all_aggregated_scores,  # Pass all three types
                cdrs,
                temp_address,
                self.state.name,
                num_model_to_explain > 1,
                aggregated_auc_ma,
                sequence_length=int(self.config.cfg.model.cnn_sequence_length),
            )

            create_statistical_analysis(
                all_aggregated_scores,
                cdrs,
                temp_address,
                self.state.name,
                num_model_to_explain > 1,
            )

        # Bundle metrics
        bundled_metrics = {
            "auc_ma": aggregated_auc_ma if save_images else None,
        }

        # Return aggregated scores and bundled metrics
        return all_aggregated_scores, bundled_metrics

    def _prepare_explanation_kwargs(self) -> dict:
        """Prepare kwargs for explanation generation."""
        kwargs = self.kwargs.copy()
        if self.config.baseline_size:
            kwargs["baselines"] = self.setup_baseline(
                self.config.baseline_size,
                int(self.config.cfg.model.cnn_sequence_length),
                self.config.cfg.model.explainer.use_padding_for_baseline,
            )
        return kwargs

    def _is_num_model_zero(self, num_model_to_explain: int) -> int:
        """If num_model_to_explain is 0, it will use the mean model"""
        if num_model_to_explain == 0:
            print("num_model_to_explain is 0, using the mean model")
            num_model_to_explain = 1
            self.state.use_mean_model = True
        return num_model_to_explain

    def save_and_upload_model(self, local_path: str, model_state_dict: dict):
        """
        Saves the model's state dictionary to a local file and optionally uploads it.
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        torch.save(model_state_dict, local_path)

    def _generate_multi_model_explanations(
        self,
        data_to_explain: DataLoader,
        calculate_metrics: bool,
        num_model_to_explain: int,
        kwargs: dict,
        cdrs: list | None,
    ) -> tuple:
        """Generate explanations for multiple models."""
        all_relevance_scores = []
        deterministic_predictions = []
        all_metrics = []

        for _ in tqdm(range(num_model_to_explain), desc="Sampling models"):
            # Setup model with sampled weights or mean model
            weights = sample_weights_from_guide(
                self.config.guide, self.state.use_mean_model
            )
            self.setup_model(weights)
            self.state.affinity_predictor.train()
            if self.config.cfg.model.prediction.save_deterministic_ckpt:
                self.save_and_upload_model(
                    self.config.cfg.model.prediction.deterministic_checkpoint_path,
                    self.state.affinity_predictor.state_dict(),
                )

            # Generate explanations for this model
            model_results = self._generate_single_model_explanations(
                data_to_explain, calculate_metrics, kwargs, cdrs
            )

            all_relevance_scores.append(model_results[0])
            deterministic_predictions.append(model_results[1])
            if calculate_metrics:
                all_metrics.append(model_results[2])

        return all_relevance_scores, deterministic_predictions, all_metrics

    def _generate_single_model_explanations(
        self,
        data_to_explain: DataLoader,
        calculate_metrics: bool,
        kwargs: dict,
        cdrs: list | None,
    ) -> tuple:
        """Generate explanations for a single model."""
        model_relevance_scores = []
        model_deterministic_predictions = []
        model_metrics = []

        seq_idx = 0
        for data_point in data_to_explain:
            embeddings = data_point["embeddings"]

            # Generate prediction
            prediction_prob = self._generate_prediction(embeddings)
            model_deterministic_predictions.extend(prediction_prob.flatten().tolist())

            # Generate attribution
            kwargs["inputs"] = embeddings
            kwargs["target"] = 0
            time_start = time.time()
            attributions = self.state.explainer.attribute(**kwargs)
            time_end = time.time()
            print(
                f"Time taken to generate attributions: {time_end - time_start} seconds"
            )

            # Calculate metrics if requested
            if calculate_metrics:
                metrics = calculate_captum_metrics(
                    attributions,
                    cdr_positions=(cdrs[seq_idx] if isinstance(cdrs, list) else None),
                    sequence_length=int(self.config.cfg.model.cnn_sequence_length),
                )
                model_metrics.append(metrics)

            # Process relevance scores
            relevance_per_position = np.sum(
                attributions.detach().cpu().numpy(), axis=1
            ).squeeze()
            model_relevance_scores.append(relevance_per_position)
            seq_idx += 1

        return model_relevance_scores, model_deterministic_predictions, model_metrics

    def _generate_prediction(self, embeddings: torch.Tensor) -> np.ndarray:
        """Generate prediction for given embeddings."""
        with torch.no_grad():
            pred = self.state.affinity_predictor(embeddings)
            pred = torch.clamp(pred, min=-10.0, max=10.0)
            prediction_prob = (
                torch.clamp(
                    torch.nn.functional.sigmoid(pred),
                    min=NUMERICAL_STABILITY_CONSTANT,
                    max=1.0 - NUMERICAL_STABILITY_CONSTANT,
                )
                .detach()
                .cpu()
                .numpy()
            )
        return prediction_prob

    def _aggregate_relevance_scores(
        self,
        all_relevance_scores: List,
        percentile_union: int = 95,
        percentile_intersection: int = 5,
    ) -> dict:
        """Aggregate relevance across models.

        Comment: if compute_multi is False, only 'mean' is returned.
        """
        num_data_points = len(all_relevance_scores[0][0])

        results: dict[str, list] = {"mean": []}
        compute_multi = len(all_relevance_scores) > 1
        if compute_multi:
            results.update(
                {
                    "union": [],
                    "intersection": [],
                    "uai_plus": [],
                }
            )

        for i in range(num_data_points):
            stacked_relevance = np.stack(
                [model_scores[0][i] for model_scores in all_relevance_scores]
            )

            mean_relevance = np.mean(stacked_relevance, axis=0)
            results["mean"].append(mean_relevance)

            if compute_multi:
                max_abs_relevance = np.max(np.abs(stacked_relevance))
                normalized_relevance = (
                    stacked_relevance / max_abs_relevance
                    if max_abs_relevance > 0
                    else stacked_relevance
                )
                epsilon_uai = self.config.cfg.model.explainer.epsilon_uai
                union_relevance = np.percentile(
                    stacked_relevance, percentile_union, axis=0
                )
                intersection_relevance = np.percentile(
                    stacked_relevance, percentile_intersection, axis=0
                )
                uai_plus_relevance = np.mean(
                    np.abs(normalized_relevance) > epsilon_uai, axis=0
                )
                results["union"].append(union_relevance)
                results["intersection"].append(intersection_relevance)
                results["uai_plus"].append(uai_plus_relevance)

        return results

    def _average_predictions(self, deterministic_predictions: List) -> List[float]:
        """Average predictions across all models."""
        return np.mean(deterministic_predictions, axis=0).tolist()


class LrpCaptumExplainer(CaptumExplainer):
    """
    LRP Captum explainer for the CNN fitness predictor used in the baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.name = "lrp_captum"

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        self._setup_lrp_rules(self.state.affinity_predictor)
        self.state.explainer = LRP(self.state.affinity_predictor)

    def _setup_lrp_rules(self, model: nn.Module) -> nn.Module:
        """
        Setup LRP rules for different module types in the model.
        Applies IdentityRule to BatchNorm, EpsilonRule to others
        (Conv, Linear, Activations, Pooling).

        Parameters:
        ----------
        model : nn.Module
            The model to set up LRP rules for

        Returns:
        -------
        nn.Module
            Model with LRP rules applied
        """
        layers = model.modules()
        for layer in layers:
            # apply epsilon rule to these layers
            if isinstance(
                layer,
                (
                    torch.nn.modules.conv.Conv1d,
                    torch.nn.modules.batchnorm.BatchNorm1d,
                    torch.nn.modules.pooling.MaxPool1d,
                    torch.nn.modules.pooling.AvgPool1d,
                    torch.nn.modules.linear.Linear,
                ),
            ):
                layer.rule = EpsilonRule(
                    self.config.cfg.model.explainer.lrp_captum.epsilon
                )

            # apply identity rule to these layers
            elif isinstance(
                layer,
                (torch.nn.modules.Identity,),
            ):
                layer.rule = IdentityRule()
            # ignore these modules
            elif isinstance(
                layer,
                (
                    DeterministicBayesianCNNCounterpart,
                    nn.Dropout,
                    nn.ModuleList,
                    nn.ReLU,
                    nn.LeakyReLU,
                    nn.SELU,
                    nn.Tanh,
                    nn.Sigmoid,
                ),
            ):
                pass
            else:
                message = (
                    f"Layer {layer} of type {type(layer)} is not a valid layer, "
                    "please add it to the list of layers"
                )
                raise ValueError(message)


class DeepLiftCaptumExplainer(CaptumExplainer):
    """
    DeepLift Captum explainer for the CNN fitness predictor used in the
    baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """
        Setup the model for DeepLift explanations by creating a deterministic
        counterpart.

        Parameters:
        ----------
        model : torch.nn.Module
            The trained probabilistic model
        guide : PyroModule
            The full baseline guide
        cfg : Optional[DictConfig]
            The model config
        """
        super().__init__(config)

        # Create DeepLift explainer
        self.state.name = "deeplift"
        self.state.needs_baseline = True

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        self.state.explainer = DeepLift(self.state.affinity_predictor)


class DeepLiftShapCaptumExplainer(CaptumExplainer):
    """
    DeepLiftShap Captum explainer for the CNN fitness predictor used in the
    baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.needs_baseline = True
        self.state.name = "deeplift_shap"
        self.config.baseline_size = (
            self.config.cfg.model.explainer.deeplift_shap.num_baselines
        )

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        self.state.explainer = DeepLiftShap(self.state.affinity_predictor)


class IntegratedGradientsExplainer(CaptumExplainer):
    """
    Integrated Gradients explainer for the CNN fitness predictor used in the
    baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.name = "integrated_gradient"
        self.kwargs[
            "n_steps"
        ] = self.config.cfg.model.explainer.integrated_gradients.steps
        self.config.baseline_size = 1

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        self.state.explainer = IntegratedGradients(self.state.affinity_predictor)


class GradCamCaptumExplainer(CaptumExplainer):
    """
    GradCAM Captum explainer for the CNN fitness predictor used in the baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.name = "gradcam_captum"

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        # Support either a single target layer index or a list of indices
        layer_idx = 0
        try:
            grad_cam_cfg = self.config.cfg.model.explainer.grad_cam
            if grad_cam_cfg is not None and grad_cam_cfg.target_cnn_layer is not None:
                layer_idx = int(grad_cam_cfg.target_cnn_layer)
        except (AttributeError, TypeError, ValueError) as e:
            print("Exception in grad_cam_cfg", e)
        if layer_idx is None:  # type: ignore[unreachable]
            layer_idx = 0

        # Fallback to gradcam_captum.target_layers (list) if provided
        try:
            gc_cfg = self.config.cfg.model.explainer.gradcam_captum
            has_layers = getattr(gc_cfg, "target_layers", None) is not None
            if gc_cfg is not None and has_layers:
                tl = gc_cfg.target_layers
                try:
                    # If it's a scalar/int-like
                    layer_idx = int(tl)
                except (TypeError, ValueError):
                    # Assume iterable; take the first index
                    layer_idx = int(tl[0])
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            print("Exception in gradcam_captum", e)

        target_layer = self.state.affinity_predictor.layers[layer_idx]
        self.state.explainer = GuidedGradCam(
            self.state.affinity_predictor,
            target_layer,
        )


class SaliencyCaptumExplainer(CaptumExplainer):
    """
    Saliency Captum explainer for the CNN fitness predictor used in the baseline guide.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.name = "saliency"

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)
        self.state.explainer = Saliency(self.state.affinity_predictor)


class RandomCaptumExplainer(CaptumExplainer):
    """
    Random explainer that mimics Captum's interface and returns random attributions.

    It leverages the base `.explain` flow (predictions, aggregation, optional plots)
    but generates attributions as random tensors with the same shape as the inputs.
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        super().__init__(config)
        self.state.name = "random"

    def setup_model(self, weights: dict = None):
        super().setup_model(weights)

        class _RandomAttributor:
            """Random attributor class"""

            def attribute(self, inputs: torch.Tensor, **_: dict) -> torch.Tensor:
                """
                Returns random attributions with the same shape and device as inputs
                """
                # Returns random attributions with the same shape and device as inputs
                return torch.randn_like(inputs)

            def is_random(self) -> bool:
                """Helper to satisfy linters expecting at least two public methods."""
                return True

        self.state.explainer = _RandomAttributor()
