# pylint: disable=E1120,C0301,R0913, R0914, R0917
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from pyro.infer import Predictive
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.explainers.captum_explainers import (
    DeepLiftCaptumExplainer,
    DeepLiftShapCaptumExplainer,
    ExplainerConfig,
    GradCamCaptumExplainer,
    IntegratedGradientsExplainer,
    LrpCaptumExplainer,
    RandomCaptumExplainer,
    SaliencyCaptumExplainer,
)
from src.explainers.cdr_extraction import extract_cdr_positions_with_abnumber
from src.utils.load_model import load_model_and_parameters

EXPLAINERS = {
    "lrp_captum": LrpCaptumExplainer,
    "deeplift": DeepLiftCaptumExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "deeplift_shap": DeepLiftShapCaptumExplainer,
    "gradcam_captum": GradCamCaptumExplainer,
    "saliency": SaliencyCaptumExplainer,
    "random": RandomCaptumExplainer,
}


def _process_predictions(posterior_samples: dict) -> tuple[list, list]:
    """Process posterior samples to extract predictions."""
    preds_proba = posterior_samples["binding_probability"].detach().cpu()
    preds_proba_neg = posterior_samples["negative_binding_probability"].detach().cpu()
    if "target_binding_probability" in posterior_samples:
        preds_proba_target = (
            posterior_samples["target_binding_probability"].detach().cpu()
        )
    else:
        preds_proba_target = None
    preds_proba_mean = torch.mean(preds_proba, dim=0).flatten().tolist()
    preds_proba_std = torch.std(preds_proba, dim=0).flatten().tolist()
    preds_proba_neg_mean = torch.mean(preds_proba_neg, dim=0).flatten().tolist()
    preds_proba_neg_std = torch.std(preds_proba_neg, dim=0).flatten().tolist()
    if preds_proba_target is not None:
        preds_proba_target_mean = (
            torch.mean(preds_proba_target, dim=0).flatten().tolist()
        )
        preds_proba_target_std = torch.std(preds_proba_target, dim=0).flatten().tolist()
    else:
        preds_proba_target_mean = None
        preds_proba_target_std = None

    return (
        preds_proba_mean,
        preds_proba_std,
        preds_proba_neg_mean,
        preds_proba_neg_std,
        preds_proba_target_mean,
        preds_proba_target_std,
        preds_proba,
        preds_proba_neg,
        preds_proba_target,
    )


def _get_save_path(cfg: DictConfig, suffix: str) -> str:
    """Generate save path for images."""
    base_path = cfg.model.prediction.model_dir_path
    if isinstance(base_path, str):
        return base_path + suffix
    return str(base_path / suffix)


def _handle_explanation(
    cfg: DictConfig,
    model,
    guide,
    _for_background: list,
    to_explain: DataLoader,
    preds_proba_mean: list,
    preds_proba_std: list,
    cdrs: Optional[list[int]],
    save_images: bool = True,
    device: torch.device = None,
) -> dict:
    """Handle explanation generation using match case for different explainers."""
    output_dict = {}

    # Get explainer configuration
    explainer_config = cfg.model.explainer
    explainer_type = explainer_config.type

    # Check if explanation is enabled and model is CNN
    if (
        not cfg.model.prediction.explain
        or cfg.model.fitness_predictor_architecture != "CNN"
    ):
        print("Explanation is not enabled or model is not CNN")
        return output_dict

    explainer_config = ExplainerConfig(
        baseline_size=None,
        model=model,
        guide=guide,
        cfg=cfg,
        device=device,
    )

    explainer = EXPLAINERS[explainer_type](config=explainer_config)

    address_save_captum_images = _get_save_path(
        cfg, f"/{explainer_type}_explained_images"
    )

    if save_images:
        os.makedirs(address_save_captum_images, exist_ok=True)
        output_dict[
            f"address_{explainer_type}_save_images"
        ] = address_save_captum_images

    aggregated_scores, bundled_metrics = explainer.explain(
        to_explain,
        preds_proba_mean,
        preds_proba_std,
        cdrs,
        address_save_captum_images,
        save_images=save_images,
        num_model_to_explain=cfg.model.explainer.num_model_to_explain,
    )
    # Save outputs
    output_dict[f"{explainer_type}_values"] = aggregated_scores["mean"]
    if bundled_metrics:
        # captum average metrics per sequence
        if bundled_metrics.get("captum") is not None:
            output_dict["captum_metrics"] = bundled_metrics["captum"]
        # AUC/MA per sequence for each aggregation
        if bundled_metrics.get("auc_ma") is not None:
            output_dict["auc_ma_metrics"] = bundled_metrics["auc_ma"]
    return output_dict


def predict_bnn(  # noqa : CCR001
    to_predict: DataLoader,
    cfg: DictConfig,
    device: torch.device | str,
    embedding_size: tuple[int, int],
    background: Optional[DataLoader] = None,
    cdrs: Optional[list[int]] = None,
    save_images: bool = True,
) -> dict[str, np.ndarray | list[np.ndarray] | str]:
    """
    Loads a saved model and perform binding probability prediction. Offers the
    possibility to explain the prediciton via shapley value deep explainer.

    Parameters:
    ----------
    to_predict (DataLoader):
        a dataLoader containing the data point to predict
    cfg (DictConfig):
        your hydra configuration
    background (Optional[DataLoader]):
        a dataLoader containing the data point to derive the the model average response,
        if explaining is switch on
    cdrs (Optional[list[int]]):
        a list of cdr starting and ending point positions, for the 3 CDRs and per data
        point

    Returns:
    -------
        A dictionnary containing the prediciton for
        preds_proba_mean: the mean binding proability,
        preds_proba_std: the standard deviation of the binding proability,
        shap_values: optionnaly the shapley value
        adresse_save_images: optionally the adresse of the folder containing the
        explained png
    """
    model, guide = load_model_and_parameters(cfg, device, embedding_size)

    model.to(device)
    guide.to(device)

    preds_proba_mean = []
    preds_proba_std = []
    preds_proba_neg_mean = []
    preds_proba_neg_std = []
    preds_proba_target_mean = []
    preds_proba_target_std = []
    preds_proba = []
    preds_proba_neg = []
    preds_proba_target = []
    for_background = []
    sequences = []
    with torch.no_grad():
        predictive_object = Predictive(
            model, guide=guide, num_samples=cfg.model.prediction.number_of_particles
        )
        # Process predictions
        for data in tqdm(to_predict, desc="Predicting"):
            data_for_inference = data.copy()
            data_for_inference["selected_count"] = None
            data_for_inference["selected_frequency"] = None
            data_for_inference["selectivity"] = None
            posterior_samples = predictive_object(data_for_inference)
            (
                batch_mean,
                batch_std,
                batch_mean_neg,
                batch_std_neg,
                batch_mean_target,
                batch_std_target,
                batch_proba,
                batch_proba_neg,
                batch_proba_target,
            ) = _process_predictions(posterior_samples)
            preds_proba_mean.extend(batch_mean)
            preds_proba_std.extend(batch_std)
            preds_proba_neg_mean.extend(batch_mean_neg)
            preds_proba_neg_std.extend(batch_std_neg)
            preds_proba.extend(batch_proba)
            preds_proba_neg.extend(batch_proba_neg)
            sequences.extend(data["sequence"])

            # Handle target probabilities (might be None)
            if batch_mean_target is not None:
                preds_proba_target_mean.extend(batch_mean_target)
                preds_proba_target_std.extend(batch_std_target)
                preds_proba_target.extend(batch_proba_target)

            else:
                # Fill with zeros if no target probabilities
                preds_proba_target_mean.extend([0.0] * len(batch_mean))
                preds_proba_target_std.extend([0.0] * len(batch_mean))

        # Process background data
        if background is not None:
            for_background.extend(list(background))

    # Prepare output
    output_dict = {
        "preds_proba_mean": preds_proba_mean,
        "preds_proba_std": preds_proba_std,
        "preds_proba_neg_mean": preds_proba_neg_mean,
        "preds_proba_neg_std": preds_proba_neg_std,
        "preds_proba_target_mean": preds_proba_target_mean,
        "preds_proba_target_std": preds_proba_target_std,
        "preds_proba": preds_proba,
        "preds_proba_neg": preds_proba_neg,
        "preds_proba_target": preds_proba_target,
        "sequence": sequences,
    }

    # Handle explanations if requested
    if cfg.model.prediction.explain:
        cdrs = extract_cdr_positions_with_abnumber(to_predict)
        explanation_dict = _handle_explanation(
            cfg,
            model,
            guide,
            for_background,
            to_predict,
            preds_proba_mean,
            preds_proba_std,
            cdrs,
            save_images,
            device,
        )
        output_dict.update(explanation_dict)

    return output_dict
