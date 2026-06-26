## Bayesian Neural Networks for Phage Display Data

### Overview
This repository contains the reference implementation for a sequence-to-score model based on Bayesian Neural Networks (BNNs). Score is relative to the binding affinity. We train BNNs inside a loop that explicitly simulates the phage display process and its measurement noise, in order to reason jointly about experimental noise and model uncertainty.

Abstract context: Phage display is a powerful technique to study interactions between proteins and other molecules. Its combination with deep learning for protein design is underused due to high noise levels and complex pre-processing and interpretation. We propose a BNN trained within a loop that simulates the experiment and its noise, aiming to enable reliable application by understanding experimental noise and uncertainty. We validate on real binding affinity measurements instead of only proxy values from phage display experiments.

### Key features
- Bayesian training loop tailored to phage display noise
- CNN/MLP/Transformer fitness predictors with configurable architectures
- Hydra-based configuration for clean overrides and reproducibility
- Optional Neptune logging for experiment tracking
- Optional explainability with Captum (Grad-CAM, LRP, Integrated Gradients, DeepLIFT/SHAP). **CAUTION**: Explainability uses CDRs to compute certain localization metrics (AUC-ROC, MA). CDRs are only supported for antibody data. For other types of proteins, the explanation features may not work.

### Repository structure (selected)
- `scripts/`: entrypoints for training and prediction (`train.py`, `predict.py`)
- `config/`: Hydra config (`default.yaml`, model and preprocessing profiles)
- `src/`: data processing, model definitions, explainers, utilities
- `tests/`: unit tests & more on small datasets for CI. Contain a test that check if your CNN configuration is valid.
- `notebook/`: Contains notebooks to visualize the results of the experiment.
- `data/`: Folder containing example data. Place your own data here.

---

## Installation

### Prerequisites
- Python >= 3.10
- Optional: NVIDIA GPU + CUDA for training/inference speed + multiple CPUs to train large experiments

### Recommended: uv (handles custom indexes)
```bash
# Install uv (see https://docs.astral.sh/uv/ for details)
curl -Ls https://astral.sh/uv/install.sh | sh

# From the repo root
uv sync

# To find the CDRs areas during explanation (not mandatory for training)
uv run ANARCI --build_models
```

## Configuration
This project uses Hydra. The main config is `config/default.yaml`, which composes:
- `preprocessing/default.yaml`
- `model/baseline.yaml`

Below is a concise reference of the most relevant options and their allowed values. All keys can be overridden from the CLI.

### Global (`config/default.yaml`)
- `to_cuda`: `cuda` or `cpu`. If `cuda` but no GPU is available, it will fall back to CPU.
- `raise_exception_if_gpus_not_available`: `true` to error out when a GPU is requested but not available; otherwise the code will continue on CPU.
- `use_neptune`: Enable/disable Neptune logging (default: `false`).
- `neptune_project`: Neptune project name when logging is enabled.
- `paths.data_dir`: Base directory for input data.
- `paths.results_dir`: Base directory where experiment outputs are written.

### Preprocessing (`config/preprocessing/default.yaml`)
- `esm_model_name`: Hugging Face ESM or ESM++ checkpoint name. Examples:
  - ESM2: `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `facebook/esm2_t33_650M_UR50D`
  - ESM++: `Synthyra/ESMplusplus_small`, `Synthyra/ESMplusplus_large`
- `esm_batch_size`: Batch size for embedding extraction.
- `representation_layer`: Integer layer index to extract representations from.
- `storage_device`: `cuda` or `cpu` for where to hold the embedding model.
- `dataloader_type`: `normal` (default; reserved for future options).

### Model profile (`config/model/baseline.yaml`)
- `type`: Profile type (e.g., `baseline`).
- `embedding_size`: `[E, L]` — embedding dimension and sequence length. Usually inferred from the chosen ESM model; can be left as reference/default.
- `cnn_sequence_length`: Sequence length used when CNN/Transformer consume per-position embeddings.
- `tags`: Free-form list of labels to attach to experiments.

#### Data splits and input files
- `splits.training_pair`, `splits.validation_pair`, `splits.test_pair`: Lists of experiment identifiers to define train/val/test pairs.
- `splits.relative_csv_file_location`: Optional mapping by assay (e.g., `DNA`, `PVP`) to CSV paths relative to `paths.data_dir`.

#### Experimental constants
- `phage_initial_population_size`: Initial clone population size used by the simulator.
- `sequencing_depth`: Reads per selection used for measurement noise simulation.

#### Architecture selection
- `fitness_predictor_architecture`: One of
  - `MLP`: Classic MLP taking as input the mean value of the ESM-2 embeddings.
  - `CNN`: Classic 2D-CNN taking as input the ESM-2 embeddings.
  - `Transformer`: Classic Transformer taking as input the ESM-2 embeddings.
  - `HybridCNN`: Classic 2D-CNN taking as input the ESM-2 embeddings. Here, only the MLP is Bayesian
  - `DoubleCNNWithNegative`: Create 1 (with 2 outputs) or 2 (with 1 output) CNNs scoring  both the positive and negative selection.
  - `CNNWithBoltzmann`: Create 4, 2 or 1 CNNs (with related outputs) scoring the Boltzmann Constants of the 4 different modes which are: Bind to Base, Do not Bind to Base, Bind to Target, Do not Bind to Target.

Each architecture has a dedicated sub-config:

- `fitness_predictor_mlp`:
  - `layers_size`: Hidden sizes ending with `1` (output). The first element should match the flattened embedding dimension.

- `fitness_predictor_cnn`:
  - `list_of_channels`: Convolutional channels (per conv block; first equals embedding channels).
  - `list_of_kernel_size`: Kernel sizes per conv block.
  - `list_of_stride`: Stride per conv block.
  - `list_of_padding`: Padding per conv block. If residual connections are used, ensure `padding = (kernel - 1) // 2` to preserve length.
  - `layers_size`: Fully-connected head sizes, ending with `1`.
  - `pooling_name`: `MaxPool1d`, `AvgPool1d`, or `none`.
  - `pool_size`, `pool_stride`: Pooling parameters per conv block.
  - `batch_norm_name`: `BatchNorm1d` or `none`.
  - `residual_connections`: `true`/`false` to enable skip connections.
  - `residual_skip_layers`: Integer skip distance for residual connections.

- `fitness_predictor_transformer`:
  - `d_model`: Model hidden size (should match embedding channels).
  - `nhead`: Number of attention heads.
  - `num_encoder_layers`: Number of encoder layers.
  - `dim_feedforward`: Feed-forward network size.
  - `ffn_activation_name`: `tanh`, `sigmoid`, `ReLU`, `LeakyReLU`.
  - `pooling_type`: `mean` or `cls`.

#### Training configuration (`model.training`)
- Paths and naming:
  - `relative_model_checkpoint_location`: Resume from a checkpoint path relative to the experiment directory (or `null`).
  - `relative_csv_file_location`: Optional CSV path relative to `paths.data_dir` to directly point to training data. If `null`, the path is inferred from `splits`.
  - `experiment_name`: Folder name under `results/`.
- Optimization and regularization:
  - `learning_rate`, `weight_decay`, `clip_norm`, `dropout_value`.
  - `activation_name`: `tanh`, `sigmoid`, `ReLU`, `LeakyReLU`.
  - `optimizer`: `AdamW`, `Adam`, `RMSprop`.
  - `use_scheduler`: `true`/`false` with `gamma_scheduler` and `end_of_warmup`.
- Bayesian/SVI controls:
  - `prior_scale`: Prior std scale for weights.
  - `number_of_particles`: Number of MC particles for ELBO.
  - `kl_annealing`: Steps per annealing cycle; `1` disables annealing.
  - `kl_annealing_type`: `linear`, `exponential`, `cyclic`.
  - `latents_to_anneal`: Optional list to restrict annealing to specific latents.
  - `guide_type`: One of `MultivariateNormal`, `DiagonalNormal`, `Normal`, `LowRankMultivariateNormal`.
  - `number_of_models`: `distinct`, `one`, `two`, `sequential_latent`, `sequential_output`.
  - `estimate_method`: `batch`, `global`, `N-1`, `moving_average`.
  - `EMA_beta`: Exponential moving average coefficient for some estimates.
- Loop/runtime:
  - `epochs`, `batch_size`, `shuffle`, `num_workers`.
  - `best_model_decision_set`: `train` or `valid`.
  - `every_n_epochs`: Evaluation/artefact saving cadence.
- Likelihood and data tweaks:
  - `distribution`: Fully-qualified likelihood class. Provided options include
    - Poisson: `src.models.baseline_model.PhageDisplayPoissonRegressor`
    - Multinomial: `src.models.baseline_model.PhageDisplayMultinomialRegressor`
  - `delete_initial_null_counts`: `add_pseudo_counts`, `delete`, or `null`.
  - `is_null_model`: `true` to train/evaluate a null baseline for sanity checks.
  - `test_data_dir`: Optional alternative directory for test-time data.

#### Prediction and explainability (`model.prediction` and `model.explainer`)
- `model.prediction`:
  - `model_dir_path`: Path to a trained experiment directory to load the model.
  - `experiment_name`: Name used for prediction outputs.
  - `background_csv_file_location`: Optional CSV (relative to `paths.data_dir`) providing background sequences for certain explainers.
  - `to_predict_csv_file_location`: CSV to score (relative to `paths.data_dir`).
  - `experiment_pair_background`, `experiment_pair_predict`: Experiment IDs to contextualize background and prediction.
  - `explain`: `true`/`false` to enable explanation export. Note: explanations are currently computed only when `fitness_predictor_architecture` is `CNN`.
  - `use_training_cfg`: If `true`, reuse architecture/training config saved alongside the trained model.
  - `number_of_particles`: MC samples during predictive inference.
  - `save_deterministic_ckpt`, `deterministic_checkpoint_path`: Optional deterministic checkpointing during prediction.

- `model.explainer`:
  - `type`: One of `lrp_captum`, `deeplift`, `integrated_gradients`, `deeplift_shap`, `gradcam_captum`, `saliency`, `random`.
  - Generic knobs:
    - `use_padding_for_baseline`: Whether to pad sequences for baseline handling.
    - `num_model_to_explain`: If `0`, explanations use the mean model; otherwise, number of sampled models.
    - `use_neptune`, `save_images`.
    - `aggregation_mode`: `mean` or `percentile` with `percentile_union` and `percentile_inter`.
    - `epsilon_uai`: Threshold for UAI+ calculation.
  - Method-specific sub-configs (present when applicable):
    - `lrp_captum.rule`: `epsilon` or `alpha_beta` with `epsilon`, `alpha`, `beta`, plus `calculate_metrics`.
    - `gradcam_captum.target_layers`: Conv layer index or list of indices.
    - `deeplift_shap.num_baselines`, number of random examples to define the shap baseline
    - `integrated_gradients.steps`, number of steps in the integral approximation

### Overriding from the CLI (Hydra)
You can override any key inline. Examples:
```bash
# Choose architecture
uv run train_phage_display model.fitness_predictor_architecture=CNN

# Point to a specific training CSV and turn on Neptune
uv run train_phage_display \
  model.training.relative_csv_file_location="data/my_dataset.csv" \
  use_neptune=true neptune_project=MyOrg/MyProject

```

---

## Experiment tracking (optional)
- Neptune can be toggled with `use_neptune` in `config/default.yaml` (default: false).
- To enable, set `use_neptune: true` and export `NEPTUNE_API_TOKEN`.

---

## Training
You can use the installed console script or run via uv. Don't forget to configurate the train/validation dataset before.

```bash
uv run train_phage_display
```

Upon successful completion of training, a new directory will be created in `results/experiments`. This folder will include:

- The best model checkpoint saved as a `.pth` file.
- A `cfg.yaml` file capturing all relevant model architecture and training parameters.

You should reuse this configuration when running the prediction script. If you enable `use_training_cfg`, the prediction script will automatically load `cfg.yaml` from your experiment folder to ensure the architecture and other settings precisely match your trained model. This avoids any issues due to configuration drift between training and inference.

If Neptune logging is enabled (`use_neptune: true`), training progress and results—including metrics, curves, and checkpoints—will be tracked in your Neptune project for easy monitoring and comparison. Otherwise, training and validation performance curves will be saved locally in the `graphs/` directory at the end of training for your review.

## Prediction and Explainability
The prediction script loads the model config (if present in the experiment directory) to ensure architecture compatibility, scores sequences, and optionally saves explainability outputs. If you don't use this option, ensure that the configuration betzeen the model you are loading and the actual one match. Don't forget to configuate the model checkpoint path created in experiments during Training and prediction dataset before.

```bash
uv run predict_phage_display
```
After running prediction and (optionally) explainability analyses, a new results directory will be created under `results/experiments` with a name starting with `predict`. This directory will contain comprehensive prediction and explainability outputs, including:

- `mean_prediction.npy`: Array of the mean predicted values for each input sequence.
- `predictions.csv`: CSV containing per-sequence prediction results, including batch means and relevant metadata.
- `predictions_raw.npy`: All raw predictions for each sequence (no mean-aggregation), enabling accurate standard deviation and uncertainty estimation.
- `sequence.npy`: Array of processed sequence representations used in prediction.
- `std_prediction.npy`: Standard deviations corresponding to each prediction, reflecting model uncertainty.
- `explainer_images/`: Folder containing all visual outputs from the selected explainability method(s), such as saliency maps or attention scores.

These files provide both quantitative results and visual explanations to support further downstream analysis or reporting.


---

## Data expectations
Training and prediction operate on CSV files with sequence columns consistent with the selected experimental pairs. When `model.training.relative_csv_file_location` is null, the path is inferred from `model.splits.relative_csv_file_location` using the first training pair. Otherwise, set `model.training.relative_csv_file_location` to point to your dataset relative to `paths.data_dir`.

---

## Testing

To run the different test, please use the following command line:

```bash
uv run pytest
```

---

## Data accreditation

The public datasets included in this repository originate from the following publication:

**Boyer et al. (2016)** — [PubMed](https://pubmed.ncbi.nlm.nih.gov/26969726/)

Please credit the original authors when using these data in derivative works or publications.

## Citation
If you use this codebase in your research, please cite the associated paper:

```bibtex
@misc{amiaudplachy2026bayespdexploringsequencebinding,
      title={Bayes-PD: Exploring a Sequence to Binding Bayesian Neural Network model trained on Phage Display data},
      author={Ilann Amiaud-Plachy and Michael Blank and Oliver Bent and Sebastien Boyer},
      year={2026},
      eprint={2601.03930},
      archivePrefix={arXiv},
      primaryClass={q-bio.PE},
      url={https://arxiv.org/abs/2601.03930},
}
```

## License
This project is released under the [MIT License](LICENSE).
