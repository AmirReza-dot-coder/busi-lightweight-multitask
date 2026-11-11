# Execution and dependencies

## Offline review

Python plus `requirements-review.txt` is enough for `tools/rebuild_review.py`. It reads the six original result archives, checks split membership/group separation, recomputes class and hard-mask metrics, and builds plots. The first packaged run passed 288 numerical comparisons at absolute tolerance 1e-6. It does not load `.keras` files or use TensorFlow. Outputs default to ignored `review_generated/`.

## Original training and inference in Colab

Use the original notebooks in a Colab GPU runtime. Review the recorded environment before changing dependencies. The source files preserve their original Drive paths and guards.

| Entry point | Prerequisites and scope |
|---|---|
| `notebooks/01_BUSI_Data_Preparation_Clean.ipynb` | BUSI images/masks, reference split, and the duplicate-reference PDF obtained by the notebook. Creates new audit outputs and checks previous membership. Existing training uses the saved split; it does not require rerunning curation. |
| `notebooks/BUSI_Paper_Settings_Standalone.ipynb` | BUSI and the fixed split. `ALLOW_DOCUMENTED_RECONSTRUCTION=False` is intentional. Read the documented assumptions before enabling reconstruction. This entry point covers the reconstructed classifier, not the final shared model. |
| `src/BUSI_Proposed_Training.py` | Saved historical proposed-model folder with `initial.keras`, `source_config.json`, and `split_manifest.csv`. Loading defines functions; call `run_proposed_training(proposed_dir, ...)`. It does not construct the original warm-start weights from raw data. |
| `notebooks/BUSI_Component_Ablation.ipynb` | Historical `ablations/without_lprm/initial.keras` plus matching source config and split. Contains the training engine and component helpers. Three seeds share the recorded BUSI-trained starting state. |
| `src/BUSI_Final_Evaluation.py` | Completed component/full-model training directories, recorded selected weights and BUSI. Call `evaluate_busi_final(...)`. It evaluates frozen models; no fitting. |
| `notebooks/BUSI_Single_vs_Multi.ipynb` | Original final-evaluation folder with `frozen_protocol.json` and `completed.json`, source manifest/configuration, BUSI, and ImageNet MobileNetV2 weights. The core network is freshly initialized except for the frozen ImageNet CNN. The notebook still uses earlier protocol metadata as an input guard. |
| `notebooks/BUSI_Cost_Loss_Ablation.ipynb` | Original final-evaluation metadata, original shared `initial.keras`, matching configs/split and BUSI. Runs paired validation-only cost-loss follow-ups. |
| `notebooks/BUSI_Validation_Repair.ipynb` | Original checkpoints/validation reports and historical starting weights. Runs the augmentation comparison, not a new test study. |
| `notebooks/BUSI_GradCAM_Standalone.ipynb` | Frozen seed-42 checkpoints and original protocol/source metadata. Rechecks validation scores before plotting selected validation cases. |
| `exploratory/BUSI_LPRM_V2_Continue.ipynb` | Existing LPRM-v2 folder and its separately saved initial model. This is not the final LPRM-v1 study. |
| `exploratory/BUSI_Targeted_Finetune.ipynb` | Original full-model controls and recorded protocol. Follow-up only; no corresponding result archive was supplied in this package. |

Defaults include `/content/drive/MyDrive/BUSI_MERGED` and `/content/drive/MyDrive/BUSI_EXPERIMENTS/proposed_model_20261018T141244_200658Z`. Exact expected model paths/hashes are in `data/required_checkpoints.json`. Code accepts explicit directories in several entry points; inspect its signature rather than rewriting historical configuration files.

Training and model inference were not rerun during packaging. The current source files are retained as supplied, rather than retroactively rewriting the experiment or pretending to have the exact source snapshot for every historical execution.

## Checkpoint handoff

`notebooks/Export_Advisor_Checkpoints.ipynb` is a new packaging helper. It mounts the owner's Drive, accepts `tools/export_checkpoints.py` and `data/required_checkpoints.json`, and copies hash-matching source models to `MyDrive/BUSI_ADVISOR_CHECKPOINTS`. It deduplicates identical model bytes and reports missing/mismatched files without training. Share that directory separately after inspection.

The inventory covers selected models and the principal historical full-model initial weights. It is not a substitute for backing up the original experiment directory: component-specific initial models, optimizer-resume state, and adjacent original metadata may also be required to replay training.

## Environment records

`environments/closure_environment.json` records Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.4.6 and pandas 2.2.3, with Tesla T4/CUDA 12.5.1/cuDNN 9. `environments/single_vs_multi_versions.json` retains that study's versions. Cost and augmentation archives record NumPy 2.1.3. They must not be described as one identical historical environment.

No training dependency installation was tested here. Runtime upgrades can alter numerics; recorded one-pixel historical differences remain disclosed.
