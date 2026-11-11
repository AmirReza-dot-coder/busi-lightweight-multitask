"""Frozen BUSI evaluation in the existing Colab notebook; no optimizer or fitting.

Self-contained: load with exec(compile(uploaded_bytes, filename, 'exec'), globals()).
Then call evaluate_busi_final(data=globals().get('cached_data')).
Primary architecture is CNN+Transformer, selected BEFORE test using mean
validation malignant balanced accuracy over seeds 42, 43, 44. The other three
component models and the original full LPRM v1 model are comparison models.
All checkpoints, thresholds and input protocols are frozen before test loading.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import sklearn
from sklearn.metrics import roc_auc_score


@tf.keras.utils.register_keras_serializable(package="BUSI_MTL")
class SpatialPositionEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.positions = self.add_weight(
            name="positions", shape=(1, int(input_shape[1]), int(input_shape[2])),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs+self.positions


@tf.keras.utils.register_keras_serializable(package="BUSI_MTL")
class ProbabilityToLogits(tf.keras.layers.Layer):
    def __init__(self, binary=False, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.binary, self.epsilon = binary, epsilon

    def call(self, inputs):
        p = tf.clip_by_value(inputs, self.epsilon, 1.0-self.epsilon)
        return tf.math.log(p)-tf.math.log1p(-p) if self.binary else tf.math.log(p)

    def get_config(self):
        return {**super().get_config(), "binary":self.binary, "epsilon":self.epsilon}


FINAL_NAMES = ["normal", "benign", "malignant"]
FINAL_SEEDS = [42, 43, 44]
FINAL_VARIANTS = ["cnn_only", "cnn_transformer", "cnn_attention", "cnn_transformer_attention", "full_lprm_v1"]
FINAL_COMPARE_FIELDS = ["val_accuracy", "val_macro_f1", "val_malignant_recall",
    "val_nonmalignant_predicted_malignant", "val_malignant_balanced_accuracy",
    "val_positive_dice", "val_malignant_dice"]


def _final_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def _final_guard(path, value):
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen evaluation settings differ: {path}. Preserve the existing result; do not edit it.")
    else:
        _final_json(path, value)


def _final_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_array_sha(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _final_training_subdir(seed):
    return "training_full_v1" if seed == 42 else f"training_full_seed{seed}_v1"


def _final_metrics(labels, masks, probabilities, prefix="val", extended=False):
    labels = np.asarray(labels, np.int32)
    p = np.asarray(probabilities["classification"], np.float64)
    soft_masks = np.asarray(probabilities["segmentation"])
    if (p.shape != (len(labels),3) or soft_masks.shape != np.shape(masks) or
            not np.isfinite(p).all() or not np.isfinite(soft_masks).all() or
            ((p < 0) | (p > 1)).any() or ((soft_masks < 0) | (soft_masks > 1)).any()):
        raise ValueError("Invalid class or mask probabilities")
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-5, rtol=0)
    p = p/p.sum(1, keepdims=True)
    predicted = p.argmax(1)
    cm = np.zeros((3,3), np.int64)
    np.add.at(cm, (labels,predicted), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0)>0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1)>0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(3), where=precision+recall>0)
    malignant, normal = labels == 2, labels == 0
    false_malignant = int(((predicted == 2) & ~malignant).sum())
    specificity = 1-false_malignant/int((~malignant).sum())
    truth = np.asarray(masks)[...,0] > 0
    binary = soft_masks[...,0] >= 0.5
    true_area, predicted_area = truth.sum((1,2)), binary.sum((1,2))
    intersection = (truth & binary).sum((1,2))
    positive = true_area > 0
    dice, iou = np.full(len(labels),np.nan), np.full(len(labels),np.nan)
    dice[positive] = 2*intersection[positive]/(true_area[positive]+predicted_area[positive])
    iou[positive] = intersection[positive]/(true_area[positive]+predicted_area[positive]-intersection[positive])
    normal_with_mask = int(((predicted == 0) & (predicted_area > 0)).sum())
    lesion_without_mask = int(((predicted != 0) & (predicted_area == 0)).sum())
    pixels = truth.shape[1]*truth.shape[2]
    values = {
        "accuracy":float(tp.sum()/len(labels)), "macro_f1":float(f1.mean()),
        "malignant_recall":float(recall[2]), "missed_malignant":int((predicted[malignant] != 2).sum()),
        "nonmalignant_predicted_malignant":false_malignant,
        "nonmalignant_specificity":float(specificity),
        "malignant_balanced_accuracy":float((recall[2]+specificity)/2),
        "normal_recall":float(recall[0]), "positive_dice":float(dice[positive].mean()),
        "malignant_dice":float(dice[malignant].mean()), "positive_iou":float(iou[positive].mean()),
        "very_low_dice":int((dice[positive] < 0.1).sum()),
        "empty_positive_predictions":int((predicted_area[positive] == 0).sum()),
        "normal_with_any_foreground":int((predicted_area[normal] > 0).sum()),
        "normal_mean_foreground_percent":float(100*predicted_area[normal].mean()/pixels),
        "normal_max_foreground_percent":float(100*predicted_area[normal].max()/pixels),
        "inconsistent_pairs":normal_with_mask+lesion_without_mask,
    }
    if extended:
        confidence, correct = p.max(1), predicted == labels
        bins = np.minimum(np.searchsorted(np.linspace(0,1,16),confidence,side="right")-1,14)
        ece = sum(float((bins == index).mean())*
                  abs(float(correct[bins == index].mean())-float(confidence[bins == index].mean()))
                  for index in range(15) if (bins == index).any())
        values.update({
            "malignant_auc":float(roc_auc_score(malignant.astype(int),p[:,2])),
            "macro_auc_ovr":float(roc_auc_score(labels,p,multi_class="ovr",average="macro",labels=[0,1,2])),
            "ece_15bins":float(ece),
            "brier_multiclass_sum":float(((p-np.eye(3)[labels])**2).sum(1).mean()),
            "negative_log_likelihood":float(-np.log(np.clip(p[np.arange(len(labels)),labels],1e-7,1)).mean()),
        })
    stats = {prefix+"_"+key:value for key,value in values.items()}
    per_class = pd.DataFrame({"class_name":FINAL_NAMES,"precision":precision,
                              "recall":recall,"f1":f1,"support":cm.sum(1)})
    return stats, cm, dice, iou, predicted, predicted_area, binary, p, per_class


def _final_read_split(frame, root, normalization, split):
    mean = np.asarray(normalization["mean"],np.float32)
    std = np.asarray(normalization["std"],np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid saved training normalization")
    images = np.empty((len(frame),224,224,3),np.float32)
    masks = np.empty((len(frame),224,224,1),np.float32)
    for index,row in frame.iterrows():
        with Image.open(root/row["image"]) as image:
            images[index] = np.asarray(image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR))
        with Image.open(root/row["mask"]) as mask:
            masks[index,...,0] = np.asarray(mask.convert("L").resize((224,224),Image.Resampling.NEAREST)) > 0
        if index % 100 == 0:
            print(split, index+1, "/", len(frame))
    labels = frame["class_name"].map(dict(zip(FINAL_NAMES,range(3)))).to_numpy(np.int32)
    return (images-mean)/std, masks, labels


def _final_check_arrays(frame, images, masks, labels):
    expected = frame["class_name"].map(dict(zip(FINAL_NAMES,range(3)))).to_numpy(np.int32)
    if (np.shape(images) != (len(frame),224,224,3) or np.shape(masks) != (len(frame),224,224,1)
            or not np.array_equal(labels,expected) or not np.isfinite(images).all()
            or not np.isin(masks,[0,1]).all()
            or not np.array_equal(np.asarray(masks).sum((1,2,3)) > 0, expected != 0)):
        raise ValueError("Split arrays, labels, mask presence or order do not match the saved manifest")


def _final_prediction(network, images):
    values = network.predict(images,batch_size=8,verbose=0)
    if not isinstance(values,dict) or not {"classification","segmentation"}.issubset(values):
        raise ValueError("Expected named classification and segmentation outputs")
    return values


def _final_paired_bootstrap(frame, directory, repeats=2000):
    labels = frame["class_name"].map(dict(zip(FINAL_NAMES,range(3)))).to_numpy(np.int32)
    groups = frame["group_id"].astype(str).to_numpy()
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    arrays = {}
    for variant in ["cnn_only","cnn_transformer"]:
        tables = [pd.read_csv(directory/variant/f"seed{seed}"/"test_per_image.csv") for seed in FINAL_SEEDS]
        if any(not np.array_equal(table["image"].to_numpy(),frame["image"].to_numpy()) for table in tables):
            raise ValueError("Prediction order differs during paired bootstrap")
        arrays[variant] = (np.stack([table["predicted_label"].to_numpy() for table in tables]),
                           np.stack([table["dice"].to_numpy() for table in tables]))
    def scores(indices, variant):
        predictions, dice = arrays[variant]
        actual = labels[indices] == 2
        selected = predictions[:,indices]
        balanced = ((selected[:,actual] == 2).mean(1)+(selected[:,~actual] != 2).mean(1))/2
        return np.array([balanced.mean(),np.nanmean(dice[:,indices],axis=1).mean()])
    full = np.arange(len(frame))
    estimate = scores(full,"cnn_transformer")-scores(full,"cnn_only")
    rng = np.random.default_rng(20261024)
    samples = []
    for _ in range(repeats*10):
        indices = np.concatenate([group_indices[i] for i in rng.integers(0,len(group_indices),len(group_indices))])
        if not (labels[indices] == 2).any() or not (labels[indices] != 2).any():
            continue
        samples.append(scores(indices,"cnn_transformer")-scores(indices,"cnn_only"))
        if len(samples) == repeats:
            break
    if len(samples) != repeats:
        raise RuntimeError("Not enough valid paired bootstrap samples")
    lower,upper = np.percentile(np.asarray(samples),[2.5,97.5],axis=0)
    return pd.DataFrame({"metric":["test_malignant_balanced_accuracy","test_positive_dice"],
                         "cnn_transformer_minus_cnn":estimate,"lower_95":lower,"upper_95":upper,
                         "resamples":repeats,"similarity_groups":len(group_indices)})


def evaluate_busi_final(data=None, study_dir=None,
                        root="/content/drive/MyDrive/BUSI_MERGED",
                        experiments_root="/content/drive/MyDrive/BUSI_EXPERIMENTS"):
    """Audit validation, freeze all 15 checkpoints, then evaluate test once.

    No training, calibration, threshold search, checkpoint selection on test,
    or seed ensembling. Repeated calls reuse a completed frozen report.
    """
    root,experiments_root = Path(root),Path(experiments_root)
    if any(str(path).startswith("/content/drive/") for path in [root,experiments_root]) and not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    if study_dir is None:
        candidates = sorted(path.parent for path in experiments_root.glob(
            "proposed_model_*/ablations/component_study_v1/study_settings.json"))
        if not candidates:
            raise FileNotFoundError("Saved component study was not found under "+str(experiments_root))
        for index,path in enumerate(candidates):
            print(f"[{index}] {path}")
        answer = "0" if len(candidates) == 1 else input("شمارهٔ پروژهٔ قبلی خودت را انتخاب کن: ").strip()
        if not answer.isdigit() or not 0 <= int(answer) < len(candidates):
            raise ValueError("Invalid study selection")
        study_dir = candidates[int(answer)]
    study_dir = Path(study_dir)
    settings = json.loads((study_dir/"study_settings.json").read_text())
    reference = Path(settings["reference"])
    manifest_path = reference/"split_manifest.csv"
    if settings.get("split_sha256") is not None and _final_sha(manifest_path) != settings["split_sha256"]:
        raise ValueError("The split manifest changed after the component study")
    if (settings.get("source_config_sha256") is not None and
            _final_sha(reference/"source_config.json") != settings["source_config_sha256"]):
        raise ValueError("The saved input configuration changed after the component study")
    manifest = pd.read_csv(manifest_path)
    if (len(manifest) != 761 or not manifest["image"].is_unique or
            manifest["split"].value_counts().to_dict() != {"train":499,"val":115,"test":147} or
            not manifest["group_id"].notna().all() or manifest.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Unexpected grouped split")
    for column in ["image","mask"]:
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in manifest[column]):
            raise ValueError("Dataset paths must be relative")
    normalization = json.loads((reference/"source_config.json").read_text())["normalization"]
    directories = {name:study_dir/name for name in FINAL_VARIANTS[:4]}
    if settings["reuse_old_hybrid_results"]:
        directories["cnn_transformer_attention"] = reference
    directories["full_lprm_v1"] = reference.parent.parent
    checkpoints = []
    for variant,folder in directories.items():
        if _final_sha(folder/"split_manifest.csv") != _final_sha(manifest_path):
            raise ValueError("Model split differs: "+variant)
        other_norm = json.loads((folder/"source_config.json").read_text())["normalization"]
        for key in ["mean","std"]:
            np.testing.assert_allclose(other_norm[key],normalization[key],atol=1e-7,rtol=0)
        for seed in FINAL_SEEDS:
            run = folder/_final_training_subdir(seed)
            for filename in ["completed.json","best.keras","validation_summary.csv","best_validation.json"]:
                if not (run/filename).is_file():
                    raise FileNotFoundError(f"Frozen evaluation requires the completed {variant}, seed={seed}: {run/filename}")
            expected = pd.read_csv(run/"validation_summary.csv").iloc[0]
            metadata = json.loads((run/"best_validation.json").read_text())
            if int(expected["selected_epoch"]) != int(metadata["epoch_number"]):
                raise ValueError("Checkpoint epoch metadata differs: "+str(run))
            checkpoints.append({"variant":variant,"seed":seed,"selected_epoch":int(expected["selected_epoch"]),
                "model":str(run/"best.keras"),"model_sha256":_final_sha(run/"best.keras"),
                "validation":{key:float(expected[key]) for key in FINAL_COMPARE_FIELDS}})
    means = {variant:np.mean([item["validation"]["val_malignant_balanced_accuracy"]
                             for item in checkpoints if item["variant"] == variant]) for variant in FINAL_VARIANTS}
    if max(means,key=means.get) != "cnn_transformer":
        raise ValueError("Saved validation ranking differs from the reported CNN+Transformer choice; inspect before test evaluation")
    directory = study_dir/"final_evaluation_v1"
    directory.mkdir(exist_ok=True)
    protocol = {"version":"1.0.0","primary_model":"cnn_transformer","reference_model":"cnn_only",
        "selection":"mean validation malignant balanced accuracy over seeds 42,43,44",
        "class_order":FINAL_NAMES,"seeds":FINAL_SEEDS,"class_prediction":"argmax",
        "mask_threshold":0.5,"test_augmentation":False,"test_threshold_search":False,
        "checkpoints":checkpoints,"manifest_sha256":_final_sha(manifest_path),
        "normalization":normalization,"resize":"PIL RGB BILINEAR 224; masks NEAREST >0; saved training Z-score",
        "calibration":"unmodified class probabilities; ECE 15 fixed bins; multiclass Brier is sum over 3 classes",
        "bootstrap":"2000 paired similarity-group resamples; three trained models held fixed; no patient-ID claim",
        "test_history_note":"Base-paper reconstruction test results already exist; this freezes the new comparison before this evaluator reads test images",
        "versions":{"tensorflow":tf.__version__,"keras":tf.keras.__version__,"numpy":np.__version__,
                    "sklearn":sklearn.__version__,"python":platform.python_version()}}
    protocol_path = directory/"frozen_protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Frozen model/protocol changed; existing final test results must be preserved")
    if (directory/"completed.json").is_file():
        print("Reusing the completed frozen test report; no new inference or training.")
        return {"directory":directory,"summary":pd.read_csv(directory/"test_summary.csv"),
                "mean_std":pd.read_csv(directory/"test_mean_std.csv",header=[0,1],index_col=0),
                "paired_ci":pd.read_csv(directory/"paired_group_bootstrap.csv")}
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Enable a GPU in Colab before inference")
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if data is not None and all(key in data for key in ["X_val","M_val","y_val"]):
        x_val,m_val,y_val = data["X_val"],data["M_val"],data["y_val"]
    else:
        x_val,m_val,y_val = _final_read_split(val_frame,root,normalization,"Validation")
    _final_check_arrays(val_frame,x_val,m_val,y_val)
    _final_guard(directory/"validation_inputs.json",{"sha256":_final_array_sha(x_val,m_val,y_val)})
    checks = []
    for item in checkpoints:
        print("Checking saved validation:",item["variant"],item["seed"])
        network = tf.keras.models.load_model(item["model"],compile=False)
        actual = _final_metrics(y_val,m_val,_final_prediction(network,x_val))[0]
        for key,expected in item["validation"].items():
            checks.append({"variant":item["variant"],"seed":item["seed"],"metric":key,
                           "saved":expected,"observed":actual[key],
                           "matches":bool(np.isclose(actual[key],expected,atol=1e-5,rtol=0))})
        del network
        gc.collect()
    checks = pd.DataFrame(checks)
    checks.to_csv(directory/"validation_recheck.csv",index=False)
    if not checks["matches"].all():
        print(checks.loc[~checks["matches"]].to_string(index=False))
        raise RuntimeError("Validation does not reproduce the saved report. Test images have NOT been loaded; send the mismatch table.")
    _final_guard(protocol_path,protocol)
    print("All checkpoints frozen. Loading the 147 test images for this final comparison.")
    frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    x_test,m_test,y_test = _final_read_split(frame,root,normalization,"Test")
    _final_check_arrays(frame,x_test,m_test,y_test)
    _final_guard(directory/"test_inputs.json",{"sha256":_final_array_sha(x_test,m_test,y_test)})
    rows = []
    for item in checkpoints:
        target = directory/item["variant"]/f"seed{item['seed']}"
        target.mkdir(parents=True,exist_ok=True)
        if (target/"completed.json").is_file():
            if json.loads((target/"completed.json").read_text()).get("model_sha256") != item["model_sha256"]:
                raise ValueError("Cached inference belongs to a different checkpoint: "+str(target))
            for filename in ["summary.json","test_per_image.csv","test_predictions.npz"]:
                if not (target/filename).is_file():
                    raise FileNotFoundError("Incomplete cached inference: "+str(target/filename))
            stats = json.loads((target/"summary.json").read_text())
            rows.append(stats)
            continue
        print("Final test:",item["variant"],item["seed"])
        network = tf.keras.models.load_model(item["model"],compile=False)
        values = _final_metrics(y_test,m_test,_final_prediction(network,x_test),prefix="test",extended=True)
        stats,cm,dice,iou,predicted,area,binary,p,per_class = values
        stats = {"variant":item["variant"],"seed":item["seed"],"selected_epoch":item["selected_epoch"],**stats}
        table = frame[["image","class_name","group_id"]].copy()
        table["label"],table["predicted_label"] = y_test,predicted
        table["dice"],table["iou"],table["predicted_pixels"] = dice,iou,area
        for index,name in enumerate(FINAL_NAMES):
            table["p_"+name] = p[:,index]
        table.to_csv(target/"test_per_image.csv",index=False)
        pd.DataFrame(cm,index=FINAL_NAMES,columns=FINAL_NAMES).to_csv(target/"test_confusion.csv")
        per_class.to_csv(target/"test_per_class.csv",index=False)
        temporary = target/"test_predictions.tmp.npz"
        np.savez_compressed(temporary,probabilities=p.astype(np.float32),
            mask_bits=np.packbits(binary.reshape(len(frame),-1),axis=1),mask_shape=np.array([224,224]),
            images=frame["image"].to_numpy(dtype=str))
        os.replace(temporary,target/"test_predictions.npz")
        _final_json(target/"summary.json",stats)
        _final_json(target/"completed.json",{"model_sha256":item["model_sha256"]})
        rows.append(stats)
        pd.DataFrame(rows).to_csv(directory/"test_summary_progress.csv",index=False)
        del network,values,binary,p
        gc.collect()
    summary = pd.DataFrame(rows)
    metrics = [column for column in summary if column.startswith("test_")]
    mean_std = summary.groupby("variant")[metrics].agg(["mean","std"])
    paired_ci = _final_paired_bootstrap(frame,directory)
    summary.to_csv(directory/"test_summary.csv",index=False)
    mean_std.to_csv(directory/"test_mean_std.csv")
    paired_ci.to_csv(directory/"paired_group_bootstrap.csv",index=False)
    _final_json(directory/"completed.json",{"primary_model":"cnn_transformer","n_test":len(frame),
        "protocol_sha256":_final_sha(protocol_path),"note":"Do not tune architecture, thresholds or checkpoints using this test report"})
    print("Final report saved:",directory)
    return {"directory":directory,"summary":summary,"mean_std":mean_std,"paired_ci":paired_ci}


print("Final evaluation code loaded. Call evaluate_busi_final(data=globals().get('cached_data')).")
