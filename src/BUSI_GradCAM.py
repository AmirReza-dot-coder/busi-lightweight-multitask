"""BUSI validation-only Grad-CAM in the existing Google Colab notebook.

Load this file, then call:
    cam_result = run_busi_gradcam()

No previous notebook variables, optimizer, fitting or test-image loading.
Uses the frozen seed-42 CNN, CNN+Transformer and full LPRM v1 checkpoints.
Before drawing, reproduces their seven saved validation metrics (21 checks).
Six examples are selected by a fixed seed, with two distinct similarity groups
per class. Each figure shows the image, reference/predicted segmentation, and
separate Grad-CAM maps for normal, benign and malignant classification scores.

Method: standard gradient-channel averaging and positive weighted activation.
The target is the FINAL classifier's pre-softmax score. For a Dense softmax,
the linear score is calculated from its input/kernel/bias without changing it.
All network calls use training=False. Ground-truth masks never enter Grad-CAM.
Each positive CAM is normalized separately; a zero map is reported explicitly.

References:
https://keras.io/examples/vision/grad_cam/
https://arxiv.org/abs/1610.02391
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
import matplotlib.pyplot as plt
import tensorflow as tf


CAM_VERSION = "1.0.0"
CAM_CLASSES = ["normal", "benign", "malignant"]
CAM_MODELS = ["cnn_only", "cnn_transformer", "full_lprm_v1"]
CAM_VAL_FIELDS = ["val_accuracy", "val_macro_f1", "val_malignant_recall",
    "val_nonmalignant_predicted_malignant", "val_malignant_balanced_accuracy",
    "val_positive_dice", "val_malignant_dice"]
CAM_TRAINING_SEED = 42
CAM_SAMPLE_SEED = 20261101
CAM_FEATURE_LAYER = "hybrid_shared_features"


@tf.keras.utils.register_keras_serializable(package="BUSI_MTL")
class SpatialPositionEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.positions = self.add_weight(
            name="positions", shape=(1, int(input_shape[1]), int(input_shape[2])),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.positions


@tf.keras.utils.register_keras_serializable(package="BUSI_MTL")
class ProbabilityToLogits(tf.keras.layers.Layer):
    def __init__(self, binary=False, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.binary, self.epsilon = binary, epsilon

    def call(self, inputs):
        p = tf.clip_by_value(inputs, self.epsilon, 1.0-self.epsilon)
        return tf.math.log(p)-tf.math.log1p(-p) if self.binary else tf.math.log(p)

    def get_config(self):
        return {**super().get_config(), "binary": self.binary, "epsilon": self.epsilon}


def _cam_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cam_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _cam_guard(path, value):
    if path.is_file() and json.loads(path.read_text()) != value:
        raise ValueError("تنظیمات این خروجی با اجرای قبلی فرق دارد؛ خروجی قبلی را حفظ کن: "+str(path))
    if not path.is_file():
        _cam_json(path, value)


def _cam_array_sha(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _cam_select_cases(frame):
    """Fixed selection before inspecting predictions; no test cases or ranking."""
    rng = np.random.default_rng(CAM_SAMPLE_SEED)
    chosen = []
    for name in CAM_CLASSES:
        indices = frame.index[frame["class_name"] == name].to_numpy()
        seen = set()
        for index in rng.permutation(indices):
            group = str(frame.loc[index, "group_id"])
            if group not in seen:
                chosen.append(int(index))
                seen.add(group)
            if len(seen) == 2:
                break
        if len(seen) != 2:
            raise ValueError("دو گروه متمایز در validation پیدا نشد: "+name)
    return chosen


def _cam_load_validation(frame, root, normalization):
    mean, std = [np.asarray(normalization[k], np.float32) for k in ["mean", "std"]]
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Saved normalization is invalid")
    images = np.empty((len(frame),224,224,3),np.float32)
    masks = np.empty((len(frame),224,224,1),np.float32)
    for i,row in frame.iterrows():
        with Image.open(root/row["image"]) as image:
            images[i] = np.asarray(image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR))
        with Image.open(root/row["mask"]) as mask:
            masks[i,...,0] = np.asarray(mask.convert("L").resize((224,224),Image.Resampling.NEAREST)) > 0
        if i % 25 == 0:
            print(f"Validation images and masks: {i+1}/{len(frame)}",flush=True)
    labels = frame["class_name"].map(dict(zip(CAM_CLASSES,range(3)))).to_numpy(np.int32)
    if not np.array_equal(masks.sum((1,2,3)) > 0, labels != 0):
        raise ValueError("Validation mask presence does not match class labels")
    return (images-mean)/std, masks, labels


def _cam_validation_metrics(labels, masks, values):
    if not isinstance(values,dict) or not {"classification","segmentation"}.issubset(values):
        raise ValueError("Expected classification and segmentation dictionary outputs")
    p = np.asarray(values["classification"], np.float64)
    soft = np.asarray(values["segmentation"])
    if p.shape != (len(labels),3) or soft.shape != masks.shape:
        raise ValueError("Unexpected prediction shapes")
    if not np.isfinite(p).all() or not np.isfinite(soft).all() or (p < 0).any() or (p > 1).any() or (soft < 0).any() or (soft > 1).any():
        raise ValueError("Invalid prediction probabilities")
    np.testing.assert_allclose(p.sum(1),1,atol=1e-5,rtol=0)
    pred = p.argmax(1)
    matrix = np.zeros((3,3),np.int64)
    np.add.at(matrix,(labels,pred),1)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp,matrix.sum(0),out=np.zeros(3),where=matrix.sum(0)>0)
    recall = np.divide(tp,matrix.sum(1),out=np.zeros(3),where=matrix.sum(1)>0)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    malignant = labels == 2
    false_malignant = int(((pred == 2) & ~malignant).sum())
    specificity = 1-false_malignant/int((~malignant).sum())
    truth, binary = masks[...,0] > 0, soft[...,0] >= 0.5
    true_area, area = truth.sum((1,2)), binary.sum((1,2))
    intersection = (truth & binary).sum((1,2))
    positive = true_area > 0
    dice = np.full(len(labels),np.nan)
    dice[positive] = 2*intersection[positive]/(true_area[positive]+area[positive])
    return {"val_accuracy":float(tp.sum()/len(labels)), "val_macro_f1":float(f1.mean()),
        "val_malignant_recall":float(recall[2]),
        "val_nonmalignant_predicted_malignant":false_malignant,
        "val_malignant_balanced_accuracy":float((recall[2]+specificity)/2),
        "val_positive_dice":float(dice[positive].mean()),
        "val_malignant_dice":float(dice[malignant].mean())}


def _cam_one(value):
    if isinstance(value,(tuple,list)):
        if len(value) != 1:
            raise ValueError("Expected a single classification tensor")
        return value[0]
    return value


def _cam_logit_source(probability_tensor):
    """Find the final softmax through identity/linear wrappers, without edits."""
    tensor = probability_tensor
    chain = []
    for _ in range(8):
        history = tensor._keras_history
        layer = history.operation if hasattr(history,"operation") else history[0]
        chain.append(layer.name)
        activation = getattr(getattr(layer,"activation",None),"__name__",None)
        if isinstance(layer,tf.keras.layers.Dense) and activation == "softmax":
            return _cam_one(layer.input), layer, "Dense linear score: "+layer.name
        if isinstance(layer,tf.keras.layers.Softmax) or (isinstance(layer,tf.keras.layers.Activation) and activation == "softmax"):
            return _cam_one(layer.input), None, "Pre-softmax tensor: "+layer.name
        identity_type = getattr(tf.keras.layers,"Identity",())
        if (identity_type and isinstance(layer,identity_type)) or (isinstance(layer,tf.keras.layers.Activation) and activation == "linear"):
            tensor = _cam_one(layer.input)
            continue
        break
    raise ValueError("لایهٔ softmax نهایی شناسایی نشد؛ این زنجیره را بفرست: "+" -> ".join(chain))


def _cam_probe(network):
    if not isinstance(network.output,dict):
        raise ValueError("The saved network must have named dictionary outputs")
    try:
        features = network.get_layer(CAM_FEATURE_LAYER).output
    except ValueError as exc:
        candidates = []
        for layer in network.layers:
            try:
                shape = tuple(layer.output.shape)
                if len(shape) == 4:
                    candidates.append((layer.name,shape))
            except (AttributeError,TypeError):
                pass
        raise ValueError("لایهٔ مشترک پیدا نشد. این فهرست را بفرست: "+str(candidates)) from exc
    if isinstance(features,(list,tuple)) or len(features.shape) != 4:
        raise ValueError("Expected a four-dimensional shared feature map")
    source, dense, description = _cam_logit_source(network.output["classification"])
    probe = tf.keras.Model(network.inputs,[features,network.output["classification"],
        network.output["segmentation"],source],name="busi_gradcam_probe")
    return probe,dense,description


def _cam_normalize(signed):
    signed = np.asarray(signed,np.float32)
    if not np.isfinite(signed).all():
        raise FloatingPointError("Non-finite Grad-CAM values")
    positive = np.maximum(signed,0)
    maximum = float(positive.max())
    return (positive/maximum if maximum > 0 else np.zeros_like(positive)), maximum


def _cam_all_classes(probe, dense, image):
    """One image, three pre-softmax targets; no masks/labels used here."""
    x = tf.convert_to_tensor(image[None,...],tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        features,p,seg,source = probe(x,training=False)
        logits = source
        if dense is not None:
            logits = tf.linalg.matmul(source,tf.cast(dense.kernel,source.dtype))
            if dense.use_bias:
                logits = logits+tf.cast(dense.bias,source.dtype)
        # Slicing/reduction must happen INSIDE the tape for all three targets.
        scores = [tf.reduce_sum(logits[:,i]) for i in range(3)]
    np.testing.assert_allclose(tf.nn.softmax(logits,axis=-1).numpy(),p.numpy(),atol=2e-5,rtol=0,
        err_msg="Read-only logits do not reproduce the saved classifier probabilities")
    raw,normalized,info = [],[],[]
    for i,score in enumerate(scores):
        gradients = tape.gradient(score,features)
        if gradients is None:
            raise RuntimeError("اتصال گرادیان به لایهٔ مشترک برقرار نیست؛ جدول preflight را بفرست.")
        tf.debugging.assert_all_finite(gradients,"Non-finite Grad-CAM gradients")
        weights = tf.reduce_mean(gradients,axis=(1,2),keepdims=True)
        signed = tf.reduce_sum(weights*features,axis=-1)[0].numpy()
        cam,maximum = _cam_normalize(signed)
        raw.append(signed);normalized.append(cam)
        info.append({"target_class":CAM_CLASSES[i],"gradient_max_abs":float(tf.reduce_max(tf.abs(gradients)).numpy()),
                     "positive_cam_max":maximum,"no_positive_cam":maximum <= 0})
    del tape
    return np.stack(raw),np.stack(normalized),p.numpy()[0],seg.numpy()[0,...,0],info


def _cam_figure(raw, truth, predicted_mask, heatmaps, probabilities, true_name, model_name, filename, png, pdf):
    predicted = CAM_CLASSES[int(np.argmax(probabilities))]
    fig,axes = plt.subplots(2,3,figsize=(9,5.8))
    axes[0,0].imshow(raw);axes[0,0].set_title("Ultrasound image")
    axes[0,1].imshow(raw)
    if truth.any():
        axes[0,1].imshow(np.ma.masked_where(~truth,truth),cmap="Reds",alpha=.42,vmin=0,vmax=1)
    axes[0,1].set_title("Reference lesion mask" if truth.any() else "Reference: empty mask")
    axes[0,2].imshow(raw)
    if predicted_mask.any():
        axes[0,2].imshow(np.ma.masked_where(~predicted_mask,predicted_mask),cmap="Blues",alpha=.42,vmin=0,vmax=1)
    axes[0,2].set_title("Predicted mask (threshold 0.5)")
    for i,ax in enumerate(axes[1]):
        ax.imshow(raw)
        if heatmaps[i].max() > 0:
            resized = tf.image.resize(heatmaps[i,...,None],(224,224),method="bilinear").numpy()[...,0]
            ax.imshow(resized,cmap="jet",vmin=0,vmax=1,alpha=.42)
            note = ""
        else:
            note = "\nNo positive CAM"
        ax.set_title(f"Grad-CAM: {CAM_CLASSES[i]}\np={probabilities[i]:.3f}"+note,fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{model_name} | truth: {true_name} | prediction: {predicted}\n{filename}",fontsize=10)
    fig.text(.5,.015,"Final pre-softmax targets; each CAM normalized separately. Coarse maps, upsampled for display.",ha="center",fontsize=7.5)
    fig.tight_layout(rect=(0,.04,1,.90))
    for path,fmt in [(png,"png"),(pdf,"pdf")]:
        temporary = path.with_name(path.stem+".tmp"+path.suffix)
        fig.savefig(temporary,format=fmt,dpi=180,bbox_inches="tight",facecolor="white")
        os.replace(temporary,path)
    plt.close(fig)


def run_busi_gradcam(evaluation_dir=None,
        dataset_root="/content/drive/MyDrive/BUSI_MERGED",
        experiments_root="/content/drive/MyDrive/BUSI_EXPERIMENTS"):
    """Create validation-only diagnostic figures from three frozen saved models."""
    dataset_root,experiments_root = Path(dataset_root),Path(experiments_root)
    if any(str(p).startswith("/content/drive/") for p in [dataset_root,experiments_root]) and not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    if evaluation_dir is None:
        found = sorted(experiments_root.glob("proposed_model_*/ablations/component_study_v1/final_evaluation_v1/frozen_protocol.json"))
        if not found:
            raise FileNotFoundError("گزارش ارزیابی نهایی قبلی در Drive پیدا نشد: "+str(experiments_root))
        for i,path in enumerate(found):
            print(f"[{i}] {path.parent}")
        answer = "0" if len(found) == 1 else input("شمارهٔ پروژهٔ قبلی را انتخاب کن: ").strip()
        if not answer.isdigit() or not 0 <= int(answer) < len(found):
            raise ValueError("شمارهٔ پروژه معتبر نیست")
        evaluation_dir = found[int(answer)].parent
    evaluation_dir = Path(evaluation_dir)
    protocol_path = evaluation_dir/"frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    completed = json.loads((evaluation_dir/"completed.json").read_text())
    if _cam_sha(protocol_path) != completed["protocol_sha256"] or protocol["class_order"] != CAM_CLASSES:
        raise ValueError("Frozen evaluation protocol or class order differs")
    study = evaluation_dir.parent
    study_settings = json.loads((study/"study_settings.json").read_text())
    reference = Path(study_settings["reference"])
    manifest_path = reference/"split_manifest.csv"
    if _cam_sha(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("The original fixed split changed")
    manifest = pd.read_csv(manifest_path)
    if (len(manifest) != 761 or not manifest["image"].is_unique or
        manifest["split"].value_counts().to_dict() != {"train":499,"val":115,"test":147} or
        not manifest["group_id"].notna().all() or manifest.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Unexpected fixed group split")
    for column in ["image","mask"]:
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in manifest[column]):
            raise ValueError("Dataset paths must be relative")
    frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if frame["class_name"].value_counts().to_dict() != {"normal":16,"benign":64,"malignant":35}:
        raise ValueError("Unexpected validation class counts")
    cases = _cam_select_cases(frame)
    checkpoints = []
    for variant in CAM_MODELS:
        items = [c for c in protocol["checkpoints"] if c["variant"] == variant and c["seed"] == CAM_TRAINING_SEED]
        if len(items) != 1:
            raise ValueError("Missing or repeated checkpoint: "+variant)
        item = items[0]
        if _cam_sha(item["model"]) != item["model_sha256"]:
            raise ValueError("Saved checkpoint changed: "+item["model"])
        checkpoints.append(item)
    out = study/"gradcam_validation_seed42_v1"
    out.mkdir(exist_ok=True)
    settings = {"version":CAM_VERSION,"protocol_sha256":_cam_sha(protocol_path),
        "dataset_root":str(dataset_root),"manifest_sha256":_cam_sha(manifest_path),
        "models":[{"variant":c["variant"],"sha256":c["model_sha256"],"epoch":c["selected_epoch"]} for c in checkpoints],
        "training_seed":CAM_TRAINING_SEED,"sample_seed":CAM_SAMPLE_SEED,"class_order":CAM_CLASSES,
        "split":"val","images":frame.loc[cases,"image"].tolist(),
        "feature_layer":CAM_FEATURE_LAYER,"score":"final_pre_softmax","normalization":protocol["normalization"],
        "mask_threshold":0.5,"cam_normalization":"positive part divided by maximum, independently for every map",
        "versions":{"tensorflow":tf.__version__,"keras":tf.keras.__version__,"python":platform.python_version()}}
    _cam_guard(out/"settings.json",settings)
    if (out/"completed.json").is_file():
        record = json.loads((out/"completed.json").read_text())
        for relative in record["files"]:
            if not (out/relative).is_file():
                raise FileNotFoundError("Incomplete cached Grad-CAM output: "+str(out/relative))
        print("Grad-CAM قبلاً کامل شده؛ خروجی ذخیره‌شده خوانده شد:",out)
        return {"directory":out,"preflight":pd.read_csv(out/"preflight.csv"),
                "summary":pd.read_csv(out/"gradcam_summary.csv"),"cases":pd.read_csv(out/"cases.csv")}
    print("Loading only the 115 validation images; no training. GPU:",bool(tf.config.list_physical_devices("GPU")),flush=True)
    x,m,y = _cam_load_validation(frame,dataset_root,protocol["normalization"])
    input_sha = _cam_array_sha(x,m,y)
    _cam_guard(out/"validation_inputs.json",{"sha256":input_sha})
    case_table = frame.loc[cases,["image","mask","class_name","group_id"]].copy()
    case_table.insert(0,"validation_index",cases)
    case_table.to_csv(out/"cases.csv",index=False)
    checks,preflight,summary,files = [],[],[],[]
    # First pass: load and validate all three checkpoints before any figures.
    for item in checkpoints:
        print("Preflight:",item["variant"],flush=True)
        network = tf.keras.models.load_model(item["model"],compile=False)
        values = network.predict(x,batch_size=8,verbose=0)
        actual = _cam_validation_metrics(y,m,values)
        for field in CAM_VAL_FIELDS:
            expected = item["validation"][field]
            checks.append({"variant":item["variant"],"metric":field,"expected":expected,
                           "observed":actual[field],"matches":bool(np.isclose(actual[field],expected,atol=1e-5,rtol=0))})
        check_table = pd.DataFrame(checks)
        check_table.to_csv(out/"validation_recheck.csv",index=False)
        if not check_table["matches"].all():
            print(check_table.loc[~check_table["matches"]].to_string(index=False))
            raise RuntimeError("Validation با گزارش قبلی تطابق ندارد؛ شکل‌ها ساخته نشدند. جدول بالا را بفرست.")
        probe,dense,description = _cam_probe(network)
        _,maps,pred,_,gradient_info = _cam_all_classes(probe,dense,x[cases[0]])
        np.testing.assert_allclose(pred,np.asarray(values["classification"])[cases[0]],atol=1e-5,rtol=0)
        preflight.append({"variant":item["variant"],"seed":CAM_TRAINING_SEED,"selected_epoch":item["selected_epoch"],
            "parameters":network.count_params(),"feature_layer":CAM_FEATURE_LAYER,"native_cam_shape":str(tuple(maps.shape[1:])),
            "logit_source":description,"gradient_connected":True,
            "all_target_gradients_zero":all(i["gradient_max_abs"] == 0 for i in gradient_info)})
        pd.DataFrame(preflight).to_csv(out/"preflight.csv",index=False)
        del probe,network,values
        gc.collect()
    print(pd.DataFrame(preflight).to_string(index=False))
    print("All 21 validation comparisons matched. Making six fixed cases per model.",flush=True)
    for item in checkpoints:
        network = tf.keras.models.load_model(item["model"],compile=False)
        probe,dense,_ = _cam_probe(network)
        folder = out/item["variant"]
        folder.mkdir(exist_ok=True)
        for number,index in enumerate(cases,1):
            row = frame.iloc[index]
            print(item["variant"],f"case {number}/6:",row["image"],flush=True)
            signed,cams,p,seg,info = _cam_all_classes(probe,dense,x[index])
            with Image.open(dataset_root/row["image"]) as image:
                raw = np.asarray(image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR))
            stem = f"case_{number:02d}_"+row["class_name"]
            png,pdf,npz = [folder/(stem+suffix) for suffix in [".png",".pdf",".npz"]]
            _cam_figure(raw,m[index,...,0]>0,seg>=0.5,cams,p,row["class_name"],item["variant"],row["image"],png,pdf)
            temp = folder/(stem+".tmp.npz")
            np.savez_compressed(temp,signed_cam=signed,positive_normalized_cam=cams,
                probabilities=p,image=np.array(row["image"]),target_classes=np.array(CAM_CLASSES))
            os.replace(temp,npz)
            files.extend(str(path.relative_to(out)) for path in [png,pdf,npz])
            for i,record in enumerate(info):
                summary.append({"variant":item["variant"],"seed":CAM_TRAINING_SEED,"case":number,
                    "image":row["image"],"true_class":row["class_name"],"predicted_class":CAM_CLASSES[int(p.argmax())],
                    "target_probability":float(p[i]),"native_height":cams.shape[1],"native_width":cams.shape[2],**record})
        del probe,network
        gc.collect()
        if _cam_sha(item["model"]) != item["model_sha256"]:
            raise RuntimeError("Checkpoint file changed during this diagnostic")
    pd.DataFrame(summary).to_csv(out/"gradcam_summary.csv",index=False)
    notes = ("Validation-only Grad-CAM: 6 fixed examples x 3 frozen seed-42 models.\n"
        "CAM targets are the three final pre-softmax class scores, not segmentation pixels.\n"
        "Native grids are retained in NPZ; upsampling is for display.\n"
        "Each positive map is normalized independently; intensity is not comparable across maps.\n"
        "A no-positive-CAM flag denotes an empty positive map, not proof of absent evidence.\n"
        "Reference masks are displayed separately and never used to calculate the CAM.\n"
        "These visualizations are qualitative diagnostics, not proof of clinical correctness.\n"
        "Method: https://keras.io/examples/vision/grad_cam/ ; https://arxiv.org/abs/1610.02391\n")
    (out/"README.txt").write_text(notes,encoding="utf-8")
    files.extend(["settings.json","validation_inputs.json","cases.csv","validation_recheck.csv","preflight.csv","gradcam_summary.csv","README.txt"])
    _cam_json(out/"completed.json",{"version":CAM_VERSION,"files":files,"figures":18,"test_images_loaded":False,"training_performed":False})
    print("Saved:",out)
    try:
        from IPython.display import display, Image as DisplayImage
        for item in checkpoints:
            first = out/item["variant"]/"case_01_normal.png"
            display(DisplayImage(filename=str(first)))
    except ImportError:
        pass
    return {"directory":out,"preflight":pd.DataFrame(preflight),"summary":pd.DataFrame(summary),"cases":case_table}


print("BUSI Grad-CAM code loaded. Run: cam_result = run_busi_gradcam()")
