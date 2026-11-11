"""Train the saved BUSI proposed network in Colab; never load/evaluate test images.

Usage in the existing Colab notebook:
    %run /content/BUSI_Proposed_Training.py
    result = run_proposed_training(proposed_dir, data={
        'X_train': X_train, 'y_train': y_train, 'M_train': M_train,
        'X_val': X_val, 'y_val': y_val, 'M_val': M_val})

The network is a research candidate, not an exact replication of the base paper.
The class-cost ratio 1:1:2 and the checkpoint policy are experimental choices.
Learned task weights are optimization coefficients, not clinical uncertainty.
Pass seed=43 or seed=44 to repeat training from the SAME saved initial weights.
The seed varies training randomness, not the warm-start model initialization.

For the separately saved LPRM v2 architecture, use:
    result = run_lprm_v2_training(v2_dir, data=cached_data, seed=42)
This is a new experiment: three refiner-only warmup epochs, then joint training.
It uses supervised mask presence and an explicit consistency objective.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import platform

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


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


@tf.keras.utils.register_keras_serializable(package="BUSI_Training")
class BoundedLogWeight(tf.keras.constraints.Constraint):
    def __call__(self, value):
        return tf.clip_by_value(value, -3.0, 3.0)

    def get_config(self):
        return {}


def soft_dice_parts(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    true_area = tf.reduce_sum(y_true, axis=(1, 2, 3))
    predicted_area = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    intersection = tf.reduce_sum(y_true*y_pred, axis=(1, 2, 3))
    dice = (2*intersection+1e-6)/(true_area+predicted_area+1e-6)
    return dice, tf.cast(true_area > 0, tf.float32)


def segmentation_loss_per_image(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2))
    dice, positive = soft_dice_parts(y_true, y_pred)
    return bce + positive*(1-dice)


@tf.keras.utils.register_keras_serializable(package="BUSI_Training")
class ProposedTrainer(tf.keras.Model):
    """Explicitly track task weights as trainable MODEL variables, not loss locals."""
    def __init__(self, network, class_weights, adaptive_task_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.class_weights = [float(x) for x in class_weights]
        self.adaptive_task_weights = bool(adaptive_task_weights)
        self.log_class = self.add_weight(
            name="log_class", shape=(), initializer="zeros", trainable=self.adaptive_task_weights,
            constraint=BoundedLogWeight())
        self.log_seg = self.add_weight(
            name="log_seg", shape=(), initializer="zeros", trainable=self.adaptive_task_weights,
            constraint=BoundedLogWeight())
        self.total_metric = tf.keras.metrics.Mean(name="loss")
        self.class_metric = tf.keras.metrics.Mean(name="classification_loss")
        self.plain_ce_metric = tf.keras.metrics.Mean(name="classification_plain_ce")
        self.seg_metric = tf.keras.metrics.Mean(name="segmentation_loss")
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="classification_accuracy")
        self.dice_metric = tf.keras.metrics.Mean(name="positive_soft_dice")

    @property
    def metrics(self):
        return [self.total_metric, self.class_metric, self.plain_ce_metric,
                self.seg_metric, self.accuracy_metric, self.dice_metric]

    def build(self, input_shape):
        if not self.network.built:
            self.network.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def _terms(self, targets, predictions):
        labels = tf.cast(tf.reshape(targets["classification"], [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions["classification"])
        weighted_ce = ce*tf.gather(tf.constant(self.class_weights, tf.float32), labels)
        seg = segmentation_loss_per_image(targets["segmentation"], predictions["segmentation"])
        # Chosen adaptive weighting objective; NOT a clinical uncertainty estimate.
        if self.adaptive_task_weights:
            total = (tf.exp(-self.log_class)*tf.reduce_mean(weighted_ce)+self.log_class
                     + tf.exp(-self.log_seg)*tf.reduce_mean(seg)+self.log_seg)
        else:
            total = tf.reduce_mean(weighted_ce)+tf.reduce_mean(seg)
        if self.losses:
            total += tf.add_n(self.losses)
        tf.debugging.assert_all_finite(total, "Non-finite training objective")
        return total, weighted_ce, ce, seg

    def _update(self, targets, predictions, terms):
        total, weighted_ce, ce, seg = terms
        self.total_metric.update_state(total, sample_weight=tf.cast(tf.shape(ce)[0], tf.float32))
        self.class_metric.update_state(weighted_ce)
        self.plain_ce_metric.update_state(ce)
        self.seg_metric.update_state(seg)
        self.accuracy_metric.update_state(targets["classification"], predictions["classification"])
        dice, positive = soft_dice_parts(targets["segmentation"], predictions["segmentation"])
        self.dice_metric.update_state(dice, sample_weight=positive)
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        images, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("Use the recorded class weights; external sample weights are not supported.")
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            terms = self._terms(targets, predictions)
        gradients = tape.gradient(terms[0], self.trainable_variables)
        pairs = [(gradient, variable) for gradient, variable in
                 zip(gradients, self.trainable_variables) if gradient is not None]
        self.optimizer.apply_gradients(pairs)
        return self._update(targets, predictions, terms)

    def test_step(self, data):
        images, targets, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(images, training=False)
        return self._update(targets, predictions, self._terms(targets, predictions))

    def get_config(self):
        return {**super().get_config(),
                "network": tf.keras.utils.serialize_keras_object(self.network),
                "class_weights": self.class_weights,
                "adaptive_task_weights": self.adaptive_task_weights}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        network = tf.keras.utils.deserialize_keras_object(config.pop("network"))
        return cls(network=network, **config)


def validation_statistics(labels, masks, probabilities):
    """Image-level validation metrics. All class orders: normal, benign, malignant."""
    labels = np.asarray(labels, dtype=np.int32)
    predicted = np.argmax(probabilities["classification"], axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(3), where=matrix.sum(0)>0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(3), where=matrix.sum(1)>0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(3), where=precision+recall>0)
    malignant = labels == 2
    normal = labels == 0
    false_malignant = int(((predicted == 2) & ~malignant).sum())
    specificity = 1-false_malignant/int((~malignant).sum())
    true_masks = np.asarray(masks)[..., 0] > 0
    predicted_masks = np.asarray(probabilities["segmentation"])[..., 0] >= 0.5
    true_area = true_masks.sum((1, 2))
    predicted_area = predicted_masks.sum((1, 2))
    intersection = (true_masks & predicted_masks).sum((1, 2))
    positive = true_area > 0
    dice = np.full(len(labels), np.nan)
    dice[positive] = 2*intersection[positive]/(true_area[positive]+predicted_area[positive])
    stats = {
        "val_accuracy": float(tp.sum()/len(labels)),
        "val_macro_f1": float(f1.mean()),
        "val_malignant_recall": float(recall[2]),
        "val_missed_malignant": int((predicted[malignant] != 2).sum()),
        "val_nonmalignant_predicted_malignant": false_malignant,
        "val_nonmalignant_specificity": float(specificity),
        "val_malignant_balanced_accuracy": float((recall[2]+specificity)/2),
        "val_normal_recall": float(recall[0]),
        "val_positive_dice": float(dice[positive].mean()),
        "val_malignant_dice": float(dice[malignant].mean()),
        "val_very_low_dice": int((dice[positive] < 0.1).sum()),
        "val_empty_positive_predictions": int((predicted_area[positive] == 0).sum()),
        "val_normal_with_any_foreground": int((predicted_area[normal] > 0).sum()),
        "val_normal_mean_foreground_percent": float(
            100*predicted_area[normal].mean()/true_masks.shape[1]/true_masks.shape[2]),
        "val_normal_max_foreground_percent": float(
            100*predicted_area[normal].max()/true_masks.shape[1]/true_masks.shape[2])}
    return stats, matrix, dice, predicted, predicted_area


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    os.replace(tmp, path)


def _save_model(model, path):
    path = Path(path)
    tmp = path.with_name(path.stem+".tmp.keras")
    model.save(str(tmp))
    os.replace(tmp, path)


def _selection_tuple(stats):
    # Predeclared: primary malignant-vs-other balanced accuracy; ties: Dice, macro-F1.
    return (stats["val_malignant_balanced_accuracy"],
            stats["val_positive_dice"], stats["val_macro_f1"])


class ValidationCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, directory, data, patience, initial_epoch):
        super().__init__()
        self.directory, self.data = Path(directory), data
        self.patience, self.wait = patience, 0
        self.initial_epoch = initial_epoch
        self.best = None
        self.progress = self.directory/"progress.json"
        self.best_metadata = self.directory/"best_validation.json"

    def _predict(self):
        probabilities = self.model.network.predict(self.data["X_val"], batch_size=8, verbose=0)
        for values in probabilities.values():
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite validation predictions")
        return validation_statistics(self.data["y_val"], self.data["M_val"], probabilities)[0]

    def _save_best(self, stats, epoch_number):
        _save_model(self.model.network, self.directory/"best.keras")
        _save_model(self.model, self.directory/"best_trainer.keras")
        _write_json(self.best_metadata, {"epoch_number": epoch_number, **stats})
        self.best = _selection_tuple(stats)

    def on_train_begin(self, logs=None):
        if self.best_metadata.exists():
            self.best = _selection_tuple(json.loads(self.best_metadata.read_text()))
        else:
            initial = self._predict()
            self._save_best(initial, self.initial_epoch)
            _write_json(self.directory/"initial_validation.json", initial)
        if self.progress.exists():
            progress = json.loads(self.progress.read_text())
            if progress.get("completed_epochs") == self.initial_epoch:
                self.wait = int(progress.get("wait", 0))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise RuntimeError("Epoch logs missing")
        stats = self._predict()
        logs.update(stats)
        logs["task_weight_classification"] = float(tf.exp(-self.model.log_class).numpy())
        logs["task_weight_segmentation"] = float(tf.exp(-self.model.log_seg).numpy())
        if not np.isfinite(float(logs["loss"])):
            raise FloatingPointError("Non-finite objective")
        score = _selection_tuple(stats)
        if score > self.best:
            self._save_best(stats, epoch+1)
            self.wait = 0
        else:
            self.wait += 1
        record = {"epoch": epoch, **{k: float(v) for k, v in logs.items()}}
        _write_json(self.directory/"epochs"/f"{epoch:04d}.json", record)
        _save_model(self.model, self.directory/"last_trainer.keras")
        _write_json(self.progress, {"completed_epochs": epoch+1, "wait": self.wait})
        print("\nValidation: malignant recall {:.3f}, false malignant {}, Dice {:.3f}, task weights {:.3f}/{:.3f}".format(
            stats["val_malignant_recall"], stats["val_nonmalignant_predicted_malignant"],
            stats["val_positive_dice"], logs["task_weight_classification"], logs["task_weight_segmentation"]))
        if self.wait >= self.patience:
            self.model.stop_training = True
            print("Early stopping: no improvement under the declared selection rule.")


def _read_data(proposed_dir, manifest, root, data):
    frames = {split: manifest[manifest["split"] == split].reset_index(drop=True)
              for split in ["train", "val"]}
    labels = {"normal": 0, "benign": 1, "malignant": 2}
    source = json.loads((proposed_dir/"source_config.json").read_text())
    stats = source["normalization"]
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    if data is None:
        print("Restoring only train/validation images and masks; no previous training is rerun.")
        data = {}
        for split, frame in frames.items():
            images = np.empty((len(frame), 224, 224, 3), np.float32)
            masks = np.empty((len(frame), 224, 224, 1), np.float32)
            for i, row in frame.iterrows():
                with Image.open(root/row["image"]) as image:
                    images[i] = np.asarray(image.convert("RGB").resize((224,224), Image.Resampling.BILINEAR))
                with Image.open(root/row["mask"]) as image:
                    masks[i,...,0] = np.asarray(image.convert("L").resize((224,224), Image.Resampling.NEAREST)) > 0
                if i % 100 == 0:
                    print(split, i+1, "/", len(frame))
            data["X_"+split] = (images-mean)/std
            data["M_"+split] = masks
            data["y_"+split] = frame["class_name"].map(labels).to_numpy(dtype=np.int32)
    else:
        data = dict(data)
        print("Using the train/validation arrays already in memory.")
    for split, frame in frames.items():
        n = len(frame)
        assert data["X_"+split].shape == (n,224,224,3)
        assert data["M_"+split].shape == (n,224,224,1)
        assert np.array_equal(data["y_"+split], frame["class_name"].map(labels).to_numpy())
        assert np.isfinite(data["X_"+split]).all()
        assert np.isin(data["M_"+split], [0,1]).all()
        assert np.array_equal(data["M_"+split].sum((1,2,3)) > 0, data["y_"+split] != 0)
    assert np.allclose(data["X_train"].mean((0,1,2), dtype=np.float64), 0, atol=0.01), "Expected training Z-score inputs."
    return data, frames


def _datasets(data, seed=42):
    rng = tf.random.Generator.from_seed(seed)

    def flip(images, targets):
        flags = (rng.uniform(shape=tf.shape(images)[:1]) < 0.5)[:,None,None,None]
        return tf.where(flags, tf.reverse(images,[2]), images), {
            "classification": targets["classification"],
            "segmentation": tf.where(flags, tf.reverse(targets["segmentation"],[2]), targets["segmentation"])}

    train = tf.data.Dataset.from_tensor_slices((data["X_train"], {
        "classification": data["y_train"], "segmentation": data["M_train"]}))
    train = train.shuffle(len(data["X_train"]), seed=seed, reshuffle_each_iteration=True).batch(8).map(flip, num_parallel_calls=1)
    val = tf.data.Dataset.from_tensor_slices((data["X_val"], {
        "classification": data["y_val"], "segmentation": data["M_val"]})).batch(8)
    return train, val


def _training_subdirectory(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer between 0 and 2**31-1")
    return "training_full_v1" if seed == 42 else f"training_full_seed{int(seed)}_v1"


def run_proposed_training(proposed_dir, data=None,
                          root="/content/drive/MyDrive/BUSI_MERGED", seed=42):
    """One declared experiment; resume checkpoints or reuse a completed result."""
    proposed_dir, root = Path(proposed_dir), Path(root)
    spec_path = proposed_dir/"architecture_spec.json"
    if spec_path.is_file() and json.loads(spec_path.read_text()).get("variant") == "lprm_v2":
        raise ValueError("Use run_lprm_v2_training(v2_dir, data=...) for the three-output LPRM v2 network.")
    initial_path = proposed_dir/"initial.keras"
    assert initial_path.is_file(), f"Initial model missing: {initial_path}"
    directory = proposed_dir/_training_subdirectory(seed)
    seed = int(seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"epochs").mkdir(exist_ok=True)
    if (directory/"completed.json").exists():
        print("Already completed; no training or test evaluation rerun.")
        summary = pd.read_csv(directory/"validation_summary.csv")
        print(summary.to_string(index=False))
        return {"directory": directory, "summary": summary, "data": data}
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Enable a GPU in Colab before training: Runtime > Change runtime type.")
    manifest_bytes = (proposed_dir/"split_manifest.csv").read_bytes()
    manifest = pd.read_csv(proposed_dir/"split_manifest.csv")
    assert manifest["image"].is_unique and len(manifest) == 761
    assert manifest["group_id"].notna().all()
    assert manifest.groupby("group_id")["split"].nunique().max() == 1
    assert manifest["split"].value_counts().to_dict() == {"train":499,"test":147,"val":115}
    assert not any(Path(p).is_absolute() or ".." in Path(p).parts for p in manifest["image"])
    data, frames = _read_data(proposed_dir, manifest, root, data)
    raw_weights = np.array([1.,1.,2.], np.float32)
    class_weights = raw_weights/raw_weights[data["y_train"]].mean()
    settings = {
        "experiment": "proposed_full_stage_A", "source": str(initial_path),
        "source_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "class_order": ["normal","benign","malignant"],
        "class_cost_ratio": [1,1,2], "normalized_class_weights": class_weights.tolist(),
        "adaptive_loss": "exp(-s_cls)*weighted_CE+s_cls+exp(-s_seg)*seg_loss+s_seg",
        "adaptive_log_weight_bounds": [-3,3], "initial_task_weights": [1,1],
        "segmentation_loss": "BCE_all_plus_Dice_on_positive_images",
        "cnn_encoder_trainable": False, "transformer_attention_lprm_heads_trainable": True,
        "optimizer": "Adam", "learning_rate": 1e-4, "clipnorm": 1.0,
        "batch_size": 8, "max_epochs": 30, "patience": 8, "seed": seed,
        "augmentation": "paired_horizontal_flip_probability_0.5",
        "mask_threshold": 0.5,
        "selection_priority": ["malignant_vs_others_balanced_accuracy", "positive_hard_dice", "macro_f1"],
        "selection_note": "prospectively declared for this experiment; previous exploratory runs used another rule",
        "test_evaluated": False,
        "resume_note": "optimizer and task weights resume; random augmentation sequence is not replayed bit-for-bit",
        "versions": {"tensorflow":tf.__version__,"keras":tf.keras.__version__,"python":platform.python_version()}}
    config_path = directory/"config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == settings, "Settings changed: preserve this run and investigate."
    else:
        _write_json(config_path, settings)
    tf.keras.utils.set_random_seed(seed)
    steps = math.ceil(len(data["X_train"])/8)
    last_path = directory/"last_trainer.keras"
    if last_path.exists():
        trainer = tf.keras.models.load_model(str(last_path))
        iterations = int(trainer.optimizer.iterations.numpy())
        assert iterations % steps == 0, "Checkpoint not at epoch boundary"
        initial_epoch = iterations//steps
        records = []
        for epoch in range(initial_epoch):
            records.append(json.loads((directory/"epochs"/f"{epoch:04d}.json").read_text()))
        pd.DataFrame(records).to_csv(directory/"training_log.csv", index=False)
        print("Resuming after completed epoch:", initial_epoch)
    else:
        network = tf.keras.models.load_model(str(initial_path), compile=False)
        # Stage A: new shared Transformer + AG/LPRM/heads train; pretrained CNN stays frozen.
        network.get_layer("shared_encoder").trainable = False
        trainer = ProposedTrainer(network=network, class_weights=class_weights.tolist(), name="proposed_trainer")
        trainer(data["X_train"][:1], training=False)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))
        initial_epoch = 0
    print("Trainable parameters:", sum(int(np.prod(w.shape)) for w in trainer.trainable_weights))
    print("Class weights:", class_weights)
    print("Selection: malignant balanced accuracy; ties: positive Dice, then macro-F1.")
    print("This is a new experimental selection rule; compare later ablations under the same rule.")
    train, val = _datasets(data, seed=seed)
    monitor = ValidationCheckpoints(directory, data, patience=8, initial_epoch=initial_epoch)
    progress = json.loads((directory/"progress.json").read_text()) if (directory/"progress.json").exists() else {}
    already_stopped = progress.get("completed_epochs") == initial_epoch and progress.get("wait",0) >= 8
    if initial_epoch < 30 and not already_stopped:
        trainer.fit(train, validation_data=val, initial_epoch=initial_epoch, epochs=30,
                    callbacks=[monitor, tf.keras.callbacks.CSVLogger(str(directory/"training_log.csv"), append=initial_epoch>0)])
    best = tf.keras.models.load_model(str(directory/"best.keras"), compile=False)
    probabilities = best.predict(data["X_val"], batch_size=8, verbose=0)
    stats, matrix, dice, predicted, predicted_area = validation_statistics(data["y_val"], data["M_val"], probabilities)
    metadata = json.loads((directory/"best_validation.json").read_text())
    summary = pd.DataFrame([{"selected_epoch": metadata["epoch_number"], **stats}])
    summary.to_csv(directory/"validation_summary.csv", index=False)
    names = ["normal","benign","malignant"]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(directory/"validation_confusion.csv")
    per_image = frames["val"][["image","class_name","group_id"]].copy()
    per_image["predicted_class"] = [names[i] for i in predicted]
    per_image["dice"], per_image["predicted_pixels"] = dice, predicted_area
    for i, name in enumerate(names):
        per_image["p_"+name] = probabilities["classification"][:,i]
    per_image.to_csv(directory/"validation_per_image.csv", index=False)
    (directory/"split_manifest.csv").write_bytes(manifest_bytes)
    _write_json(directory/"completed.json", {"selected_epoch":metadata["epoch_number"],"test_evaluated":False})
    print(summary.to_string(index=False))
    print("Saved:", directory)
    if metadata["epoch_number"] == 0:
        print("Initial model retained; do not report this as successful training.")
    return {"directory": directory, "summary": summary, "data": data}


def bernoulli_js_numpy(first, second):
    """Per-image Jensen-Shannon divergence in [0, 1], using Bernoulli variables."""
    first, second = np.broadcast_arrays(np.asarray(first, np.float64),
                                        np.asarray(second, np.float64))
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("Presence probabilities must be finite")
    if ((first < 0) | (first > 1) | (second < 0) | (second > 1)).any():
        raise ValueError("Presence probabilities must be in [0, 1]")
    first, second = np.clip(first, 1e-6, 1-1e-6), np.clip(second, 1e-6, 1-1e-6)
    middle = (first+second)/2
    def kl(value):
        return value*np.log(value/middle)+(1-value)*np.log((1-value)/(1-middle))
    return np.clip((kl(first)+kl(second))/(2*np.log(2)), 0, 1)


def _bernoulli_js_tensor(first, second):
    first = tf.clip_by_value(tf.cast(first, tf.float32), 1e-6, 1-1e-6)
    second = tf.clip_by_value(tf.cast(second, tf.float32), 1e-6, 1-1e-6)
    middle = (first+second)/2
    def kl(value):
        return value*tf.math.log(value/middle)+(1-value)*tf.math.log((1-value)/(1-middle))
    return tf.clip_by_value((kl(first)+kl(second))/(2*tf.math.log(2.0)), 0.0, 1.0)


def hard_consistency_counts(probabilities):
    """Actual argmax class versus thresholded mask; ANY foreground pixel counts."""
    predicted = np.argmax(probabilities["classification"], axis=1)
    nonempty = (np.asarray(probabilities["segmentation"]) >= 0.5).reshape(len(predicted), -1).any(1)
    normal_with_mask = int(((predicted == 0) & nonempty).sum())
    lesion_without_mask = int(((predicted != 0) & ~nonempty).sum())
    return {"val_inconsistent_pairs": normal_with_mask+lesion_without_mask,
            "val_normal_class_with_mask": normal_with_mask,
            "val_lesion_class_without_mask": lesion_without_mask}


def presence_consistency_statistics(masks, probabilities):
    truth = (np.asarray(masks).reshape(len(masks), -1) > 0).any(1)
    mask_presence = np.asarray(probabilities["mask_presence"], dtype=float).reshape(-1)
    class_presence = 1-np.asarray(probabilities["classification"])[:, 0]
    if mask_presence.shape != truth.shape:
        raise ValueError("mask_presence must contain one probability per image")
    divergence = bernoulli_js_numpy(class_presence, mask_presence)
    return {**hard_consistency_counts(probabilities),
            "val_presence_accuracy": float(((mask_presence >= 0.5) == truth).mean()),
            "val_presence_brier": float(((mask_presence-truth.astype(float))**2).mean()),
            "val_consistency_proxy_js": float(divergence.mean())}


@tf.keras.utils.register_keras_serializable(package="BUSI_Training")
class LPRMV2Trainer(ProposedTrainer):
    """Main supervised tasks plus mask-presence supervision and soft consistency.

    The presence head is a learned proxy. Its agreement is reported separately
    from the actual class/mask contradictions counted after thresholding.
    """
    def __init__(self, network, class_weights, presence_weights,
                 presence_factor=0.2, consistency_factor=0.1,
                 adaptive_task_weights=True, **kwargs):
        super().__init__(network, class_weights,
                         adaptive_task_weights=adaptive_task_weights, **kwargs)
        self.presence_weights = [float(x) for x in presence_weights]
        self.presence_factor = float(presence_factor)
        self.consistency_factor = float(consistency_factor)
        if (len(self.presence_weights) != 2 or
                not all(np.isfinite(x) and x > 0 for x in self.presence_weights)):
            raise ValueError("Expected two positive presence weights: absent, present")
        if not all(np.isfinite(x) and x >= 0 for x in
                   [self.presence_factor, self.consistency_factor]):
            raise ValueError("Loss factors must be finite and nonnegative")
        self.presence_metric = tf.keras.metrics.Mean(name="presence_loss")
        self.consistency_metric = tf.keras.metrics.Mean(name="consistency_js")

    @property
    def metrics(self):
        return super().metrics + [getattr(self, name) for name in
            ["presence_metric", "consistency_metric"] if hasattr(self, name)]

    def _terms(self, targets, predictions):
        main, weighted_ce, plain_ce, seg = super()._terms(targets, predictions)
        truth = tf.cast(tf.reduce_any(targets["segmentation"] > 0, axis=(1, 2, 3)), tf.float32)
        presence = tf.cast(tf.reshape(predictions["mask_presence"], [-1]), tf.float32)
        auxiliary = tf.keras.losses.binary_crossentropy(truth[:, None], presence[:, None])
        auxiliary *= tf.gather(tf.constant(self.presence_weights, tf.float32), tf.cast(truth, tf.int32))
        divergence = _bernoulli_js_tensor(1-predictions["classification"][:, 0], presence)
        total = (main + self.presence_factor*tf.reduce_mean(auxiliary)
                 + self.consistency_factor*tf.reduce_mean(divergence))
        tf.debugging.assert_all_finite(total, "Non-finite LPRM v2 objective")
        return total, weighted_ce, plain_ce, seg, auxiliary, divergence

    def _update(self, targets, predictions, terms):
        super()._update(targets, predictions, terms[:4])
        self.presence_metric.update_state(terms[4])
        self.consistency_metric.update_state(terms[5])
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        return {**super().get_config(), "presence_weights": self.presence_weights,
                "presence_factor": self.presence_factor,
                "consistency_factor": self.consistency_factor}


def _v2_validation(network, data):
    probabilities = network.predict(data["X_val"], batch_size=8, verbose=0)
    if not isinstance(probabilities, dict) or not {
            "classification", "segmentation", "mask_presence"}.issubset(probabilities):
        raise ValueError("Expected the saved LPRM v2 model with three named outputs")
    if not all(np.isfinite(value).all() for value in probabilities.values()):
        raise FloatingPointError("Non-finite LPRM v2 validation predictions")
    stats = validation_statistics(data["y_val"], data["M_val"], probabilities)[0]
    stats.update(presence_consistency_statistics(data["M_val"], probabilities))
    return probabilities, stats


class LPRMV2Checkpoints(ValidationCheckpoints):
    def _predict(self):
        return _v2_validation(self.model.network, self.data)[1]

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print("Actual class/mask contradictions: {}; presence accuracy: {:.3f}; proxy JS: {:.5f}".format(
            int(logs["val_inconsistent_pairs"]), logs["val_presence_accuracy"],
            logs["val_consistency_proxy_js"]))


def _configure_v2_trainability(network, phase):
    network.trainable = True
    if phase == "warmup":
        for layer in network.layers:
            layer.trainable = layer.name.startswith("lprm_v2_") or layer.name == "mask_presence"
        if not network.get_layer("lprm_v2_mask_encoder").trainable:
            raise AssertionError("The v2 mask encoder must be trainable")
    elif phase == "joint":
        network.get_layer("shared_encoder").trainable = False
    else:
        raise ValueError("Unknown training phase")
    if network.get_layer("shared_encoder").trainable:
        raise AssertionError("The pretrained CNN encoder must remain frozen")


def _guard_settings(path, settings):
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Saved settings differ: {path}. Preserve this run and investigate.")
    else:
        _write_json(path, settings)


def _run_v2_phase(directory, data, seed, settings, network_factory):
    """Resume at epoch boundaries; a resumed random sequence need not be identical."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"epochs").mkdir(exist_ok=True)
    _guard_settings(directory/"config.json", settings)
    if (directory/"phase_completed.json").exists():
        for name in ["best.keras", "last_trainer.keras", "best_validation.json"]:
            if not (directory/name).is_file():
                raise FileNotFoundError(directory/name)
        print("Reusing completed phase:", settings["phase"])
        return
    steps = math.ceil(len(data["X_train"])/8)
    last_path = directory/"last_trainer.keras"
    if last_path.exists():
        trainer = tf.keras.models.load_model(str(last_path))
        if not isinstance(trainer, LPRMV2Trainer):
            raise TypeError("The resume checkpoint is not an LPRMV2Trainer")
        iterations = int(trainer.optimizer.iterations.numpy())
        if iterations % steps:
            raise ValueError("Resume checkpoint is not at an epoch boundary")
        initial_epoch = iterations//steps
        records = [json.loads((directory/"epochs"/f"{i:04d}.json").read_text())
                   for i in range(initial_epoch)]
        pd.DataFrame(records).to_csv(directory/"training_log.csv", index=False)
        print("Resuming", settings["phase"], "after epoch", initial_epoch)
    else:
        network = network_factory()
        _configure_v2_trainability(network, settings["phase"])
        trainer = LPRMV2Trainer(
            network=network, class_weights=settings["class_weights"],
            presence_weights=settings["presence_weights"],
            presence_factor=settings["presence_factor"],
            consistency_factor=settings["consistency_factor"],
            adaptive_task_weights=settings["adaptive_task_weights"],
            name="lprm_v2_"+settings["phase"]+"_trainer")
        trainer(data["X_train"][:1], training=False)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings["learning_rate"], clipnorm=1.0))
        initial_epoch = 0
    if initial_epoch > settings["max_epochs"]:
        raise ValueError("Checkpoint exceeds the declared epoch limit")
    print("Phase:", settings["phase"], "| Trainable parameters:",
          sum(int(np.prod(w.shape)) for w in trainer.trainable_weights))
    print("Presence factor:", settings["presence_factor"],
          "| Consistency factor:", settings["consistency_factor"])
    progress_path = directory/"progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    already_stopped = (progress.get("completed_epochs") == initial_epoch and
                       progress.get("wait", 0) >= settings["patience"])
    if initial_epoch < settings["max_epochs"] and not already_stopped:
        train, val = _datasets(data, seed=seed)
        monitor = LPRMV2Checkpoints(directory, data, settings["patience"], initial_epoch)
        trainer.fit(train, validation_data=val, initial_epoch=initial_epoch,
                    epochs=settings["max_epochs"], callbacks=[monitor,
                        tf.keras.callbacks.CSVLogger(str(directory/"training_log.csv"),
                                                     append=initial_epoch > 0)])
    if not (directory/"best_validation.json").is_file() or not last_path.is_file():
        raise FileNotFoundError("The phase did not produce complete checkpoints")
    completed_epochs = int(trainer.optimizer.iterations.numpy())//steps
    _write_json(directory/"phase_completed.json", {
        "phase": settings["phase"], "completed_epochs": completed_epochs,
        "last_trainer_sha256": hashlib.sha256(last_path.read_bytes()).hexdigest(),
        "test_evaluated": False})


def _v2_refinement_diagnostic(network, data, directory):
    """Compare coarse and final predictions WITHIN the same selected network."""
    probe = tf.keras.Model(network.inputs, {
        "coarse_class": network.get_layer("classification_coarse").output,
        "coarse_mask": network.get_layer("segmentation_coarse").output,
        "final_class": network.get_layer("classification").output,
        "final_mask": network.get_layer("segmentation").output})
    values = probe.predict(data["X_val"], batch_size=8, verbose=0)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Non-finite refinement diagnostic")
    rows = []
    for stage in ["coarse", "final"]:
        probabilities = {"classification": values[stage+"_class"],
                         "segmentation": values[stage+"_mask"]}
        stats = validation_statistics(data["y_val"], data["M_val"], probabilities)[0]
        rows.append({"stage": stage, **hard_consistency_counts(probabilities), **stats})
    stages = pd.DataFrame(rows)
    stages.to_csv(directory/"coarse_vs_final.csv", index=False)
    before = np.argmax(values["coarse_class"], axis=1)
    after = np.argmax(values["final_class"], axis=1)
    truth = data["y_val"]
    changes = pd.DataFrame([{
        "mean_class_probability_change": float(np.abs(values["final_class"]-values["coarse_class"]).mean()),
        "mean_mask_probability_change": float(np.abs(values["final_mask"]-values["coarse_mask"]).mean()),
        "changed_class_predictions": int((before != after).sum()),
        "wrong_to_correct": int(((before != truth) & (after == truth)).sum()),
        "correct_to_wrong": int(((before == truth) & (after != truth)).sum())}])
    changes.to_csv(directory/"refinement_changes.csv", index=False)
    return stages, changes


def run_lprm_v2_training(v2_dir, data=None,
                         root="/content/drive/MyDrive/BUSI_MERGED", seed=42):
    """Train the saved three-output v2 model; never load or evaluate test images.

    Warmup: three epochs, LPRM parameters only, LR 1e-3, fixed task weights,
    supervised presence loss 0.2, consistency disabled while the proxy learns.
    Joint: at most 30 epochs, LR 1e-4, pretrained CNN frozen, adaptive main
    task weights, supervised presence loss 0.2 and Bernoulli JS loss 0.1.
    These coefficients define an exploratory experiment, not the base paper.
    """
    _training_subdirectory(seed)  # Reuse strict seed validation, not the v1 path.
    seed = int(seed)
    v2_dir, root = Path(v2_dir), Path(root)
    initial_path = v2_dir/"initial.keras"
    if not initial_path.is_file():
        raise FileNotFoundError(f"Saved LPRM v2 initial model missing: {initial_path}")
    directory = v2_dir/f"training_v2_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"completed.json").exists():
        print("LPRM v2 already completed; training and test evaluation are not rerun.")
        return {"directory": directory,
                "summary": pd.read_csv(directory/"validation_summary.csv"),
                "diagnostic": pd.read_csv(directory/"coarse_vs_final.csv"),
                "changes": pd.read_csv(directory/"refinement_changes.csv"), "data": data}
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Enable a GPU in Colab: Runtime > Change runtime type.")
    manifest_path = v2_dir/"split_manifest.csv"
    manifest_bytes = manifest_path.read_bytes()
    manifest = pd.read_csv(manifest_path)
    if (len(manifest) != 761 or not manifest["image"].is_unique or
        manifest["split"].value_counts().to_dict() != {"train":499,"val":115,"test":147} or
        not manifest["group_id"].notna().all() or
        manifest.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("The saved split differs from the agreed 761-image grouped split")
    for column in ["image", "mask"]:
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in manifest[column]):
            raise ValueError("Expected relative dataset paths in the manifest")
    data, frames = _read_data(v2_dir, manifest, root, data)
    class_raw = np.array([1., 1., 2.], np.float32)
    class_weights = (class_raw/class_raw[data["y_train"]].mean()).tolist()
    present = data["M_train"].sum((1, 2, 3)) > 0
    counts = np.bincount(present.astype(int), minlength=2)
    if (counts == 0).any():
        raise ValueError("Presence supervision requires both normal and lesion training images")
    presence_weights = (len(present)/(2*counts)).tolist()
    settings = {
        "experiment": "lprm_v2_presence_consistency_20261018",
        "source": str(initial_path),
        "source_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_config_sha256": hashlib.sha256((v2_dir/"source_config.json").read_bytes()).hexdigest(),
        "class_order": ["normal", "benign", "malignant"],
        "class_cost_ratio": [1, 1, 2], "class_weights": class_weights,
        "presence_order": ["absent", "present"], "presence_weights": presence_weights,
        "presence_target": "any ground-truth foreground pixel; training split only for weights",
        "presence_factor": 0.2, "joint_consistency_factor": 0.1,
        "consistency": "Bernoulli JS(1-p_normal, learned mask_presence)/log(2)",
        "consistency_note": "soft proxy; actual hard class/mask contradictions reported separately",
        "segmentation_loss": "BCE_all_plus_Dice_on_positive_images",
        "adaptive_loss": "exp(-s_cls)*weighted_CE+s_cls+exp(-s_seg)*seg_loss+s_seg",
        "adaptive_log_weight_bounds": [-3, 3],
        "warmup": {"epochs": 3, "learning_rate": 1e-3, "consistency_factor": 0.0,
                   "adaptive_task_weights": False, "trainable": "LPRM v2 and presence head only"},
        "joint": {"max_epochs": 30, "learning_rate": 1e-4, "patience": 8,
                  "adaptive_task_weights": True, "trainable": "all except shared CNN encoder"},
        "joint_initialization": "last warmup checkpoint; new Adam and log task weights reset to zero",
        "optimizer": "Adam", "clipnorm": 1.0, "batch_size": 8,
        "augmentation": "paired_horizontal_flip_probability_0.5", "seed": seed,
        "mask_threshold": 0.5, "hard_presence_definition": "any predicted foreground pixel",
        "selection_priority": ["malignant_vs_others_balanced_accuracy", "positive_hard_dice", "macro_f1"],
        "selection_candidates": "initial model, warmup checkpoints, joint checkpoints",
        "test_evaluated": False,
        "resume_note": "optimizer resumes within each phase; random sequence not replayed bit-for-bit",
        "versions": {"tensorflow": tf.__version__, "keras": tf.keras.__version__,
                     "python": platform.python_version()}}
    _guard_settings(directory/"config.json", settings)
    tf.keras.utils.set_random_seed(seed)
    common = {"class_weights": class_weights, "presence_weights": presence_weights,
              "presence_factor": 0.2, "batch_size": 8, "seed": seed}
    warmup_dir, joint_dir = directory/"warmup", directory/"joint"
    def warmup_network():
        network = tf.keras.models.load_model(str(initial_path), compile=False)
        if not isinstance(network.output, dict) or set(network.output) != {
                "classification", "segmentation", "mask_presence"}:
            raise ValueError("Choose lprm_v2/initial.keras, which has three named outputs")
        return network
    _run_v2_phase(warmup_dir, data, seed, {
        **common, "phase": "warmup", "max_epochs": 3, "patience": 4,
        "learning_rate": 1e-3, "adaptive_task_weights": False, "consistency_factor": 0.0},
        warmup_network)
    def joint_network():
        warmup_trainer = tf.keras.models.load_model(str(warmup_dir/"last_trainer.keras"), compile=False)
        return warmup_trainer.network
    _run_v2_phase(joint_dir, data, seed, {
        **common, "phase": "joint", "max_epochs": 30, "patience": 8,
        "learning_rate": 1e-4, "adaptive_task_weights": True, "consistency_factor": 0.1},
        joint_network)
    candidates = [(phase, phase_dir, json.loads((phase_dir/"best_validation.json").read_text()))
                  for phase, phase_dir in [("warmup", warmup_dir), ("joint", joint_dir)]]
    selected_phase, selected_dir, metadata = max(candidates, key=lambda item: _selection_tuple(item[2]))
    best = tf.keras.models.load_model(str(selected_dir/"best.keras"), compile=False)
    probabilities, stats = _v2_validation(best, data)
    _, matrix, dice, predicted, predicted_area = validation_statistics(data["y_val"], data["M_val"], probabilities)
    summary = pd.DataFrame([{"selected_stage": selected_phase,
                             "selected_phase_epoch": metadata["epoch_number"], **stats}])
    summary.to_csv(directory/"validation_summary.csv", index=False)
    names = ["normal", "benign", "malignant"]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(directory/"validation_confusion.csv")
    per_image = frames["val"][["image", "class_name", "group_id"]].copy()
    per_image["predicted_class"] = [names[i] for i in predicted]
    per_image["dice"], per_image["predicted_pixels"] = dice, predicted_area
    per_image["mask_presence_probability"] = probabilities["mask_presence"].reshape(-1)
    for i, name in enumerate(names):
        per_image["p_"+name] = probabilities["classification"][:, i]
    per_image.to_csv(directory/"validation_per_image.csv", index=False)
    diagnostic, changes = _v2_refinement_diagnostic(best, data, directory)
    _save_model(best, directory/"best.keras")
    (directory/"split_manifest.csv").write_bytes(manifest_bytes)
    _write_json(directory/"selected_checkpoint.json", {"stage": selected_phase, **metadata})
    _write_json(directory/"completed.json", {
        "selected_stage": selected_phase, "selected_phase_epoch": metadata["epoch_number"],
        "test_evaluated": False})
    print(summary.to_string(index=False))
    print("Saved:", directory)
    if selected_phase == "warmup" and metadata["epoch_number"] == 0:
        print("Initial model retained: training did not improve the declared selection metrics.")
    print("Compare coarse_vs_final.csv and refinement_changes.csv before judging LPRM success.")
    return {"directory": directory, "summary": summary, "diagnostic": diagnostic,
            "changes": changes, "data": data}


print("BUSI training functions loaded: run_proposed_training(...) or run_lprm_v2_training(...).")
