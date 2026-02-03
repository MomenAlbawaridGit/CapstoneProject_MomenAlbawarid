import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import cv2
from PIL import Image


class ScoreCAMBrainTumorXAI:
    def __init__(
        self,
        model_path=os.path.join("models", "mobilenet_nd5_final_finetuned_cw_v2.h5"),
        class_names=None,
        img_size=(224, 224),
        contrast_factor=2.0,
        eps=1e-8,
        verbose=False,
    ):
        self.model_path = model_path
        self.img_size = tuple(img_size)
        self.contrast_factor = float(contrast_factor)
        self.eps = float(eps)
        self.verbose = bool(verbose)

        if class_names is None:
            class_names = [
                "glioma_tumor",
                "meningioma_tumor",
                "no_tumor",
                "pituitary_tumor",
            ]
        self.class_names = list(class_names)

        # Load model once
        if self.verbose:
            print("Loading model from:", self.model_path)
        self.loaded_model = tf.keras.models.load_model(self.model_path)

        # Wrapper model (optional but matches your setup)
        wrapped_input = tf.keras.Input(shape=(224, 224, 3), name="wrapped_input")
        wrapped_output = self.loaded_model(wrapped_input, training=False)
        self.model = tf.keras.Model(inputs=wrapped_input, outputs=wrapped_output)

        # Build Score-CAM model once
        self.scorecam_model = self._build_scorecam_model(self.loaded_model)

    # ----------------------------
    # Preprocessing (match training)
    # ----------------------------
    def preprocess_image(self, path: str) -> np.ndarray:
        """
        EXACTLY match training pipeline:
          - resize 224x224 (bilinear)
          - [0,255] -> [0,1]
          - tf.image.adjust_contrast(image, 2.0)  (contrast around mean)
          - clip [0,1]
          - back to [0,255]
          - preprocess_input -> [-1,1]
          - add batch dim
        """
        img = load_img(path, target_size=self.img_size, interpolation="bilinear")  # important
        img_array = img_to_array(img).astype(np.float32)  # (H,W,3) in [0,255]

        # Use TF ops to match training behavior exactly
        image = tf.convert_to_tensor(img_array, dtype=tf.float32)
        image = image / 255.0

        image = tf.image.adjust_contrast(image, self.contrast_factor)
        image = tf.clip_by_value(image, 0.0, 1.0)

        image = image * 255.0
        image = preprocess_input(image)  # MobileNetV2 => [-1,1]

        image = tf.expand_dims(image, axis=0)  # (1,224,224,3)
        return image.numpy()

    # ----------------------------
    # Prediction
    # ----------------------------
    def predict(self, image_path: str):
        x = self.preprocess_image(image_path)
        preds = self.model.predict(x, verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = preds[0] if preds.ndim == 2 else preds  # (num_classes,)

        pred_index = int(np.argmax(preds))
        pred_label = self.class_names[pred_index]
        confidence = float(preds[pred_index])
        return pred_label, confidence, pred_index

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _pick_4d_tensor(x):
        if isinstance(x, (list, tuple)):
            for t in x:
                if hasattr(t, "shape") and len(t.shape) == 4:
                    return t
            for t in x:
                if hasattr(t, "shape"):
                    return t
            return None
        return x

    def _ensure_probs(self, preds: tf.Tensor) -> tf.Tensor:
        p0 = preds[0]
        minv = tf.reduce_min(p0)
        maxv = tf.reduce_max(p0)
        s = tf.reduce_sum(p0)
        if (minv >= -1e-3) and (maxv <= 1.0 + 1e-3) and (tf.abs(s - 1.0) < 1e-2):
            return preds
        return tf.nn.softmax(preds, axis=-1)

    def _find_backbone_model_layer(self, base_model: tf.keras.Model):
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.Model):
                return layer
        return None

    def _build_scorecam_model(self, base_model: tf.keras.Model) -> tf.keras.Model:
        inp = base_model.inputs[0]
        pred_out = base_model.outputs[0] if isinstance(base_model.outputs, (list, tuple)) else base_model.output

        backbone = self._find_backbone_model_layer(base_model)
        if backbone is None:
            raise ValueError("No nested backbone Model layer found (unexpected for MobileNetV2-based models).")

        feat_raw = backbone(inp)  # may be Tensor OR list/tuple
        feat = self._pick_4d_tensor(feat_raw)
        if feat is None or len(feat.shape) != 4:
            raise ValueError(f"Backbone returned no usable 4D feature tensor. Got: {type(feat_raw)}")

        if self.verbose:
            print(f"[Score-CAM] Using backbone: {backbone.name}, feature tensor shape={feat.shape}")

        sc_model = tf.keras.Model(inputs=inp, outputs=[feat, pred_out])
        _ = sc_model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)  # sanity build
        return sc_model

    # ----------------------------
    # Overlay (JET)
    # ----------------------------
    @staticmethod
    def overlay_heatmap_jet(orig_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        orig_rgb = orig_rgb.astype(np.uint8)
        h, w = orig_rgb.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig_rgb, 1.0, heatmap_rgb, alpha, 0)
        return overlay

    # ----------------------------
    # Score-CAM computation (your stable version)
    # ----------------------------
    def compute_scorecam(self, image_path: str, class_index: int = None, max_channels: int = 32):
        # original image for overlay (224x224)
        img = load_img(image_path, target_size=self.img_size)
        orig_rgb = img_to_array(img).astype(np.uint8)

        x_np = self.preprocess_image(image_path)
        x = tf.convert_to_tensor(x_np, dtype=tf.float32)

        feats, preds = self.scorecam_model(x, training=False)
        probs = self._ensure_probs(preds)

        if class_index is None:
            class_index = int(tf.argmax(probs[0]).numpy())

        confidence = float(probs[0, class_index].numpy())

        feats = tf.nn.relu(feats[0])  # (h,w,c)
        c = int(feats.shape[-1])

        ch_mean = tf.reduce_mean(feats, axis=(0, 1))  # (c,)
        top_idx = tf.argsort(ch_mean, direction="DESCENDING")[:min(max_channels, c)].numpy().tolist()

        cam = tf.zeros(self.img_size, dtype=tf.float32)

        for ci in top_idx:
            a = feats[:, :, ci]  # (h,w)

            mask = tf.image.resize(a[..., None], self.img_size)  # (224,224,1)
            mask = mask - tf.reduce_min(mask)
            mask = mask / (tf.reduce_max(mask) + self.eps)

            masked_x = x * mask[None, ...]

            masked_preds = self.loaded_model(masked_x, training=False)
            masked_preds = masked_preds[0] if isinstance(masked_preds, (list, tuple)) else masked_preds
            masked_probs = self._ensure_probs(masked_preds)

            score = masked_probs[0, class_index]
            cam += tf.squeeze(mask) * tf.nn.relu(score)

        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + self.eps)

        heatmap = cam.numpy().astype(np.float32)
        return orig_rgb, heatmap, class_index, confidence

    # ----------------------------
    # Full analysis used by GUI
    # ----------------------------
    def analyze(self, image_path: str, alpha: float = 0.45, max_channels: int = 32):
        pred_label, pred_conf, pred_index = self.predict(image_path)
        orig_rgb, heatmap, _, confidence = self.compute_scorecam(
            image_path,
            class_index=pred_index,
            max_channels=max_channels
        )

        overlay_rgb = self.overlay_heatmap_jet(orig_rgb, heatmap, alpha=alpha)

        # Return PIL images for GUI
        orig_pil = Image.fromarray(orig_rgb)
        overlay_pil = Image.fromarray(overlay_rgb)
        return pred_label, confidence, orig_pil, overlay_pil
