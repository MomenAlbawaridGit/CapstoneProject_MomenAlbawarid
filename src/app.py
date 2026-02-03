import os
import threading
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from xai import ScoreCAMBrainTumorXAI


def pil_fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img = image.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def pretty_label(raw: str) -> str:
    # meningioma_tumor -> Meningioma Tumor
    return raw.replace("_", " ").strip().title()


class LoadingModal(ctk.CTkToplevel):
    def __init__(self, parent, text="Analyzing..."):
        super().__init__(parent)
        self.title("Please wait")
        self.geometry("360x140")
        self.resizable(False, False)
        self.configure(fg_color="#0B1220")

        self.transient(parent)
        self.grab_set()

        self.label = ctk.CTkLabel(
            self, text=text,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E6F1FF"
        )
        self.label.pack(pady=(22, 10))

        self.pb = ctk.CTkProgressBar(self, mode="indeterminate", width=280)
        self.pb.pack(pady=(6, 0))
        self.pb.start()

        self.hint = ctk.CTkLabel(
            self, text="Running Score-CAM on CPU…",
            font=ctk.CTkFont(size=13),
            text_color="#9FB3C8"
        )
        self.hint.pack(pady=(10, 0))

        # Center relative to parent
        self.update_idletasks()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        w = 360
        h = 140
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def close(self):
        try:
            self.pb.stop()
        except Exception:
            pass
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BrainTumorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Theme: Dark “radiology”
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Brain Tumor Classifier • MRI + Score-CAM")
        self.geometry("1100x680")
        self.minsize(980, 620)
        self.configure(fg_color="#070D19")

        # XAI engine (keep logic unchanged)
        self.engine = ScoreCAMBrainTumorXAI(
            model_path=os.path.join("models", "mobilenet_nd5_final_finetuned_cw_v2.h5"),
            verbose=False
        )

        # State
        self.selected_path = None
        self.loading_modal = None
        self.is_analyzing = False

        # UI
        self._build_header()
        self._build_body()
        self._build_footer()
        self.reset_ui(full=True)

    # ---------- UI Sections ----------
    def _build_header(self):
        self.header = ctk.CTkFrame(self, fg_color="#0B1220", corner_radius=18)
        self.header.pack(fill="x", padx=18, pady=(18, 10))

        self.title_label = ctk.CTkLabel(
            self.header,
            text="Brain Tumor Classification",  # ✅ removed (MobileNetV2)
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#E6F1FF"
        )
        self.title_label.pack(anchor="w", padx=18, pady=(14, 4))

        self.subtitle_label = ctk.CTkLabel(
            self.header,
            text="Select an MRI image, analyze, then review Score-CAM localization.",
            font=ctk.CTkFont(size=14),
            text_color="#9FB3C8"
        )
        self.subtitle_label.pack(anchor="w", padx=18, pady=(0, 12))

    def _build_body(self):
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=18, pady=10)

        # Left column: input panel
        self.left = ctk.CTkFrame(self.body, fg_color="#0B1220", corner_radius=18)
        self.left.pack(side="left", fill="y", padx=(0, 10))

        self.drop_title = ctk.CTkLabel(
            self.left,
            text="Input MRI",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E6F1FF"
        )
        self.drop_title.pack(anchor="w", padx=16, pady=(16, 6))

        # ✅ No drag-drop zone. Just a browse hint card.
        self.browse_card = ctk.CTkFrame(self.left, fg_color="#0E1A2E", corner_radius=16, width=360, height=140)
        self.browse_card.pack(padx=16, pady=(8, 10))
        self.browse_card.pack_propagate(False)

        self.browse_label = ctk.CTkLabel(
            self.browse_card,
            text="Click 'Browse MRI' to select an image",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#CFE8FF"
        )
        self.browse_label.pack(expand=True)

        self.path_label = ctk.CTkLabel(
            self.left,
            text="No file selected",
            font=ctk.CTkFont(size=12),
            text_color="#9FB3C8",
            wraplength=340,
            justify="left"
        )
        self.path_label.pack(anchor="w", padx=16, pady=(0, 14))

        # Results summary
        self.result_card = ctk.CTkFrame(self.left, fg_color="#0E1A2E", corner_radius=16)
        self.result_card.pack(fill="x", padx=16, pady=(0, 16))

        self.pred_label = ctk.CTkLabel(
            self.result_card,
            text="Prediction: —",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#49D3C7"
        )
        self.pred_label.pack(anchor="w", padx=14, pady=(12, 2))

        self.conf_label = ctk.CTkLabel(
            self.result_card,
            text="Confidence: —",
            font=ctk.CTkFont(size=14),
            text_color="#CFE8FF"
        )
        self.conf_label.pack(anchor="w", padx=14, pady=(0, 12))

        # Right column: images
        self.right = ctk.CTkFrame(self.body, fg_color="#0B1220", corner_radius=18)
        self.right.pack(side="left", fill="both", expand=True)

        self.images_title = ctk.CTkLabel(
            self.right,
            text="Visualization",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E6F1FF"
        )
        self.images_title.pack(anchor="w", padx=16, pady=(16, 10))

        self.images_row = ctk.CTkFrame(self.right, fg_color="transparent")
        self.images_row.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.orig_panel = ctk.CTkFrame(self.images_row, fg_color="#0E1A2E", corner_radius=16)
        self.orig_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.cam_panel = ctk.CTkFrame(self.images_row, fg_color="#0E1A2E", corner_radius=16)
        self.cam_panel.pack(side="left", fill="both", expand=True)

        self.orig_title = ctk.CTkLabel(
            self.orig_panel, text="Original",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#CFE8FF"
        )
        self.orig_title.pack(anchor="w", padx=14, pady=(12, 6))

        self.cam_title = ctk.CTkLabel(
            self.cam_panel, text="Score-CAM Overlay",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#CFE8FF"
        )
        self.cam_title.pack(anchor="w", padx=14, pady=(12, 6))

        self.orig_image_label = ctk.CTkLabel(self.orig_panel, text="—", text_color="#9FB3C8")
        self.orig_image_label.pack(expand=True, pady=(0, 12))

        self.cam_image_label = ctk.CTkLabel(self.cam_panel, text="—", text_color="#9FB3C8")
        self.cam_image_label.pack(expand=True, pady=(0, 12))

        # Hold CTkImage references
        self._ctk_orig_img = None
        self._ctk_cam_img = None

    def _build_footer(self):
        self.footer = ctk.CTkFrame(self, fg_color="transparent")
        self.footer.pack(fill="x", padx=18, pady=(0, 18))

        self.btn_browse = ctk.CTkButton(
            self.footer,
            text="Browse MRI",
            command=self.on_browse,
            fg_color="#123A5A",
            hover_color="#16507A",
            corner_radius=14,
            height=44
        )
        self.btn_browse.pack(side="left", padx=(0, 10))

        self.btn_analyze = ctk.CTkButton(
            self.footer,
            text="Analyze",
            command=self.on_analyze,
            fg_color="#1A6A66",
            hover_color="#218A84",
            corner_radius=14,
            height=44
        )
        self.btn_analyze.pack(side="left", padx=(0, 10))

        self.btn_new = ctk.CTkButton(
            self.footer,
            text="New Image",
            command=self.on_new_image,
            fg_color="#2A3347",
            hover_color="#36415A",
            corner_radius=14,
            height=44
        )
        self.btn_new.pack(side="left")

        self.status = ctk.CTkLabel(
            self.footer,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="#9FB3C8"
        )
        self.status.pack(side="right", padx=(10, 0))

    # ---------- Safe image clearing (prevents pyimage errors) ----------
    def _force_clear_tk_image(self, label: ctk.CTkLabel):
        """
        CustomTkinter wraps an internal tkinter.Label. Sometimes it keeps stale image refs.
        Force-clear the internal tk label image to avoid 'pyimageX doesn't exist'.
        """
        try:
            label._label.configure(image="")
        except Exception:
            pass

    def _clear_label_image(self, label: ctk.CTkLabel, text="—"):
        self._force_clear_tk_image(label)
        label.configure(image=None, text=text)

    def _set_label_image(self, label: ctk.CTkLabel, ctk_img):
        self._force_clear_tk_image(label)
        label.configure(image=ctk_img, text="")

    # ---------- Reset UI ----------
    def reset_ui(self, full: bool = False):
        """
        Reset everything so user can browse + analyze again.
        full=True is for startup, full=False used for New Image.
        """
        self.selected_path = None
        self.is_analyzing = False

        self.path_label.configure(text="No file selected")
        self.status.configure(text="Ready")

        self.pred_label.configure(text="Prediction: —")
        self.conf_label.configure(text="Confidence: —")

        # Drop image references
        self._ctk_orig_img = None
        self._ctk_cam_img = None

        # Clear image labels safely
        self._clear_label_image(self.orig_image_label, text="—")
        self._clear_label_image(self.cam_image_label, text="—")

        # Buttons
        self.btn_browse.configure(state="normal")
        self.btn_analyze.configure(state="disabled")
        self.btn_new.configure(state="disabled")

        if full:
            # no extra work needed
            pass

    # ---------- Actions ----------
    def on_browse(self):
        if self.is_analyzing:
            return

        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.set_selected_image(path)

    def on_new_image(self):
        # ✅ start over cleanly
        if self.is_analyzing:
            return
        self.reset_ui(full=False)

    def set_selected_image(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            messagebox.showerror("Unsupported file", "Please choose a JPG or PNG image.")
            return

        self.selected_path = path
        self.path_label.configure(text=path)
        self.status.configure(text="Image loaded. Press Analyze.")

        # Preview original
        try:
            pil = Image.open(path).convert("RGB")
            preview = pil_fit(pil, 420, 420)
            self._ctk_orig_img = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
            self._set_label_image(self.orig_image_label, self._ctk_orig_img)
        except Exception:
            self._clear_label_image(self.orig_image_label, text="Preview failed")

        # Clear previous CAM & text
        self._ctk_cam_img = None
        self._clear_label_image(self.cam_image_label, text="—")
        self.pred_label.configure(text="Prediction: —")
        self.conf_label.configure(text="Confidence: —")

        self.btn_analyze.configure(state="normal")
        self.btn_new.configure(state="normal")

    def on_analyze(self):
        if not self.selected_path or self.is_analyzing:
            return

        self.is_analyzing = True
        self.status.configure(text="Analyzing…")
        self.btn_analyze.configure(state="disabled")
        self.btn_browse.configure(state="disabled")
        self.btn_new.configure(state="disabled")

        self.loading_modal = LoadingModal(self, text="Analyzing MRI…")

        t = threading.Thread(target=self._analyze_worker, daemon=True)
        t.start()

    def _analyze_worker(self):
        try:
            pred_label_raw, confidence, orig_pil, cam_pil = self.engine.analyze(
                self.selected_path,
                alpha=0.45,
                max_channels=32
            )

            # Pretty class name for UI
            pred_label = pretty_label(pred_label_raw)

            orig_preview = pil_fit(orig_pil, 420, 420)
            cam_preview = pil_fit(cam_pil, 420, 420)

            def on_done():
                self.is_analyzing = False

                if self.loading_modal:
                    self.loading_modal.close()
                    self.loading_modal = None

                self.pred_label.configure(text=f"Prediction: {pred_label}")
                self.conf_label.configure(text=f"Confidence: {confidence:.3f}")

                self._ctk_orig_img = ctk.CTkImage(light_image=orig_preview, dark_image=orig_preview, size=orig_preview.size)
                self._ctk_cam_img = ctk.CTkImage(light_image=cam_preview, dark_image=cam_preview, size=cam_preview.size)

                self._set_label_image(self.orig_image_label, self._ctk_orig_img)
                self._set_label_image(self.cam_image_label, self._ctk_cam_img)

                self.status.configure(text="Done. Click New Image to analyze another.")
                self.btn_browse.configure(state="normal")
                self.btn_new.configure(state="normal")
                self.btn_analyze.configure(state="normal")

            self.after(0, on_done)

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))

            def on_fail():
                self.is_analyzing = False

                if self.loading_modal:
                    self.loading_modal.close()
                    self.loading_modal = None

                self.status.configure(text="Error")
                self.btn_browse.configure(state="normal")
                self.btn_new.configure(state="normal")
                self.btn_analyze.configure(state="normal")

                messagebox.showerror("Analysis failed", f"{e}\n\nDetails:\n{err}")

            self.after(0, on_fail)


if __name__ == "__main__":
    app = BrainTumorApp()
    app.mainloop()

