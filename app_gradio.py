import os
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
import cv2

# =============================================================
# KONFIGURASI
# =============================================================

# Ganti sesuai model yang mau dipakai:
#   "efficientnetb0"  -> Best Model (recommended)
#   "mobilenetv2"     -> Runner-up
#   "cnn_baseline"    -> CNN dari nol
MODEL_NAME = "efficientnetb0"

MODEL_PATH = f"{MODEL_NAME}.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Gigi 3", "Gigi 4"]

# Info display untuk header
MODEL_INFO = {
    "efficientnetb0": {
        "display": "EfficientNetB0 (Transfer Learning)",
        "metrics": "Accuracy: 90% | F1-Score: 0.8974 | BEST MODEL",
    },
    "mobilenetv2": {
        "display": "MobileNetV2 (Transfer Learning)",
        "metrics": "Accuracy: 81% | F1-Score: 0.81",
    },
    "cnn_baseline": {
        "display": "CNN Baseline (from scratch)",
        "metrics": "Accuracy: 46% | F1-Score: 0.3443",
    },
}

# =============================================================
# LOAD MODEL
# =============================================================

print("=" * 60)
print(f"MEMUAT MODEL: {MODEL_NAME.upper()}...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}\n"
        f"Pastikan file {MODEL_PATH} ada di folder yang sama "
        f"dengan app_gradio.py. Atau ganti variabel MODEL_NAME "
        f"di bagian KONFIGURASI."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model berhasil dimuat dari: {MODEL_PATH}")
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")


# =============================================================
# PREPROCESSING - PER MODEL (INI YANG SEBELUMNYA BUG!)
# =============================================================

def preprocess_image(img_pil):
    """
    Preprocessing yang KONSISTEN dengan training.

    PENTING:
    - EfficientNet pakai preprocessing built-in, butuh input [0, 255]
    - MobileNetV2 & CNN training pakai rescale 1/255, butuh [0, 1]

    Kalau salah preprocessing -> prediksi ngawur meskipun model bagus.
    """
    img_resized = img_pil.convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img_resized, dtype=np.float32)

    if MODEL_NAME == "efficientnetb0":
        # JANGAN dibagi 255. EfficientNet handle internal.
        img_tensor = np.expand_dims(img_arr, 0)
    else:
        # MobileNetV2 & CNN-Baseline
        img_arr = img_arr / 255.0
        img_tensor = np.expand_dims(img_arr, 0)

    # Untuk display dan Grad-CAM butuh versi [0, 255] uint8
    img_uint8 = np.array(img_resized, dtype=np.uint8)
    return img_tensor, img_uint8


# =============================================================
# FUNGSI GRAD-CAM
# =============================================================

def get_gradcam_heatmap(image_tensor):
    """
    Generate Grad-CAM heatmap untuk visualisasi area fokus model.
    Versi robust yang cari Conv2D layer terakhir di seluruh model.
    """
    try:
        # Cari Conv2D layer terakhir (cek di level model dan sub-model)
        last_conv_layer = None
        last_conv_name = None

        def find_last_conv(layers):
            for layer in reversed(layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer
                # Cek kalau ini sub-model (transfer learning)
                if hasattr(layer, "layers"):
                    inner = find_last_conv(layer.layers)
                    if inner is not None:
                        return inner
            return None

        last_conv_layer = find_last_conv(model.layers)
        if last_conv_layer is None:
            return None

        last_conv_name = last_conv_layer.name

        # Build grad model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output if last_conv_name in [l.name for l in model.layers]
                     else last_conv_layer.output,
                     model.output]
        )

        # Karena layer mungkin di dalam sub-model, kita pakai trick:
        # rebuild dengan functional API
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output]
            )
        except Exception:
            return None

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(image_tensor)
            pred = preds[:, 0]

        grads = tape.gradient(pred, conv_out)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def apply_gradcam_overlay(image_uint8, heatmap, alpha=0.5):
    """Overlay heatmap ke gambar original."""
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (image_uint8 * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlay


# =============================================================
# FUNGSI PREDIKSI UTAMA
# =============================================================

def predict_image(image):
    """
    Fungsi prediksi utama yang dipanggil oleh Gradio.

    Input : image (numpy array dari Gradio)
    Output:
        - dict confidence per kelas (label)
        - text hasil prediksi (markdown)
        - gambar dengan Grad-CAM overlay
    """
    if image is None:
        return ({"Gigi 3": 0.0, "Gigi 4": 0.0},
                "⚠️ Silakan upload gambar terlebih dahulu",
                None)

    try:
        # === Preprocessing yang BENAR per model ===
        img_pil = Image.fromarray(image)
        img_tensor, img_uint8 = preprocess_image(img_pil)

        # === Prediksi ===
        prob = float(model.predict(img_tensor, verbose=0)[0, 0])

        # === Format hasil ===
        confidence_dict = {
            "Gigi 3": float(1 - prob),
            "Gigi 4": float(prob),
        }

        pred_class = CLASS_NAMES[int(prob > 0.5)]
        confidence_value = prob if prob > 0.5 else 1 - prob

        result_text = (
            f"🎯 **Prediksi: Garpu Bergigi {pred_class.split()[-1]}**\n\n"
            f"📊 **Confidence: {confidence_value * 100:.2f}%**\n\n"
            f"💡 Detail probabilitas:\n"
            f"   • Gigi 3: {(1 - prob) * 100:.2f}%\n"
            f"   • Gigi 4: {prob * 100:.2f}%\n\n"
            f"🔧 Model: `{MODEL_NAME}`"
        )

        if confidence_value > 0.9:
            result_text += "\n\n✅ Model sangat yakin dengan prediksi ini."
        elif confidence_value > 0.7:
            result_text += "\n\n👍 Model cukup yakin dengan prediksi ini."
        else:
            result_text += "\n\n⚠️ Model kurang yakin - gambar mungkin ambigu."

        # === Grad-CAM ===
        gradcam_img = None
        try:
            heatmap = get_gradcam_heatmap(img_tensor)
            if heatmap is not None:
                gradcam_img = apply_gradcam_overlay(img_uint8, heatmap, alpha=0.45)
        except Exception as e:
            print(f"Grad-CAM gagal: {e}")

        return confidence_dict, result_text, gradcam_img

    except Exception as e:
        error_text = f"❌ Error saat prediksi: {str(e)}"
        return {"Gigi 3": 0.0, "Gigi 4": 0.0}, error_text, None


# =============================================================
# BUILD INTERFACE GRADIO
# =============================================================

custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, sans-serif !important;
}
"""

info = MODEL_INFO.get(MODEL_NAME, {"display": MODEL_NAME, "metrics": ""})

with gr.Blocks(
    title="Klasifikasi Garpu Gigi 3 vs Gigi 4",
    theme=gr.themes.Soft(primary_hue="teal"),
    css=custom_css,
) as demo:

    gr.Markdown(
        f"""
        # 🍴 Klasifikasi Citra Garpu Bergigi 3 vs Gigi 4
        ### UAS Machine Learning - Muhammad Rizki Saputra (2310817310014)
        **Model: {info['display']}**
        **{info['metrics']}**

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Gambar Garpu")
            image_input = gr.Image(
                label="Upload gambar garpu (JPG/PNG)",
                type="numpy",
                height=350,
            )

            predict_btn = gr.Button(
                "🔍 Prediksi Sekarang",
                variant="primary",
                size="lg",
            )

            clear_btn = gr.Button("🗑️ Reset", variant="secondary")

            gr.Markdown(
                """
                ### 📝 Petunjuk:
                1. Upload gambar garpu (JPG/PNG)
                2. Pastikan garpu terlihat jelas di gambar
                3. Klik tombol **Prediksi Sekarang**
                4. Lihat hasil prediksi & visualisasi
                """
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Hasil Prediksi")

            label_output = gr.Label(
                label="Probabilitas Kelas",
                num_top_classes=2,
            )

            result_text = gr.Markdown(
                value="*Upload gambar dan klik tombol Prediksi untuk melihat hasil.*"
            )

            gr.Markdown("### 🔥 Grad-CAM (Area Fokus Model)")
            gradcam_output = gr.Image(
                label="Heatmap menunjukkan area yang menjadi fokus model",
                height=300,
            )

    gr.Markdown(
        """
        ---
        ### 📚 Informasi Model
        - **Arsitektur**: EfficientNetB0 dengan Transfer Learning dari ImageNet
        - **Training**: Dataset 614 gambar (Roboflow + Pinterest manual)
        - **Augmentation**: 8 teknik (rotate, flip, brightness, contrast, dll)
        - **Methodology**: Group-aware splitting untuk evaluasi yang reliable
        - **Best Result**: 90% Accuracy, 0.8974 F1-Score

        ### ⚠️ Catatan Preprocessing
        EfficientNet butuh input piksel raw `[0, 255]` (TIDAK dibagi 255),
        karena layer preprocessing sudah built-in di arsitekturnya. Kalau
        dibagi 255 sebelum masuk model, akurasinya akan jatuh drastis ke
        sekitar 50% (random guessing).

        ### 🎨 Penjelasan Visualisasi Grad-CAM
        Warna merah/kuning pada heatmap menunjukkan area yang paling
        diperhatikan model saat menentukan jumlah gigi garpu. Idealnya,
        area ujung garpu (tempat gigi) harus menjadi fokus utama.

        ---
        *Universitas Lambung Mangkurat - Program Studi Teknologi Informasi - 2026*
        """
    )

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[label_output, result_text, gradcam_output],
    )

    clear_btn.click(
        fn=lambda: (None, {"Gigi 3": 0.0, "Gigi 4": 0.0},
                    "*Upload gambar dan klik tombol Prediksi untuk melihat hasil.*", None),
        inputs=None,
        outputs=[image_input, label_output, result_text, gradcam_output],
    )


# =============================================================
# LAUNCH APP
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MENJALANKAN APLIKASI DEMO...")
    print("=" * 60)
    print(f"Model aktif: {MODEL_NAME}")
    print("Buka URL yang muncul di browser kamu.")
    print("Untuk stop: Tekan Ctrl+C di terminal\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )