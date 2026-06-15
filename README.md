# 🍴 Klasifikasi Citra Garpu Bergigi 3 dan Gigi 4

> **Ujian Akhir Semester — Machine Learning**
> Universitas Lambung Mangkurat · Program Studi Teknologi Informasi

---

## 📌 Deskripsi

Proyek ini membandingkan **7 skenario model** untuk klasifikasi citra garpu bergigi 3 (*three-pronged*) vs bergigi 4 (*four-pronged*) menggunakan dua pendekatan utama:

- **Machine Learning Klasik** — ekstraksi fitur HOG + Support Vector Machine (4 variasi)
- **Deep Learning + Transfer Learning** — CNN from scratch, MobileNetV2, dan EfficientNetB0

Model terbaik adalah **EfficientNetB0** dengan akurasi **90%** dan F1-Score **0.8974** pada test set.

---

## 📂 Struktur Direktori

```
garpu-uas-dataset/
├── dataset/
│   ├── train/
│   │   ├── 3_prong/        # ±700 gambar (setelah augmentasi)
│   │   └── 4_prong/        # ±700 gambar (setelah augmentasi)
│   ├── val/
│   │   ├── 3_prong/
│   │   └── 4_prong/
│   └── test/
│       ├── 3_prong/        # 50 gambar asli
│       └── 4_prong/        # 50 gambar asli
│
├── models/
│   ├── efficientnetb0.keras    # ✅ Model terbaik
│   ├── mobilenetv2.keras
│   └── svm_best_grid.pkl
│
├── notebooks/
│   └── UAS_ML_GarpuClassification.ipynb
│
├── app_gradio.py               # Demo inferensi interaktif
├── comparison_all_models.csv   # Rekap hasil semua model
└── README.md
```

---

## 📊 Dataset

| Properti | Detail |
|---|---|
| Sumber | [Kaggle — Garpu UAS Dataset](https://www.kaggle.com/datasets/mhmdrizkisaputra/garpu-uas-dataset) |
| Total gambar | 614 (307 per kelas) |
| Kelas | `3_prong` · `4_prong` |
| Split | 70% Train / 15% Val / 15% Test |
| Strategi split | **Group-Aware Splitting** (tanpa kebocoran data antar grup) |
| Augmentasi | Flip, Rotate, Zoom, Brightness → ~1000 gambar/kelas di train |

---

## 🤖 Model & Hasil

### Machine Learning (HOG + SVM)

| # | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 1 | SVM-Linear | 0.63 | 0.6349 | 0.6354 | 0.63 |
| 2 | SVM-RBF | 0.58 | 0.5790 | 0.5798 | 0.5785 |
| 3 | SVM-Best-Grid | 0.59 | 0.5881 | 0.5889 | 0.5880 |
| 4 | SVM-PCA | 0.53 | 0.5321 | 0.5323 | 0.5296 |

### Deep Learning

| # | Model | Accuracy | Precision | Recall | F1-Score | Keterangan |
|---|---|---|---|---|---|---|
| 1 | **EfficientNetB0** | **0.90** | **0.9088** | **0.8929** | **0.8974** | ⭐ Best |
| 2 | MobileNetV2 | 0.81 | 0.8206 | 0.8192 | 0.8100 | Runner-up |
| 3 | CNN-Baseline | 0.46 | 0.5601 | 0.5071 | 0.3443 | Worst |

> **Kesimpulan:** Transfer Learning (EfficientNetB0) unggul +27 poin akurasi dibanding SVM terbaik. CNN from scratch hanya 46% — membuktikan pentingnya pretrained features pada dataset terbatas.

---

## ⚙️ Instalasi

```bash
# Clone repository
git clone https://github.com/Yoruuu00/garpu-uas-ml.git
cd garpu-uas-ml

# Install dependencies
pip install tensorflow scikit-learn opencv-python gradio numpy matplotlib seaborn
```

**Versi yang digunakan:**
```
tensorflow >= 2.12
scikit-learn >= 1.2
opencv-python >= 4.7
gradio >= 3.40
numpy >= 1.23
```

---

## 🚀 Cara Penggunaan

### 1. Training ulang (opsional)
Buka dan jalankan notebook secara berurutan:
```
notebooks/UAS_ML_GarpuClassification.ipynb
```

### 2. Inferensi via script
```python
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('models/efficientnetb0.keras')

img = Image.open('path/to/gambar.jpg').resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

pred = model.predict(img_array)
label = "4 Gigi" if pred[0][0] > 0.5 else "3 Gigi"
print(f"Prediksi: {label} (confidence: {pred[0][0]:.2f})")
```

### 3. Demo Gradio (interaktif)
```bash
python app_gradio.py
```
Akses di browser: `http://localhost:7860`

---

## 🔍 Interpretasi Model (Grad-CAM)

Visualisasi Grad-CAM menunjukkan model EfficientNetB0 **fokus pada area ujung gigi garpu** — bukan gagang atau latar belakang — membuktikan model belajar fitur yang relevan dan tidak terjebak spurious correlation.

---

## 📈 Analisis Overfitting

| Model | Train Acc | Val Acc | Gap | Status |
|---|---|---|---|---|
| EfficientNetB0 | ~0.91 | ~0.92 | −0.007 | ✅ Sehat |
| MobileNetV2 | ~0.82 | ~0.83 | −0.005 | ✅ Sehat |
| CNN-Baseline | ~0.55 | ~0.48 | +0.070 | ⚠️ Slight Overfit |

Gap negatif pada model Transfer Learning membuktikan dropout + augmentasi + pretrained weights bekerja sebagai regularisasi yang efektif.

---

## 👤 Author

**Muhammad Rizki Saputra**
NIM: 2310817310014
Program Studi Teknologi Informasi — Fakultas Teknik
Universitas Lambung Mangkurat

Dosen Pengampu: **Erika Maulidiya, S.Kom., M.Kom**

---

## 📚 Referensi Utama

- Sandler et al. (2018) — MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
- Tan & Le (2019) — EfficientNet: Rethinking Model Scaling for CNNs. ICML.
- Tan & Le (2021) — EfficientNetV2: Smaller Models and Faster Training. ICML.
- Dalal & Triggs (2005) — HOG for Human Detection. CVPR.
- Selvaraju et al. (2017) — Grad-CAM. ICCV.
- Amraee et al. (2022) — HOG vs LBP for Small Metal Object Classification. *Visual Computing for Industry, Biomedicine, and Art.*

(jurnal diatas adalah jurnal pelapor modelnya, sedangkan untuk penelitian terbaru dijadikan sebagai pendamping)
---

<div align="center">
  <sub>UAS Machine Learning · 2025/2026 · Universitas Lambung Mangkurat</sub>
</div>
