# 📋 IMPROVED PROJECT PLAN: Analisis Sentimen Reaksi Publik terhadap Penunjukan Liam Rosenior sebagai Pelatih Chelsea

**Status**: ✅ PROJECT STRUCTURE CREATED
**Tanggal**: 15 Januari 2026
**Anggota Kelompok**: [Nama Anda - Solo Project]

---

## 📁 Project Structure

```
analisis-sentiment-pelatih-baru-chelsea-liam-rosenior/
├── data/
│   ├── raw/
│   │   └── tweets.csv                    # Dataset asli (621 tweets)
│   ├── processed/
│   │   ├── tweets_cleaned.csv            # Setelah cleaning
│   │   └── tweets_labeled.csv            # Setelah VADER labeling
│   └── README.md                         # Dokumentasi data
│
├── notebooks/                            # 6 Jupyter Notebooks
│   ├── 0_setup_colab.ipynb               # Setup lingkungan Colab
│   ├── 1_data_preprocessing.ipynb        # Data cleaning & filtering
│   ├── 2_exploratory_analysis.ipynb      # EDA & visualisasi
│   ├── 3_sentiment_labeling.ipynb        # VADER sentiment analysis
│   ├── 4_ml_modeling.ipynb               # LR & NB model training
│   └── 5_results_visualization.ipynb     # Export hasil
│
├── src/                                  # Python utility modules
│   ├── __init__.py
│   ├── utils.py                          # Path handling, Colab support
│   ├── preprocessing.py                  # Data cleaning functions
│   ├── feature_engineering.py            # TF-IDF, feature extraction
│   └── models.py                         # ML training & evaluation
│
├── outputs/
│   ├── figures/                          # Visualisasi untuk laporan
│   ├── tables/                           # Tabel hasil analisis
│   ├── models/                           # Model terlatih (.pkl)
│   └── metrics/                          # Metrics JSON
│
├── reports/
│   ├── draft/
│   └── final/
│       ├── laporan_sentimen_analisis.pdf # Laporan akhir
│       ├── presentasi.pdf                # PPT presentasi
│       └── poster.pdf                    # Poster
│
├── requirements.txt                      # Dependencies Python
├── README.md                             # Dokumentasi proyek
├── .gitignore                            # Git ignore rules
└── PLAN.md                               # Rencana proyek ini
```

---

## 📓 Execution Order (Google Colab Compatible)

| Notebook | Nama | Tujuan | Output |
|----------|------|--------|--------|
| 0 | Setup Colab | Mount Drive, install packages | Environment ready |
| 1 | Data Preprocessing | Clean data, filter @grok | tweets_cleaned.csv |
| 2 | Exploratory Analysis | EDA, WordCloud, statistics | Visualizations |
| 3 | Sentiment Labeling | VADER analysis | tweets_labeled.csv |
| 4 | ML Modeling | Train LR + NB | Trained models + metrics |
| 5 | Results Export | Dashboard, tables | Report-ready outputs |

---

## 🎯 Langkah Eksekusi

### **PHASE 1: Setup (Hari 1)**
```
1. Upload folder proyek ke Google Drive
2. Buka 0_setup_colab.ipynb di Colab
3. Run semua cell untuk setup lingkungan
4. Verifikasi tweets.csv ter-load
```

### **PHASE 2: Preprocessing (Hari 1-2)**
```
1. Jalankan 1_data_preprocessing.ipynb
2. Hapus @grok queries, non-English, duplicates
3. Hasil: ~250-300 tweets bersih
4. Simpan ke data/processed/
```

### **PHASE 3: EDA & Labeling (Hari 2-3)**
```
1. Jalankan 2_exploratory_analysis.ipynb
2. Buat WordCloud, statistik engagement
3. Jalankan 3_sentiment_labeling.ipynb
4. Apply VADER, kategorisasi sentiment
```

### **PHASE 4: ML Modeling (Hari 3-5)**
```
1. Jalankan 4_ml_modeling.ipynb
2. Train Logistic Regression + Naive Bayes
3. Hyperparameter tuning dengan GridSearchCV
4. Bandingkan performa model
```

### **PHASE 5: Deliverables (Hari 5-12)**
```
1. Jalankan 5_results_visualization.ipynb
2. Export semua visualisasi
3. Buat laporan (PDF)
4. Buat PPT presentasi
5. Desain poster
6. Rekam video presentasi
```

---

## 📊 Deliverables

### Academic Reports
- [ ] Laporan (PDF) - 15-20 halaman
  - Bab I: Pendahuluan
  - Bab II: Tinjauan Pustaka
  - Bab III: Metodologi
  - Bab IV: Hasil dan Pembahasan
  - Bab V: Kesimpulan dan Saran

### Visual Materials
- [ ] Presentasi (PPT PDF) - 10-12 slides
- [ ] Poster (A3) - Layout 3 kolom
- [ ] Video (MP4) - 3-5 menit

### Analysis Outputs
- [x] 6 Jupyter Notebooks (Google Colab compatible)
- [x] 4 Python utility modules
- [x] Visualisasi (PNG high-resolution)
- [x] Tabel hasil (CSV)
- [x] Trained models (.pkl)

---

## 🔧 Technical Details

### Dataset
- **Sumber**: X (Twitter) - 6 Januari 2026
- **Original**: 621 tweets
- **After Cleaning**: ~250-300 tweets
- **Features**: Tweet Content, Views, Likes, Retweets, Replies, Timestamp

### Machine Learning
- **Models**: Logistic Regression, Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF (max_features=500, ngram_range=(1,2))
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Hyperparameter Tuning**: GridSearchCV (5-fold CV)

### Sentiment Analysis
- **Method**: VADER Sentiment Analysis
- **Thresholds**:
  - compound ≥ 0.05 → Positive
  - compound ≤ -0.05 → Negative
  - -0.05 < compound < 0.05 → Neutral

---

## 🚀 Cara Menggunakan di Google Colab

### Langkah 1: Upload ke Google Drive
1. Download folder proyek ini
2. Upload ke Google Drive di path:
   ```
   /content/drive/MyDrive/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior
   ```

### Langkah 2: Buka di Colab
1. Buka Google Colab (colab.research.google.com)
2. File → Open notebook
3. Pilih notebook dari Google Drive

### Langkah 3: Run Setup
1. Buka `notebooks/0_setup_colab.ipynb`
2. Run semua cell
3. Colab akan mount Drive dan install packages

### Langkah 4: Execute in Order
1. `0_setup_colab.ipynb` → Setup
2. `1_data_preprocessing.ipynb` → Clean data
3. `2_exploratory_analysis.ipynb` → EDA
4. `3_sentiment_labeling.ipynb` → VADER
5. `4_ml_modeling.ipynb` → ML models
6. `5_results_visualization.ipynb` → Export

---

## 📝 Catatan Penting

1. **Data Cleaning**: Filter @grok queries karena bukan opini genuine tentang topik
2. **Dataset Size**: 621 → ~250-300 tweets cukup untuk tugas akademis
3. **Google Colab**: RAM dan storage gratis, tapi notebook akan reset jika idle lama
4. **Backup**: Simpan output ke Google Drive agar tidak hilang

---

## ✅ Checklist Proyek

- [x] Create project folder structure
- [x] Create 6 Jupyter Notebooks
- [x] Create 4 Python utility modules
- [x] Create requirements.txt
- [x] Create .gitignore
- [x] Create README.md
- [x] Create data/raw/tweets.csv
- [ ] Run notebook 0 (setup)
- [ ] Run notebook 1 (preprocessing)
- [ ] Run notebook 2 (EDA)
- [ ] Run notebook 3 (sentiment labeling)
- [ ] Run notebook 4 (ML modeling)
- [ ] Run notebook 5 (results export)
- [ ] Write final report (PDF)
- [ ] Create presentation (PPT PDF)
- [ ] Design poster (PDF)
- [ ] Record video (MP4)
- [ ] Submit deliverables

---

**Created**: January 2026
**Status**: Ready for Execution 🎯
