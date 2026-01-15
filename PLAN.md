Berdasarkan analisis dataset yang kamu miliki, berikut **improvement plan** yang lebih terstruktur dan actionable: [tweets](/home/emmanuelabayor/projects/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior/tweets.csv)

## **IMPROVED PROJECT PLAN: Analisis Sentimen Reaksi Publik terhadap Penunjukan Liam Rosenior sebagai Pelatih Chelsea**

### **BAGIAN 1: Definisi Studi Kasus (Lebih Fokus & Spesifik)**

**Judul Studi Kasus (Revised):**  
*"Analisis Sentimen Media Sosial terhadap Penunjukan Liam Rosenior sebagai Pelatih Chelsea FC: Studi Reaksi Publik di Platform X (Januari 2026)"*

**Konteks Kasus:**
- Dataset mencakup 408 tweets tentang pengumuman kontrak 6,5 tahun Liam Rosenior di Chelsea [tweets](/home/emmanuelabayor/projects/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior/tweets.csv)
- Banyak tweet menunjukkan skeptisisme ("6 years contract like they won't sack him before 2027") dan pertanyaan (@grok mentions) [tweets](/home/emmanuelabayor/projects/analisis-sentiment-pelatih-baru-chelsea-liam-rosenior/tweets.csv)
- Relevansi: Analisis opini publik terhadap keputusan manajemen klub sepak bola

**Rumusan Masalah (Lebih Spesifik):**
1. Bagaimana distribusi sentimen publik terhadap penunjukan Liam Rosenior?
2. Topik atau aspek apa yang paling sering dibahas dalam diskusi publik?
3. Model machine learning mana yang paling akurat untuk memprediksi sentimen pada dataset ini?

***

### **BAGIAN 2: Data Preparation & Preprocessing (Ditambahkan Detail Teknis)**

**2.1 Data Cleaning:**
```python
# Tambahan langkah cleaning yang perlu dilakukan:
- Hapus baris dengan Tweet Content = NaN (15 baris dari 408)
- Hapus duplicate tweets (jika ada)
- Filter tweets berbahasa Inggris saja (gunakan langdetect)
- filter grok mentions untuk analisis terpisah
- Handle emoji (convert ke text atau hapus)
```

**2.2 Feature Engineering (BARU - Penting!):**
```python
# Tambahkan fitur tambahan untuk analisis lebih kaya:
- tweet_length: Panjang karakter tweet
- word_count: Jumlah kata
- hashtag_count: Jumlah hashtag
- mention_count: Jumlah mention
- has_grok_mention: Boolean (apakah ada @grok)
- engagement_score: (Likes + Retweets*2 + Replies*3)
- hour_posted: Jam posting untuk time series
```

**2.3 Text Preprocessing Pipeline:**
```python
1. Lowercase conversion
2. Remove URLs (regex: https?://\S+)
3. Remove mentions (@username) - KECUALI untuk analisis mention
4. Remove special characters & punctuation
5. Remove extra whitespaces
6. Tokenization
7. Remove stopwords (NLTK English stopwords)
8. Lemmatization (bukan stemming, lebih akurat)
```

***

### **BAGIAN 3: Exploratory Data Analysis (Lebih Komprehensif)**

**3.1 Analisis Deskriptif:**
- Statistik engagement (Views, Likes, Retweets, Replies)
- Top 10 most engaged tweets
- User activity distribution (siapa yang paling aktif?)

**3.2 Analisis Temporal:**
- Time series: Jumlah tweets per jam (kapan puncak diskusi?)
- Engagement trend over time

**3.3 Analisis Teks:**
- **WordCloud:** Kata paling sering muncul
- **Bigram & Trigram Analysis:** Frasa umum ("6 years contract", "will be sacked")
- **Top Mentions:** @grok, @ChelseaFC, dll
- **Top Hashtags:** (jika ada)

**3.4 Sentiment Pre-labeling:**
```python
# Gunakan VADER (lebih baik untuk social media)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Threshold scoring:
- compound_score >= 0.05 → Positive
- -0.05 < compound_score < 0.05 → Neutral
- compound_score <= -0.05 → Negative
```

***

### **BAGIAN 4: Machine Learning Modeling (Ditingkatkan)**

**4.1 Data Splitting:**
- Train: 80% (314 tweets)
- Test: 20% (79 tweets)
- Stratified split (pastikan proporsi sentiment sama)

**4.2 Feature Extraction:**
```python
# Gunakan dua metode untuk perbandingan:
1. TF-IDF Vectorizer (max_features=500, ngram_range=(1,2))
2. Count Vectorizer (untuk Naive Bayes)
```

**4.3 Model yang Digunakan (Expanded):**

| Model | Alasan Pemilihan |
|-------|------------------|
| **Logistic Regression** (Wajib) | Baseline model, mudah interpretasi, cepat |
| **Multinomial Naive Bayes** (Wajib) | Terbukti baik untuk text classification |

**4.4 Hyperparameter Tuning:**
```python
# Gunakan GridSearchCV untuk setiap model
# Contoh untuk Logistic Regression:
param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 200],
    'solver': ['liblinear', 'saga']
}
```

**4.5 Evaluasi Model (Lebih Detail):**
```python
Metrik yang digunakan:
1. Accuracy Score
2. Precision, Recall, F1-Score (per class)
3. Confusion Matrix (visualisasi)
4. Classification Report
5. ROC-AUC Score (jika applicable untuk multi-class)
```

***

### **BAGIAN 5: Struktur Laporan (Revised - Lebih Akademis)**

#### **I. PENDAHULUAN**
- **Latar Belakang:**  
  - Pentingnya analisis opini publik dalam era media sosial
  - Kasus Chelsea FC dan siklus pergantian pelatih
  - Signifikansi kontrak 6,5 tahun Rosenior
  
- **Rumusan Masalah:** (3 poin di atas)
  
- **Tujuan Penelitian:**
  1. Mengidentifikasi distribusi sentimen publik
  2. Mengekstrak topik utama diskusi
  3. Membandingkan performa model ML untuk sentiment analysis

- **Manfaat Penelitian:**
  - Bagi klub: Memahami persepsi publik
  - Bagi akademis: Studi kasus NLP pada domain sports

#### **II. TINJAUAN PUSTAKA (BARU - Penting untuk laporan akademis)**
- Definisi Sentiment Analysis
- Metode feature extraction (TF-IDF)
- Algoritma ML yang umum digunakan
- Penelitian terdahulu tentang sentiment analysis di sports domain

#### **III. METODOLOGI**
**3.1 Sumber Data:**
- Platform: X (Twitter)
- Periode: 6 Januari 2026
- Jumlah: 408 tweets → 393 valid (setelah cleaning)
- Keyword: "Liam Rosenior", "Chelsea", "@grok"

**3.2 Alat dan Teknologi:**
- Python 3.x
- Libraries: pandas, numpy, sklearn, nltk, vaderSentiment, matplotlib, seaborn, wordcloud

**3.3 Tahapan Analisis:**
```
Flowchart:
Data Collection → Data Cleaning → EDA → 
Feature Engineering → Sentiment Labeling → 
Train-Test Split → Model Training → 
Evaluation → Interpretation
```

#### **IV. HASIL DAN PEMBAHASAN**
**4.1 Hasil EDA:**
- Tabel statistik deskriptif
- Grafik engagement distribution
- WordCloud & Top Terms
- Time series chart

**4.2 Hasil Sentiment Labeling:**
- Pie chart distribusi sentiment (Positif/Netral/Negatif)
- Contoh tweets per kategori sentiment

**4.3 Hasil Model ML:**
- Tabel perbandingan accuracy 4 model
- Confusion matrix terbaik model
- Classification report detail
- Analisis error (contoh misclassification)

**4.4 Pembahasan:**
- Interpretasi kecenderungan sentiment (kemungkinan mayoritas Neutral/Negatif karena skeptisisme)
- Faktor yang mempengaruhi (kontrak panjang, rekam jejak pelatih)
- Limitation: Dataset 1 hari saja, bias platform

#### **V. KESIMPULAN DAN SARAN**
**Kesimpulan:**
- Sentiment dominan: [sesuai hasil]
- Model terbaik: [nama model + accuracy]
- Insight: Publik skeptis terhadap kontrak panjang

**Saran:**
- Untuk penelitian lanjutan: Tambah data beberapa minggu ke depan
- Gunakan deep learning (LSTM, BERT) untuk akurasi lebih tinggi
- Analisis aspek-based sentiment (sentiment terhadap: kontrak, track record, dll)

***

### **BAGIAN 6: Deliverables (Lebih Terstruktur)**

**6.1 Laporan (PDF) - 15-20 halaman:**
- Cover, Abstrak, Daftar Isi
- Bab I-V (sesuai struktur di atas)
- Daftar Pustaka (minimal 5 referensi)
- Lampiran: Script Python, Sample data

**6.2 Presentasi (PPT → PDF) - 10-12 slides:**
```
Slide 1: Cover (Judul + Tim)
Slide 2: Background & Problem Statement
Slide 3: Dataset Overview (stats + sample tweets)
Slide 4: Methodology Flowchart
Slide 5: EDA - WordCloud & Top Terms
Slide 6: EDA - Engagement Analysis
Slide 7: Sentiment Distribution (Pie Chart)
Slide 8: ML Models Comparison Table
Slide 9: Best Model Performance (Confusion Matrix)
Slide 10: Key Findings & Insights
Slide 11: Conclusion
Slide 12: Thank You + Q&A
```

**6.3 Poster (A3/A4):**
- Layout: 3 kolom
- Kolom 1: Introduction + Methodology
- Kolom 2: Visualisasi (WordCloud, Pie Chart, Model Comparison)
- Kolom 3: Results + Conclusion

**6.4 Video Presentasi (3-5 menit):**
- Format: MP4, 1080p
- Script narasi (800-1000 kata)
- Struktur:
  - 0:00-0:30 → Intro & Problem
  - 0:30-1:30 → Data & Method
  - 1:30-3:00 → Results & Visualization
  - 3:00-4:00 → Model Performance
  - 4:00-5:00 → Conclusion & Q&A prompt

***

### **BAGIAN 7: Timeline Eksekusi (BARU)**

| Tahap | Durasi | Output |
|-------|--------|--------|
| Data Cleaning & Preprocessing | 1 hari | Clean dataset CSV |
| EDA & Visualization | 1 hari | Grafik & insights |
| Sentiment Labeling | 0.5 hari | Labeled dataset |
| ML Modeling & Tuning | 1.5 hari | Trained models + metrics |
| Laporan Writing | 2 hari | Draft PDF |
| PPT & Poster Design | 1 hari | PDF + Poster |
| Video Recording | 0.5 hari | MP4 video |
| Review & Finalisasi | 0.5 hari | Final submission package |
| **TOTAL** | **8 hari** | **Complete deliverables** |

***

### **IMPROVEMENT HIGHLIGHTS dari Plan Lama:**

✅ **Judul kasus lebih spesifik** (fokus pada Liam Rosenior, bukan general Grok AI)  
✅ **Ditambahkan Tinjauan Pustaka** (standar laporan akademis)  
✅ **Feature Engineering lebih kaya** (engagement_score, temporal features)  
✅ **4 model ML** (bukan hanya 2) dengan hyperparameter tuning  
✅ **Evaluasi lebih komprehensif** (confusion matrix, ROC-AUC)  
✅ **Timeline eksekusi jelas** (8 hari breakdown)  
✅ **Deliverables lebih detail** (struktur slide by slide, layout poster)  
✅ **Preprocessing pipeline** yang lebih teknis & reproducible

***

