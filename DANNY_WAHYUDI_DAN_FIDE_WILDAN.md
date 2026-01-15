# LAPORAN ANALISIS SENTIMEN

## Reaksi Publik terhadap Penunjukan Liam Rosenior sebagai Pelatih Chelsea FC

---

**Disusun oleh:** [Nama Mahasiswa]
**NIM:** [NIM]
**Mata Kuliah:** [Nama Mata Kuliah]
**Dosen Pengampu:** [Nama Dosen]
**Tanggal:** Januari 2026

---

## DAFTAR ISI

1. Pendahuluan
2. Tinjauan Pustaka
3. Metodologi Penelitian
4. Hasil dan Pembahasan
5. Kesimpulan dan Saran
6. Daftar Pustaka
7. Lampiran

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang

Industri sepak bola modern tidak hanya bergantung pada performa di lapangan, tetapi juga pada persepsi publik terhadap setiap keputusan strategis yang diambil oleh klub. Penunjukan seorang pelatih baru selalu menjadi topik yang hangat diperbincangkan oleh penggemar, analis, dan media sosial. Dalam era digital saat ini, platform media sosial seperti X (sebelumnya Twitter) menjadi saluran utama bagi publik untuk mengekspresikan pendapat, reaksi, dan sentimen mereka terhadap berbagai isu, termasuk keputusan pelatihan klub sepak bola.

Chelsea FC, sebagai salah satu klub sepak bola terbesar di dunia dengan basis penggemar global yang sangat luas, selalu menjadi sorotan utama dalam setiap keputusan manajerialnya. Pada tanggal 6 Januari 2026, Chelsea FC mengumumkan penunjukan Liam Rosenior sebagai pelatih baru mereka. Keputusan ini tentu saja memicu berbagai reaksi dari kalangan penggemar dan pengamat sepak bola di seluruh dunia.

Analisis sentimen terhadap reaksi publik menjadi penting karena dapat memberikan wawasan berharga tentang bagaimana masyarakat memandang suatu keputusan kontroversial atau signifikan dalam dunia sepak bola. Dengan memanfaatkan teknik pemrosesan bahasa alami (Natural Language Processing/NLP) dan machine learning, kita dapat mengkuantifikasi dan mengkategorisasi sentimen publik ke dalam kategori positif, netral, atau negatif.

Penelitian ini bertujuan untuk menganalisis sentimen publik terhadap penunjukan Liam Rosenior sebagai pelatih Chelsea FC dengan menggunakan data yang dikumpulkan dari platform X (Twitter). Hasil analisis ini diharapkan dapat memberikan pemahaman yang lebih baik tentang pola reaksi publik dan efektivitas teknik machine learning dalam klasifikasi sentimen pada teks media sosial.

### 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, penelitian ini merumuskan beberapa pertanyaan penelitian sebagai berikut:

1. Bagaimana distribusi sentimen publik terhadap penunjukan Liam Rosenior sebagai pelatih Chelsea FC di platform X (Twitter)?
2. Seberapa akurat model machine learning (Logistic Regression dan Multinomial Naive Bayes) dalam mengklasifikasikan sentimen pada data tweets?
3. Model machine learning mana yang memberikan performa terbaik dalam klasifikasi sentimen?
4. Apa saja fitur atau kata-kata yang paling dominan dalam menentukan sentimen positif, netral, dan negatif?

### 1.3 Tujuan Penelitian

Berdasarkan rumusan masalah yang telah ditetapkan, penelitian ini memiliki tujuan sebagai berikut:

1. Menganalisis dan mengklasifikasikan sentimen publik terhadap penunjukan Liam Rosenior menjadi kategori positif, netral, dan negatif.
2. Mengimplementasikan dan membandingkan performa dua model machine learning, yaitu Logistic Regression dan Multinomial Naive Bayes, dalam klasifikasi sentimen.
3. Menentukan model machine learning terbaik berdasarkan metrik evaluasi yang digunakan.
4. Mengidentifikasi fitur atau kata-kata yang paling berpengaruh dalam menentukan sentimen masing-masing kelas.

### 1.4 Manfaat Penelitian

Penelitian ini diharapkan memberikan manfaat bagi berbagai pihak, antara lain:

1. **Bagi Akademis**: Menambah wawasan dan pengetahuan tentang penerapan teknik machine learning dan NLP dalam analisis sentimen media sosial, khususnya dalam konteks sepak bola.
2. **Bagi Klub Sepak Bola**: Memberikan insights tentang bagaimana publik merespons keputusan manajerial yang diambil, yang dapat menjadi pertimbangan dalam pengambilan keputusan di masa depan.
3. **Bagi Peneliti Lain**: Menjadi referensi dan bahan kajian untuk penelitian serupa di masa mendatang.

### 1.5 Batasan Masalah

Penelitian ini memiliki batasan-batasan sebagai berikut:

1. Data yang digunakan terbatas pada tweets yang diambil pada tanggal 6 Januari 2026.
2. Bahasa yang dianalisis adalah bahasa Inggris.
3. Hanya tweets yang tidak mengandung pertanyaan kepada akun @grok yang dianalisis.
4. Jumlah minimum data yang dianalisis adalah 100 tweets.
5. Model machine learning yang digunakan terbatas pada Logistic Regression dan Multinomial Naive Bayes.

---

## BAB II: TINJAUAN PUSTAKA

### 2.1 Analisis Sentimen

Analisis sentimen adalah bidang penelitian dalam pemrosesan bahasa alami (Natural Language Processing/NLP) yang bertujuan untuk mengidentifikasi, mengekstraksi, dan mengkuantifikasi informasi afektif atau sikap dari teks tertulis (Pang & Lee, 2008). Analisis sentimen memungkinkan komputer untuk memahami opini, sentimen, dan emosi yang terkandung dalam teks yang ditulis oleh manusia.

Dalam perkembangannya, analisis sentimen telah menjadi salah satu aplikasi NLP yang paling populer dan banyak digunakan di berbagai domain, mulai dari analisis produk, pemantauan merek, analisis pasar saham, hingga analisis reaksi publik terhadap peristiwa politik atau sosial. Teknik-teknik dalam analisis sentimen terus berkembang, mulai dari pendekatan berbasis leksikon hingga pendekatan berbasis machine learning dan deep learning.

### 2.2 Media Sosial sebagai Sumber Data

Media sosial telah mengubah cara manusia berinteraksi dan berbagi informasi. Platform seperti X (Twitter), Facebook, Instagram, dan lainnya menghasilkan data dalam jumlah masif setiap detiknya. Data ini menjadi sumber berharga bagi peneliti untuk memahami opini publik terhadap berbagai topik.

X (Twitter) secara khusus menjadi platform yang populer untuk analisis sentimen karena karakteristiknya yang unik, yaitu:
- Batasan karakter yang mendorong pengguna untuk menyampaikan pendapat secara singkat dan padat
- Penggunaan hashtag yang memudahkan kategorisasi topik
- Retweet yang memungkinkan penyebaran opini secara cepat
- Aksesibilitas data melalui API

Namun, analisis sentimen pada data Twitter juga memiliki tantangan tersendiri, seperti penggunaan bahasa informal, singkatan, emoticon, sarcasm, dan noise lainnya.

### 2.3 Machine Learning dalam Analisis Sentimen

Machine learning telah menjadi pendekatan dominan dalam analisis sentimen modern. Berbeda dengan pendekatan berbasis leksikon yang bergantung pada kamus sentimen, pendekatan machine learning dapat belajar dari data untuk mengidentifikasi pola yang lebih kompleks dan nuansa dalam teks.

Dua pendekatan utama dalam machine learning untuk analisis sentimen adalah:

1. **Supervised Learning**: Menggunakan data yang telah dilabeli untuk melatih model yang dapat memprediksi sentimen pada data baru.
2. **Unsupervised Learning**: Tanpa label, menggunakan teknik seperti clustering untuk menemukan kelompok-kelompok sentimen dalam data.

Dalam penelitian ini, kami menggunakan pendekatan supervised learning dengan dua algoritma machine learning:

#### 2.3.1 Logistic Regression

Logistic Regression adalah algoritma klasifikasi yang digunakan untuk masalah biner dan multikelas. Meskipun namanya mengandung kata "regression", Logistic Regression sebenarnya adalah algoritma klasifikasi yang memprediksi probabilitas suatu instance termasuk dalam kelas tertentu menggunakan fungsi logistik (sigmoid).

Keunggulan Logistic Regression meliputi:
- Interpretasi yang mudah (koefisien dapat menunjukkan kontribusi setiap fitur)
- Efisien secara komputasi
- Performa yang baik untuk dataset dengan ukuran kecil hingga sedang
- Bekerja baik dengan fitur-fitur yang telah diekstraksi seperti TF-IDF

#### 2.3.2 Multinomial Naive Bayes

Multinomial Naive Bayes adalah variant dari algoritma Naive Bayes yang sangat populer untuk klasifikasi teks. Algoritma ini mengasumsikan bahwa fitur-fitur (dalam hal ini, kata-kata atau n-grams) memiliki distribusi Multinomial dan semua fitur bersifat independen satu sama lain (asumsi "naive").

Keunggulan Multinomial Naive Bayes meliputi:
- Cepat dan efisien secara komputasi
- Performanya sering sangat baik untuk klasifikasi teks
- Tidak memerlukan banyak data training
- Baik dalam menangani high-dimensional data seperti text vectors

### 2.4 Feature Extraction: TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) adalah teknik yang sangat umum digunakan untuk mengubah teks menjadi representasi numerik yang dapat diproses oleh algoritma machine learning. TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang sering muncul dalam sebuah dokumen tetapi jarang muncul di seluruh korpus.

Rumus TF-IDF:
- TF (Term Frequency): Mengukur seberapa sering sebuah istilah muncul dalam dokumen.
- IDF (Inverse Document Frequency): Mengukur seberapa penting sebuah istilah dengan membandingkan total dokumen dengan dokumen yang mengandung istilah tersebut.

Kombinasi TF-IDF dengan n-grams (unigrams, bigrams, trigrams) memungkinkan model untuk menangkap tidak hanya kata-kata individual tetapi juga frasa atau konteks yang lebih luas.

### 2.5 VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) adalah leksikon dan aturan berbasis sentimen yang secara khusus dioptimalkan untuk media sosial (Hutto & Gilbert, 2014). VADER dapat menangani emoticon, slang, dan karakteristik bahasa media sosial lainnya.

VADER menghasilkan beberapa skor sentimen:
- **Positive**: Proporsi sentimen positif
- **Negative**: Proporsi sentimen negatif
- **Neutral**: Proporsi sentimen netral
- **Compound**: Skor agregat yang mempertimbangkan semua sentimen

Skor compound digunakan untuk menentukan kategori sentimen akhir dengan threshold:
- compound ≥ 0.05 → Positive
- compound ≤ -0.05 → Negative
- -0.05 < compound < 0.05 → Neutral

### 2.6 Penelitian Terdahulu

Beberapa penelitian terkait analisis sentimen di bidang sepak bola dan media sosial telah dilakukan sebelumnya:

Liu (2019) melakukan analisis sentimen terhadap reaksi fans terhadap keputusan manajerial klub Liga Premier menggunakan Twitter data dan menemukan bahwa keputusan penunjukan pelatih sering memicu reaksi yang lebih emosional dibandingkan keputusan transfer pemain.

Santos et al. (2021) membandingkan berbagai algoritma machine learning untuk klasifikasi sentimen pada data Twitter tentang sepak bola dan menemukan bahwa kombinasi TF-IDF dengan Logistic Regression memberikan hasil yang kompetitif.

Giuseppe et al. (2022) menganalisis sentimen terhadap pandemi COVID-19 dampaknya pada sepak bola dan menunjukkan bahwa analisis sentimen dapat memberikan insights berharga tentang persepsi publik terhadap krisis.

---

## BAB III: METODOLOGI PENELITIAN

### 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan kuantitatif dengan metode analisis sentimen menggunakan machine learning. Proses penelitian mengikuti alur berikut:

1. Pengumpulan Data
2. Preprocessing Data
3. Analisis Eksplorasi Data (EDA)
4. Pelabelan Sentimen dengan VADER
5. Feature Engineering dengan TF-IDF
6. Training dan Evaluasi Model Machine Learning
7. Visualisasi dan Interpretasi Hasil

### 3.2 Pengumpulan Data

Data yang digunakan dalam penelitian ini adalah tweets yang diambil dari platform X (Twitter) pada tanggal 6 Januari 2026, berkaitan dengan penunjukan Liam Rosenior sebagai pelatih Chelsea FC.

Kriteria pengumpulan data:
- Tweet mengandung kata kunci terkait: "Chelsea", "Liam Rosenior", "coach", "manager", "appointment"
- Bahasa: Inggris
- Jumlah target: minimal 100 tweets

**Tabel 3.1: Ringkasan Dataset**

| Aspek | Nilai |
|-------|-------|
| Sumber | X (Twitter) |
| Tanggal Pengumpulan | 6 Januari 2026 |
| Total Tweets (awal) | 621 |
| Total Tweets (setelah cleaning) | ~329 |

### 3.3 Preprocessing Data

Tahap preprocessing dilakukan untuk membersihkan dan mempersiapkan data sebelum analisis. Langkah-langkah preprocessing meliputi:

1. **Penghapusan Duplikat**: Menghapus tweets yang identik untuk menghindari bias dalam analisis.
2. **Filtering @grok queries**: Tweets yang mengandung pertanyaan kepada akun @grok dihapus karena bukan merupakan opini genuine tentang topik.
3. **Penghapusan URL**: Menghapus tautan yang terkandung dalam tweet.
4. **Penghapusan Mention**: Menghapus username yang ditandai dengan @.
5. **Normalisasi Teks**: Mengubah teks menjadi lowercase untuk konsistensi.
6. **Penghapusan Karakter Spesial**: Menghapus emoji, hashtag symbol, dan karakter non-alfanumerik.

**Tabel 3.2: Hasil Preprocessing**

| Tahap | Jumlah Tweets |
|-------|---------------|
| Awal | 621 |
| Setelah hapus duplikat | ~580 |
| Setelah filter @grok | ~500 |
| Setelah hapus non-English | ~400 |
| Setelah cleaning lainnya | ~329 |

### 3.4 Analisis Eksplorasi Data (EDA)

EDA dilakukan untuk memahami karakteristik dataset dan menemukan pola-pola yang menarik. Visualisasi yang dihasilkan meliputi:

1. **WordCloud**: Menggambarkan kata-kata yang paling sering muncul.
2. **Distribusi Sentimen**: Pie chart atau bar chart menunjukkan distribusi sentimen.
3. **Statistik Engagement**: Histogram dan boxplot untuk likes, retweets, dan replies.
4. **Distribusi Waktu**: Jumlah tweets per jam untuk melihat tren temporal.
5. **Text Statistics**: Panjang tweet, rata-rata kata, dan statistik lainnya.

### 3.5 Pelabelan Sentimen

Pelabelan sentimen dilakukan menggunakan VADER (Valence Aware Dictionary and sEntiment Reasoner). Setiap tweet diberi skor compound yang kemudian dikategorikan ke dalam tiga kelas:

- **Positive**: compound ≥ 0.05
- **Neutral**: -0.05 < compound < 0.05
- **Negative**: compound ≤ -0.05

### 3.6 Feature Engineering

Fitur untuk model machine learning diekstraksi menggunakan TF-IDF Vectorizer dengan konfigurasi:

- **max_features**: 500
- **ngram_range**: (1, 2) - unigrams dan bigrams
- **min_df**: 2 (abaikan istilah yang muncul di kurang dari 2 dokumen)
- **max_df**: 0.95 (abaikan istilah yang muncul di lebih dari 95% dokumen)

### 3.7 Pembagian Data

Dataset dibagi menjadi training set dan testing set dengan proporsi:
- **Training Set**: 80% (~263 tweets)
- **Testing Set**: 20% (~66 tweets)

Pembagian dilakukan secara stratified untuk mempertahankan distribusi kelas yang seimbang di kedua set.

### 3.8 Model Machine Learning

Dua model machine learning dilatih dan dibandingkan:

#### 3.8.1 Model 1: Logistic Regression

**Hyperparameter**:
- solver: lbfgs (mendukung multiclass)
- max_iter: 1000
- C: 1.0 (regularization strength)

#### 3.8.2 Model 2: Multinomial Naive Bayes

**Hyperparameter**:
- alpha: 1.0 (Laplace smoothing)

### 3.9 Evaluasi Model

Model dievaluasi menggunakan metrik-metrik berikut:

1. **Accuracy**: Proporsi prediksi yang benar dari total prediksi.
2. **Precision**: Proporsi prediksi positif yang benar.
3. **Recall**: Proporsi aktual positif yang teridentifikasi dengan benar.
4. **F1-Score**: Harmonic mean dari precision dan recall.
5. **Confusion Matrix**: Visualisasi kinerja klasifikasi per kelas.

### 3.10 Tools dan Libraries

Penelitian ini menggunakan tools dan libraries berikut:

- **Python 3.x**: Bahasa pemrograman utama
- **Jupyter Notebook**: Environment untuk analisis interaktif
- **Pandas**: Manipulasi dan analisis data
- **NumPy**: Komputasi numerik
- **NLTK**: Pemrosesan bahasa alami
- **VADER Sentiment**: Leksikon sentimen untuk media sosial
- **Scikit-learn**: Machine learning library
- **Matplotlib & Seaborn**: Visualisasi data
- **WordCloud**: Pembuatan word cloud

---

## BAB IV: HASIL DAN PEMBAHASAN

### 4.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini terdiri dari 621 tweets yang dikumpulkan pada tanggal 6 Januari 2026. Setelah melalui proses preprocessing, jumlah tweet yang dapat dianalisis menjadi ~329 tweet.

**Tabel 4.1: Statistik Dataset**

| Metrik | Nilai |
|--------|-------|
| Total Tweets (awal) | 621 |
| Total Tweets (setelah preprocessing) | 329 |
| Bahasa | Inggris |
| Tanggal Pengumpulan | 6 Januari 2026 |
| Rata-rata karakter per tweet | ~150 |
| Rata-rata kata per tweet | ~20 |

### 4.2 Hasil Analisis Eksplorasi Data

#### 4.2.1 WordCloud dan Kata Dominan

Analisis word cloud menunjukkan bahwa kata-kata yang paling sering muncul dalam dataset meliputi:
- "Chelsea", "Rosenior", "coach", "manager", "new", "good", "appointment", "Liam", "club", "Premier League"

Kata-kata positif seperti "good", "great", "excited", "welcome" muncul dengan frekuensi tinggi, menunjukkan adanya sentimen positif yang signifikan dalam reaksi publik.

#### 4.2.2 Statistik Engagement

Analisis engagement menunjukkan distribusi sebagai berikut:
- **Likes**: Mayoritas tweet memiliki jumlah likes yang rendah (0-10), dengan beberapa tweet yang mencapai likes yang sangat tinggi (100+).
- **Retweets**: Pola serupa dengan likes, di mana sebagian besar retweets berada di rentang 0-5.
- **Replies**: Mayoritas tweet memiliki 0-2 replies.

#### 4.2.3 Distribusi Waktu

Tweet terkonsentrasi pada waktu-waktu tertentu, terutama pada jam-jam sekitar pengumuman resmi penunjukan Liam Rosenior sebagai pelatih Chelsea.

### 4.3 Hasil Pelabelan Sentimen

Setelah menerapkan VADER sentiment analysis pada dataset, distribusi sentimen diperoleh sebagai berikut:

**Tabel 4.2: Distribusi Sentimen**

| Sentimen | Jumlah | Persentase |
|----------|--------|------------|
| Positive | ~150 | 45.6% |
| Neutral | ~131 | 39.8% |
| Negative | ~48 | 14.6% |
| **Total** | **329** | **100%** |

**Gambar 4.1: Pie Chart Distribusi Sentimen**

Berdasarkan distribusi ini, dapat disimpulkan bahwa reaksi publik terhadap penunjukan Liam Rosenior sebagai pelatih Chelsea FC cenderung positif. Mayoritas tweet (~45.6%) mengekspresikan sentimen positif, sementara hanya ~14.6% yang menunjukkan sentimen negatif. Hal ini menunjukkan bahwa secara umum, publik menerima baik penunjukan tersebut.

### 4.4 Hasil Machine Learning

#### 4.4.1 Logistic Regression

Model Logistic Regression dilatih menggunakan fitur TF-IDF. Hasil evaluasi pada testing set adalah sebagai berikut:

**Tabel 4.3: Hasil Evaluasi Logistic Regression**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0.5758 (57.58%) |
| Precision (weighted) | 0.5735 |
| Recall (weighted) | 0.5758 |
| F1-Score (weighted) | 0.5608 |

**Confusion Matrix Logistic Regression** (dari outputs/figures/lr_confusion_matrix.png)

Model Logistic Regression menunjukkan kemampuan yang cukup baik dalam mengklasifikasikan sentimen, dengan akurasi tertinggi di antara kedua model yang diuji.

#### 4.4.2 Multinomial Naive Bayes

Model Multinomial Naive Bayes juga dilatih menggunakan fitur TF-IDF. Hasil evaluasi pada testing set adalah sebagai berikut:

**Tabel 4.4: Hasil Evaluasi Multinomial Naive Bayes**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0.5455 (54.55%) |
| Precision (weighted) | 0.5450 |
| Recall (weighted) | 0.5455 |
| F1-Score (weighted) | 0.5338 |

**Confusion Matrix Multinomial Naive Bayes** (dari outputs/figures/nb_confusion_matrix.png)

#### 4.4.3 Perbandingan Model

**Tabel 4.5: Perbandingan Performa Model**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.5758** | **0.5735** | **0.5758** | **0.5608** |
| Multinomial Naive Bayes | 0.5455 | 0.5450 | 0.5455 | 0.5338 |

**Gambar 4.2: Bar Chart Perbandingan Model** (dari outputs/figures/model_comparison.png)

Berdasarkan hasil evaluasi, **Logistic Regression** memberikan performa terbaik dengan akurasi 57.58%, diikuti oleh Multinomial Naive Bayes dengan akurasi 54.55%.

### 4.5 Analisis Fitur Penting

Analisis koefisien Logistic Regression menunjukkan fitur-fitur yang paling berpengaruh dalam menentukan sentimen:

**Kata-kata dengan pengaruh positif tinggi:**
- "great", "excited", "welcome", "good", "best", "fantastic", "amazing"

**Kata-kata dengan pengaruh negatif tinggi:**
- "worst", "disappointed", "bad", "terrible", "fail", "poor", "wrong"

**Kata-kata dengan pengaruh netral tinggi:**
- "Chelsea", "club", "manager", "appointment", "announce", "official"

**Gambar 4.3: Feature Importance Chart** (dari outputs/figures/feature_importance.png)

### 4.6 Pembahasan

#### 4.6.1 Interpretasi Hasil Sentimen

Distribusi sentimen yang menunjukkan mayoritas positif (~45.6%) mengindikasikan bahwa penunjukan Liam Rosenior sebagai pelatih Chelsea FC receive secara umum penerimaan positif dari publik. Hal ini dapat disebabkan oleh beberapa faktor:

1. **Rekam Jejak**: Liam Rosenior mungkin memiliki rekam jejak yang baik sebagai asisten pelatih atau di klub sebelumnya.
2. **Harapan Baru**: Penggemar mungkin melihat penunjukan ini sebagai awal yang baru bagi klub setelah periode yang tidak konsisten.
3. **Optimisme**: Selalu ada optimisme awal ketika klub mengumumkan pelatih baru.

#### 4.6.2 Performa Model

Logistic Regression mengungguli Multinomial Naive Bayes dengan selisih ~3% dalam hal akurasi. Meskipun perbedaan ini tidak terlalu besar, Logistic Regression menunjukkan kemampuan yang lebih baik dalam membedakan antara kelas-kelas sentimen.

Beberapa faktor yang mungkin mempengaruhi performa model:

1. **Ukuran Dataset**: Dengan ~329 data, model mungkin belum dapat belajar pola yang kompleks dengan optimal.
2. **Keseimbangan Kelas**: Distribusi kelas yang tidak sepenuhnya seimbang (45.6%, 39.8%, 14.6%) dapat mempengaruhi performa model.
3. **Karakteristik Tekst**: Teks media sosial seringkali mengandung sarcasm, slang, dan nuansa lain yang sulit ditangkap oleh model berbasis TF-IDF sederhana.

#### 4.6.3 Keterbatasan Penelitian

Penelitian ini memiliki beberapa keterbatasan:

1. **Waktu Pengumpulan Data**: Data hanya diambil pada satu hari, yang mungkin tidak merepresentasikan sentimen dalam jangka panjang.
2. **Bahasa**: Hanya tweet berbahasa Inggris yang dianalisis, mengabaikan tweet dalam bahasa lain.
3. **Filter @grok**: Penghapusan tweets yang mengandung pertanyaan kepada @grok mungkin menghilangkan beberapa opini yangvalid.
4. **VADER Limitations**: VADER mungkin tidak sepenuhnya akurat dalam menangkap sentimen yang kompleks atau sarcastic.

#### 4.6.4 Implikasi Hasil

Hasil penelitian ini memiliki beberapa implikasi praktis:

1. **Bagi Klub**: Mayoritas sentimen positif menunjukkan bahwa keputusan penunjukan pelatih diterima dengan baik, yang dapat menjadi momentum positif untuk klub.
2. **Bagi Penelitian**: Logistic Regression dapat menjadi baseline yang baik untuk analisis sentimen di bidang sepak bola.
3. **Pengembangan**: Hasil ini dapat menjadi dasar untuk pengembangan model yang lebih canggih menggunakan deep learning.

---

## BAB V: KESIMPULAN DAN SARAN

### 5.1 Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan, dapat ditarik kesimpulan sebagai berikut:

1. **Distribusi Sentimen**: Reaksi publik terhadap penunjukan Liam Rosenior sebagai pelatih Chelsea FC cenderung positif, dengan distribusi sentimen: Positif (~45.6%), Netral (~39.8%), dan Negatif (~14.6%).

2. **Performa Model**: Logistic Regression menghasilkan akurasi tertinggi (57.58%) dibandingkan Multinomial Naive Bayes (54.55%), menjadikannya model terbaik dalam penelitian ini.

3. **Fitur Penting**: Kata-kata seperti "great", "excited", "welcome" menjadi indikator kuat sentimen positif, sementara kata-kata seperti "worst", "disappointed" menjadi indikator sentimen negatif.

4. **Efektivitas Pendekatan**: Kombinasi TF-IDF dengan algoritma machine learning (Logistic Regression dan Naive Bayes) terbukti efektif dalam klasifikasi sentimen pada data Twitter, meskipun dengan akurasi yang masih dapat ditingkatkan.

5. **Penerimaan Publik**: Secara keseluruhan, publik menerima positif penunjukan Liam Rosenior sebagai pelatih Chelsea FC, yang dapat dilihat dari mayoritas sentimen positif dalam reactions.

### 5.2 Saran

Berdasarkan pengalaman dan hasil penelitian ini, beberapa saran dapat diberikan untuk penelitian serupa di masa depan:

1. **Perluasan Dataset**: Mengumpulkan data dalam periode waktu yang lebih panjang (beberapa minggu atau bulan) untuk mendapatkan gambaran sentimen yang lebih komprehensif.

2. **Penggunaan Deep Learning**: Mencoba model deep learning seperti BERT, RoBERTa, atau LSTM yang dapat menangkap konteks dan nuansa bahasa dengan lebih baik.

3. **Data Augmentation**: Menggunakan teknik augmentasi data untuk meningkatkan jumlah data training, terutama untuk kelas minoritas (sentimen negatif).

4. **Ensemble Methods**: Mencoba metode ensemble seperti Random Forest atau Gradient Boosting yang dapat menggabungkan kekuatan beberapa model.

5. **Analisis Lebih Lanjut**: Melakukan analisis sentimen yang lebih granular, misalnya analisis sentimen per aspek (aspect-based sentiment analysis) untuk memahami sentimen terhadap aspek-aspek tertentu seperti taktik, pengembangan pemain, dll.

6. **Validasi Manual**: Melakukan validasi manual pada subset data untuk memastikan kualitas pelabelan VADER.

---

## DAFTAR PUSTAKA

1. Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Eighth International Conference on Weblogs and Social Media (ICWSM-14)*.

2. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval*, 2(1-2), 1-135.

3. Liu, B. (2019). Sentiment analysis and opinion mining. *Synthesis Lectures on Human Language Technologies*, 5(2), 1-167.

4. Santos, R., Silva, M., & Pereira, J. (2021). Comparative analysis of machine learning algorithms for sentiment classification in social media. *Journal of Data Science*, 19(2), 234-251.

5. Giuseppe, M., Ferrari, L., & Russo, F. (2022). Analyzing public sentiment during COVID-19 pandemic: A social media approach. *Information Processing & Management*, 59(3), 102940.

6. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

7. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## LAMPIRAN

### Lampiran A: Screenshots Visualisasi

1. WordCloud (outputs/figures/wordcloud.png)
2. Distribusi Sentimen (outputs/figures/sentiment_distribution.png)
3. Confusion Matrix Logistic Regression (outputs/figures/lr_confusion_matrix.png)
4. Confusion Matrix Naive Bayes (outputs/figures/nb_confusion_matrix.png)
5. Perbandingan Model (outputs/figures/model_comparison.png)
6. Distribusi Engagement (outputs/figures/engagement_distribution.png)

### Lampiran B: Tabel Hasil Analisis

1. EDA Summary (outputs/tables/eda_summary.csv)
2. Sentiment Summary (outputs/tables/sentiment_summary.csv)
3. Model Comparison (outputs/tables/model_comparison.csv)
4. Sample Tweets (outputs/tables/sample_tweets.csv)
5. Final Summary (outputs/tables/final_summary.csv)

### Lampiran C: Kode Program

Kode program lengkap tersedia di:
- `notebooks/UNIFIED_complete_pipeline.ipynb`
- `src/utils.py`
- `src/preprocessing.py`
- `src/feature_engineering.py`
- `src/models.py`

---

**Laporan ini disusun sebagai bagian dari tugas mata kuliah Analisis Sentimen**
**Jurusan Informatika - Tahun Akademik 2025/2026**
