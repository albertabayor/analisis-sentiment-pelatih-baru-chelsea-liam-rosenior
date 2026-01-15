# Analisis Sentimen Reaksi Publik terhadap Penunjukan Liam Rosenior sebagai Pelatih Chelsea

## 📋 Project Overview
Sentiment analysis of 621 Twitter/X reactions to Liam Rosenior's appointment as Chelsea FC head coach in January 2026.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Upload this entire project folder to Google Drive
2. Open `notebooks/0_setup_colab.ipynb` in Google Colab
3. Run all cells to setup environment
4. Proceed with notebooks 1-5 in order

### Option 2: Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

## 📁 Project Structure

```
analisis-sentiment-pelatih-baru-chelsea-liam-rosenior/
├── data/
│   ├── raw/
│   │   └── tweets.csv                    # Original dataset (621 tweets)
│   ├── processed/
│   │   ├── tweets_cleaned.csv            # After cleaning (@grok filtered)
│   │   └── tweets_labeled.csv            # After VADER labeling
│   └── README.md                         # Data documentation
│
├── notebooks/
│   ├── 0_setup_colab.ipynb               # Colab setup (mount Drive, install)
│   ├── 1_data_preprocessing.ipynb        # Clean data, filter @grok
│   ├── 2_exploratory_analysis.ipynb      # EDA, WordCloud, statistics
│   ├── 3_sentiment_labeling.ipynb        # VADER + distribution analysis
│   ├── 4_ml_modeling.ipynb               # LR + NB models + evaluation
│   └── 5_results_visualization.ipynb      # Export charts for report
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Data cleaning functions
│   ├── feature_engineering.py            # TF-IDF, feature extraction
│   ├── models.py                         # Model training utilities
│   └── utils.py                          # Helper functions (paths, Colab check)
│
├── outputs/
│   ├── figures/                          # PNG/SVG charts for report
│   ├── tables/                           # CSV/PDF result tables
│   ├── models/                           # Saved model files (.pkl)
│   └── metrics/                          # Evaluation metrics JSON
│
├── reports/
│   ├── draft/                            # Report drafts
│   └── final/
│       ├── laporan_sentimen_analisis.pdf # Final report
│       ├── presentasi.pdf                # Final PPT
│       └── poster.pdf                    # Final poster
│
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
├── .gitignore                            # Git ignore rules
└── PLAN.md                               # Detailed project plan
```

## 📓 Notebooks Execution Order

1. **`0_setup_colab.ipynb`** - Initial environment setup (Google Drive mount, package installation)
2. **`1_data_preprocessing.ipynb`** - Clean and filter data (remove @grok queries, English filter)
3. **`2_exploratory_analysis.ipynb`** - EDA and visualization (WordCloud, engagement analysis)
4. **`3_sentiment_labeling.ipynb`** - Apply VADER sentiment analysis
5. **`4_ml_modeling.ipynb`** - Train and evaluate ML models (LR + NB)
6. **`5_results_visualization.ipynb`** - Export final visualizations for report

## 📊 Dataset

- **Source**: X (Twitter) - 6 January 2026
- **Original**: 621 tweets
- **Expected after cleaning**: ~250-300 tweets (after filtering @grok queries and non-English)
- **Features**: Tweet Content, Engagement metrics (Views, Likes, Retweets, Replies), Timestamps

## 🤖 Models

### Machine Learning Models
- **Logistic Regression** (Required)
- **Multinomial Naive Bayes**

### Feature Extraction
- **TF-IDF Vectorization** (max_features=500, ngram_range=(1,2))
- Additional features: word_count, char_count, hashtag_count, mention_count, engagement_score

### Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Classification Report

## 📦 Dependencies

```txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
vaderSentiment==3.3.2
langdetect==1.0.9
matplotlib==3.7.2
seaborn==0.12.2
wordcloud==1.9.2
plotly==5.15.0
jupyter==1.0.0
ipython==8.14.0
```

## 📝 Deliverables

### Analysis
- [x] Data preprocessing notebook
- [x] EDA notebook with visualizations
- [x] Sentiment labeling notebook
- [x] ML modeling notebook
- [ ] Results visualization notebook

### Academic
- [ ] Final report (PDF) - 15-20 halaman
- [ ] Presentation (PPT PDF) - 10-12 slides
- [ ] Poster (A3/A4) - 3 kolom layout
- [ ] Video presentation (MP4) - 3-5 menit

## 👥 Team
- **Anggota Kelompok**: [Nama Anda]

## 📅 Timeline (10-12 Days for Solo Execution)

| Day | Task | Output |
|-----|------|--------|
| 1 | Data cleaning & preprocessing | Clean CSV (~250-300 rows) |
| 2 | EDA & VADER labeling | Visualizations + labeled dataset |
| 3-4 | LR + NB models with tuning | Trained models + metrics |
| 5-6 | Draft report (Sections I-IV) | Report draft |
| 7 | Create PPT slides | PPT PDF |
| 8 | Design poster | Poster |
| 9 | Record video + editing | MP4 video |
| 10-11 | Review, refine, finalize | Complete package |
| 12 | Final QA | Ready for submission |

## 🔧 Python Utility Modules

### `src/utils.py`
- `get_project_root()` - Detect local or Colab environment
- `get_data_path()` - Return correct data path
- `setup_colab_drive()` - Mount Drive if on Colab
- `install_requirements()` - Install packages in Colab
- `ensure_directories()` - Create output directories if missing

### `src/preprocessing.py`
- `remove_empty_tweets(df)` - Remove rows with empty tweet content
- `remove_duplicates(df)` - Remove duplicate tweets
- `filter_grok_queries(df)` - Remove questions TO @grok
- `filter_english_language(df)` - Detect and filter English tweets
- `clean_text(text)` - Full preprocessing pipeline
- `add_engagement_features(df)` - Add engagement_score, etc.

### `src/feature_engineering.py`
- `create_tfidf_features(X_train, X_test)` - TF-IDF vectorization
- `add_text_features(df)` - word_count, char_count, hashtag_count, mention_count
- `prepare_features(df)` - Combine TF-IDF + additional features

### `src/models.py`
- `train_logistic_regression(X_train, y_train)` - Train LR model
- `train_naive_bayes(X_train, y_train)` - Train NB model
- `evaluate_model(model, X_test, y_test)` - Return evaluation metrics
- `plot_confusion_matrix(y_true, y_pred)` - Visualize confusion matrix
- `compare_models(results_dict)` - Compare model performances
- `save_model(model, filename)` - Save trained model

## 🚫 .gitignore Rules

The following are ignored (not committed to git):
- `data/processed/` - Cleaned and labeled data
- `outputs/` - Figures, tables, models, metrics
- `reports/draft/` and `reports/final/` - Report files
- `venv/` - Virtual environment
- `.ipynb_checkpoints/` - Jupyter notebook checkpoints
- Python cache files

Only these are committed:
- `data/raw/tweets.csv` - Original dataset
- `notebooks/*.ipynb` - Jupyter notebooks
- `src/*.py` - Python utility modules
- `requirements.txt` - Dependencies
- Documentation files

## 🔑 Google Colab Compatibility

This project is fully compatible with Google Colab:

1. **Dynamic Path Handling**: Automatically detects Colab environment
2. **Google Drive Integration**: Mounts Drive for persistent storage
3. **Automatic Package Installation**: Installs requirements in Colab
4. **Data Persistence**: Processed data saved to Drive

## 📄 License

Academic project for coursework. All rights reserved.

## 🤝 Contributing

This is an individual project. No contributions accepted.

## 📧 Contact

For questions about this project, contact the team member listed above.

---

**Created**: January 2026
**Last Updated**: January 2026
