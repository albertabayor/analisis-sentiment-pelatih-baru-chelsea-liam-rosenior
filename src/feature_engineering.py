"""
Sentiment Analysis Project - Chelsea Liam Rosenior Appointment

Feature engineering and transformation functions.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def create_tfidf_features(X_train, X_test, max_features=500, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train (pd.Series): Training text data
        X_test (pd.Series): Test text data
        max_features (int): Maximum number of features
        ngram_range (tuple): Range of n-grams to extract
    
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ TF-IDF features created: {X_train_tfidf.shape[1]} features")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def create_count_features(X_train, X_test, max_features=500):
    """
    Create Count Vectorizer features (for Naive Bayes).
    
    Args:
        X_train (pd.Series): Training text data
        X_test (pd.Series): Test text data
        max_features (int): Maximum number of features
    
    Returns:
        tuple: (X_train_count, X_test_count, vectorizer)
    """
    print(f"Creating Count features (max_features={max_features})...")
    
    # Initialize Count vectorizer
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    # Fit and transform
    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test)
    
    print(f"✓ Count features created: {X_train_count.shape[1]} features")
    
    return X_train_count, X_test_count, vectorizer


def add_text_features(df, text_col='Tweet Content'):
    """
    Add additional text-based features to the dataframe.
    
    Features:
    - tweet_length: Character count
    - word_count: Number of words
    - hashtag_count: Number of hashtags
    - mention_count: Number of @mentions
    - url_count: Number of URLs
    - avg_word_length: Average word length
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
        text_col (str): Column name containing tweet text
    
    Returns:
        pd.DataFrame: Dataframe with added features
    """
    print("Adding text-based features...")
    
    # Ensure text column exists and is string
    df[text_col] = df[text_col].astype(str)
    
    # Character count
    df['tweet_length'] = df[text_col].apply(len)
    
    # Word count
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    
    # Hashtag count
    df['hashtag_count'] = df[text_col].apply(lambda x: x.count('#'))
    
    # Mention count
    df['mention_count'] = df[text_col].apply(lambda x: x.count('@'))
    
    # URL count
    df['url_count'] = df[text_col].apply(lambda x: x.count('http'))
    
    # Average word length (excluding empty words)
    def avg_word_length(text):
        words = text.split()
        if len(words) == 0:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    df['avg_word_length'] = df[text_col].apply(avg_word_length)
    
    print("✓ Text features added:")
    print(f"  - tweet_length, word_count, hashtag_count")
    print(f"  - mention_count, url_count, avg_word_length")
    
    return df


def add_temporal_features(df, date_col='Tweet Creation Date'):
    """
    Add temporal features from timestamp.
    
    Features:
    - hour_posted: Hour of the day (0-23)
    - day_of_week: Day of the week (0=Monday, 6=Sunday)
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
        date_col (str): Column name containing timestamp
    
    Returns:
        pd.DataFrame: Dataframe with temporal features
    """
    print("Adding temporal features...")
    
    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract hour and day of week
    df['hour_posted'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    print("✓ Temporal features added:")
    print(f"  - hour_posted: Hour of day (0-23)")
    print(f"  - day_of_week: Day of week (0=Monday, 6=Sunday)")
    
    return df


def prepare_features_for_ml(df, text_col='Tweet Content', use_text_features=False):
    """
    Prepare features for machine learning (text only or text + additional features).
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
        text_col (str): Column name containing tweet text
        use_text_features (bool): Whether to add additional text features
    
    Returns:
        pd.DataFrame: Dataframe with prepared features
    """
    df_prep = df.copy()
    
    if use_text_features:
        # Add text-based features
        df_prep = add_text_features(df_prep, text_col=text_col)
        
        # Add temporal features
        if 'Tweet Creation Date' in df_prep.columns:
            df_prep = add_temporal_features(df_prep)
    
    return df_prep


def get_feature_importance(vectorizer, model, n_top=10):
    """
    Get top features by importance (for models that support it).
    
    Args:
        vectorizer: Fitted vectorizer (TF-IDF or Count)
        model: Fitted model with feature_importances_ or coef_ attribute
        n_top (int): Number of top features to return
    
    Returns:
        pd.DataFrame: Top features with their importance scores
    """
    print(f"Extracting top {n_top} important features...")
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get importance scores
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For Logistic Regression, use absolute coefficients
        importances = np.abs(model.coef_[0])
    else:
        print("Warning: Model does not have feature_importances_ or coef_ attribute")
        return None
    
    # Create dataframe of feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort and get top features
    top_features = feature_importance.sort_values('importance', ascending=False).head(n_top)
    
    print("✓ Top features extracted")
    
    return top_features


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering functions...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Tweet Content': [
            "This is a great tweet! #chelsea @liam",
            "Not happy about the decision @grok",
            "I love football and sports",
        ],
        'Tweet Creation Date': ['2026-01-06T10:30:00', '2026-01-06T11:45:00', '2026-01-06T12:00:00']
    })
    
    # Add text features
    df_with_features = add_text_features(sample_data)
    print(df_with_features[['Tweet Content', 'tweet_length', 'word_count', 'hashtag_count', 'mention_count']])
    
    # Add temporal features
    df_with_temporal = add_temporal_features(sample_data)
    print(df_with_temporal[['Tweet Creation Date', 'hour_posted', 'day_of_week']])
