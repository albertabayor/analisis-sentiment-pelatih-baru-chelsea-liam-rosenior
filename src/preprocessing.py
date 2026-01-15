"""
Sentiment Analysis Project - Chelsea Liam Rosenior Appointment

Data preprocessing and cleaning functions.
"""

import re
import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
from src.utils import get_data_path, get_processed_data_path


def remove_empty_tweets(df):
    """
    Remove rows with empty or NaN tweet content.
    
    Args:
        df (pd.DataFrame): Original dataframe
    
    Returns:
        pd.DataFrame: Dataframe with empty tweets removed
        int: Number of tweets removed
    """
    initial_count = len(df)
    
    # Remove rows where Tweet Content is NaN or empty
    df_cleaned = df.dropna(subset=['Tweet Content'])
    df_cleaned = df_cleaned[df_cleaned['Tweet Content'].str.strip() != '']
    
    removed_count = initial_count - len(df_cleaned)
    print(f"✓ Removed {removed_count} empty tweets")
    
    return df_cleaned, removed_count


def remove_duplicates(df):
    """
    Remove duplicate tweets based on content.
    
    Args:
        df (pd.DataFrame): Dataframe with potential duplicates
    
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
        int: Number of duplicates removed
    """
    initial_count = len(df)
    
    # Remove duplicates based on Tweet Content
    df_unique = df.drop_duplicates(subset=['Tweet Content'], keep='first')
    
    removed_count = initial_count - len(df_unique)
    print(f"✓ Removed {removed_count} duplicate tweets")
    
    return df_unique, removed_count


def is_grok_query(tweet_content):
    """
    Check if a tweet is a question TO @grok (rather than about the topic).
    
    Patterns that indicate @grok query:
    - Starts with @grok
    - Contains @grok followed by question words (who, what, where, when, why, how)
    - Short tweets that are likely queries
    
    Args:
        tweet_content (str): Tweet content to check
    
    Returns:
        bool: True if tweet is a @grok query, False otherwise
    """
    if not isinstance(tweet_content, str):
        return False
    
    tweet_lower = tweet_content.lower().strip()
    
    # Check if tweet starts with @grok
    if tweet_lower.startswith('@grok'):
        return True
    
    # Check for @grok followed by question words
    if '@grok' in tweet_lower:
        question_words = ['who is', 'what is', 'where', 'when', 'why', 'how', 'can you',
                         'tell me', 'explain', 'describe']
        for qw in question_words:
            if qw in tweet_lower:
                return True
    
    return False


def filter_grok_queries(df):
    """
    Remove tweets that are questions TO @grok AI.
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
    
    Returns:
        pd.DataFrame: Dataframe without @grok queries
        int: Number of @grok queries removed
    """
    initial_count = len(df)
    
    # Identify @grok queries
    is_grok = df['Tweet Content'].apply(is_grok_query)
    
    # Remove @grok queries
    df_filtered = df[~is_grok]
    
    removed_count = initial_count - len(df_filtered)
    print(f"✓ Removed {removed_count} @grok query tweets")
    
    return df_filtered, removed_count


def is_english(text):
    """
    Detect if text is in English language.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        bool: True if English, False otherwise or if detection fails
    """
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If detection fails, assume non-English
        return False


def filter_english_language(df):
    """
    Filter tweets to keep only English language.
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
    
    Returns:
        pd.DataFrame: Dataframe with only English tweets
        int: Number of non-English tweets removed
    """
    initial_count = len(df)
    
    # Detect language for each tweet
    print("Detecting tweet languages...")
    is_english_mask = df['Tweet Content'].apply(is_english)
    
    # Filter to English only
    df_english = df[is_english_mask]
    
    removed_count = initial_count - len(df_english)
    print(f"✓ Removed {removed_count} non-English tweets")
    print(f"✓ Remaining English tweets: {len(df_english)}")
    
    return df_english, removed_count


def clean_text(text):
    """
    Clean and preprocess tweet text.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions (@username) - keeping for later analysis
    4. Remove hashtags (#hashtag) - keeping text only
    5. Remove special characters and punctuation
    6. Remove extra whitespaces
    7. Remove emojis (or convert to text)
    
    Args:
        text (str): Raw tweet text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ''
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # 3. Remove mentions (keep for analysis separately)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove hashtags (keep text only)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 5. Remove special characters and punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 6. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def add_engagement_features(df):
    """
    Add engagement-related features to the dataframe.
    
    Features added:
    - engagement_score: (Likes + Retweets*2 + Replies*3)
    - tweet_length: Character count of tweet
    - word_count: Number of words in tweet
    - hashtag_count: Number of hashtags in tweet
    - mention_count: Number of mentions in tweet
    
    Args:
        df (pd.DataFrame): Dataframe with tweets
    
    Returns:
        pd.DataFrame: Dataframe with added features
    """
    # Ensure numeric columns are properly typed
    numeric_cols = ['Views', 'Likes', 'Retweets', 'Replies']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Engagement score (weighted: Likes=1, Retweets=2, Replies=3)
    df['engagement_score'] = df['Likes'] + (df['Retweets'] * 2) + (df['Replies'] * 3)
    
    # Text features
    df['tweet_length'] = df['Tweet Content'].apply(lambda x: len(str(x)))
    df['word_count'] = df['Tweet Content'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df['Tweet Content'].apply(lambda x: str(x).count('#'))
    df['mention_count'] = df['Tweet Content'].apply(lambda x: str(x).count('@'))
    
    print("✓ Added engagement and text features")
    
    return df


def preprocess_pipeline(df, clean_text_content=False):
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Remove empty tweets
    2. Remove duplicates
    3. Filter out @grok queries
    4. Filter English language
    5. Add engagement features
    6. Clean text content (optional)
    
    Args:
        df (pd.DataFrame): Original dataframe
        clean_text_content (bool): Whether to apply text cleaning
    
    Returns:
        pd.DataFrame: Fully preprocessed dataframe
    """
    print("\n=== Starting Data Preprocessing Pipeline ===\n")
    print(f"Initial dataset: {len(df)} tweets\n")
    
    # Step 1: Remove empty tweets
    df, _ = remove_empty_tweets(df)
    
    # Step 2: Remove duplicates
    df, _ = remove_duplicates(df)
    
    # Step 3: Filter @grok queries
    df, _ = filter_grok_queries(df)
    
    # Step 4: Filter English language
    df, _ = filter_english_language(df)
    
    # Step 5: Add engagement features
    df = add_engagement_features(df)
    
    # Step 6: Clean text content (optional)
    if clean_text_content:
        print("\nCleaning tweet text...")
        df['Tweet Content Cleaned'] = df['Tweet Content'].apply(clean_text)
        print("✓ Text content cleaned")
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Final dataset: {len(df)} tweets\n")
    
    return df


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    # Load sample data
    df = pd.read_csv(get_data_path('tweets.csv'))
    print(f"Loaded {len(df)} tweets from {get_data_path('tweets.csv')}")
    
    # Run pipeline
    df_cleaned = preprocess_pipeline(df, clean_text_content=True)
    
    # Display sample
    print("\nSample cleaned tweets:")
    print(df_cleaned[['Tweet Content', 'Tweet Content Cleaned', 'engagement_score']].head())
