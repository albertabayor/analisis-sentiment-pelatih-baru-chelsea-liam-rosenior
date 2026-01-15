# Data Directory

This directory contains the dataset for sentiment analysis.

## Structure

- `raw/` - Original, unmodified data (committed to git)
- `processed/` - Cleaned and labeled data (not committed)

## Files

### `raw/tweets.csv`
- **Source**: X (Twitter) - 6 January 2026
- **Rows**: 621 tweets
- **Columns**:
  - Tweet Link
  - Author Handle
  - Tweet Content
  - Views
  - Likes
  - Retweets
  - Replies
  - Tweet Creation Date
  - Scraped Date

### `processed/tweets_cleaned.csv`
- Cleaned dataset after preprocessing
- Expected: ~250-300 tweets
- Processing steps:
  - Remove empty tweets
  - Remove duplicates
  - Filter out @grok queries
  - Filter English language only
  - Apply text cleaning pipeline

### `processed/tweets_labeled.csv`
- Cleaned dataset with VADER sentiment labels
- Columns added:
  - `vader_score` - Compound sentiment score (-1 to +1)
  - `sentiment` - Categorical: Positive/Neutral/Negative

## Data Notes

- Many tweets contain @grok mentions (questions TO @grok AI)
- These should be filtered out as they don't represent genuine opinions
- Original dataset includes multilingual content (English filter needed)
- Engagement metrics available for analysis
