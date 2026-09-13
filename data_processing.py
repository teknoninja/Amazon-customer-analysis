import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/twcs.csv"  # The CSV file inside the downloaded zip (usually twcs/twcs.csv but often extracted to data/twcs.csv)
PROCESSED_DATA_PATH = "data/amazon_help_interactions.csv"
BRAND_ID = "AmazonHelp"

def process_data():
    data_path = DATA_PATH
    if not os.path.exists(data_path):
        alt_path = "data/twcs/twcs.csv"
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            logger.error(f"Dataset not found at {DATA_PATH} or {alt_path}. Please run download_data.py first.")
            return

    logger.info(f"Loading dataset from {data_path}...")
    # Read only required columns to save memory
    df = pd.read_csv(data_path, usecols=["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"])
    
    logger.info("Filtering for AmazonHelp responses...")
    # Get all tweets from AmazonHelp
    brand_tweets = df[df["author_id"] == BRAND_ID].copy()
    
    # We want customer questions that AmazonHelp responded to.
    # The customer question is the tweet that AmazonHelp replied to (in_response_to_tweet_id).
    # Some brand tweets are initial tweets (not replies), but we care about support interactions.
    
    # Get the IDs of the customer tweets that the brand replied to
    customer_tweet_ids = brand_tweets["in_response_to_tweet_id"].dropna().unique()
    
    logger.info("Fetching corresponding customer tweets...")
    customer_tweets = df[df["tweet_id"].isin(customer_tweet_ids)].copy()
    
    # Rename columns for clarity before joining
    brand_tweets = brand_tweets.rename(columns={
        "tweet_id": "brand_tweet_id",
        "text": "brand_text",
        "in_response_to_tweet_id": "customer_tweet_id"
    })[["brand_tweet_id", "brand_text", "customer_tweet_id"]]
    
    customer_tweets = customer_tweets.rename(columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_text",
        "author_id": "customer_author_id"
    })[["customer_tweet_id", "customer_text", "customer_author_id"]]
    
    logger.info("Merging customer questions with brand responses...")
    interactions = pd.merge(customer_tweets, brand_tweets, on="customer_tweet_id", how="inner")
    
    # Drop duplicates if multiple responses exist (just keep the first one for simplicity)
    interactions = interactions.drop_duplicates(subset=["customer_tweet_id"])
    
    logger.info(f"Total AmazonHelp interactions found: {len(interactions)}")
    
    # Simple rule-based intent classification for the baseline and exploration
    # In a real scenario, this would be more complex, but we need a 'small set of intents' defined from the data.
    # Intents: Shipping, Account/Prime, Product/Kindle, Refund/Return, General
    
    def classify_intent_heuristic(text):
        text = str(text).lower()
        if any(word in text for word in ["shipping", "delivery", "arrive", "tracking", "package", "delayed"]):
            return "Shipping"
        elif any(word in text for word in ["prime", "account", "login", "password", "charge", "charged", "subscription"]):
            return "Account/Billing"
        elif any(word in text for word in ["kindle", "alexa", "echo", "fire", "device", "app"]):
            return "Product/Device"
        elif any(word in text for word in ["refund", "return", "cancel", "money back"]):
            return "Refund/Return"
        else:
            return "General Support"
            
    logger.info("Applying heuristic intent labels...")
    interactions["heuristic_intent"] = interactions["customer_text"].apply(classify_intent_heuristic)
    
    interactions.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Saved processed data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_data()
