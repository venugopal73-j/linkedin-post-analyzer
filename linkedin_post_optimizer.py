"""
LinkedIn Post Optimizer - Streamlit Cloud Version (Enhanced)
"""

import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import random

# Environment config
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Initialize NLTK
def initialize_nltk_resources():
    try:
        for resource in ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords']:
            nltk.download(resource, quiet=True)
        return True
    except Exception as e:
        return f"Error initializing NLTK resources: {e}"

# Setup tools
analyzer = SentimentIntensityAnalyzer()
TRENDING_HASHTAGS = ['#AI', '#Productivity', '#Innovation', '#PromptEngineering', '#TechForGood']

# Metrics & Utilities
def flesch_kincaid(text):
    words = len(word_tokenize(text))
    sentences = len(sent_tokenize(text))
    if sentences == 0 or words == 0:
        return 0
    avg_words_per_sentence = words / sentences
    syllables = sum([sum(1 for c in word if c.lower() in 'aeiou') for word in word_tokenize(text)])
    score = round(206.835 - 1.015 * avg_words_per_sentence - 84.6 * (syllables / words))
    return max(0, min(100, score))

def analyze_tone_vader(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def spelling_check(text):
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return 3 if corrected != text else 0
    except:
        return 2

def detect_hashtags_mentions(text):
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    return len(hashtags), len(mentions)

def detect_call_to_action(text):
    ctas = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view', 'discuss']
    return any(cta in text.lower() for cta in ctas)

def detect_emojis(text):
    return len(re.findall(r'[^\w\s,.!?]', text))

def detect_emotional_appeal(text):
    positive_words = ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success']
    return sum(1 for word in positive_words if word in text.lower())

def generate_hashtags(post, top_n=3):
    words = nltk.word_tokenize(post.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w.isalpha() and w not in stop_words]
    word_counts = Counter(filtered).most_common(top_n)
    keywords = ['#' + word for word, _ in word_counts]
    return list(dict.fromkeys(TRENDING_HASHTAGS + keywords))[:top_n]

def hook_strength(post):
    first_line = sent_tokenize(post)[0] if sent_tokenize(post) else ""
    score = 10 if 8 <= len(first_line.split()) <= 18 and any(p in first_line.lower() for p in ['?', 'ever', 'struggle', 'secret', 'reveal']) else 5
    return score

def calculate_score(post):
    readability = flesch_kincaid(post)
    tone = analyze_tone_vader(post)
    spelling_issues = spelling_check(post)
    word_count = len(word_tokenize(post))
    cta = detect_call_to_action(post)
    hashtags, mentions = detect_hashtags_mentions(post)
    emojis = detect_emojis(post)
    emotional_appeal = detect_emotional_appeal(post)
    hook = hook_strength(post)

    spelling_score = max(0, 10 - spelling_issues)
    length_score = 10 if 80 <= word_count <= 600 else max(0, 10 - abs(word_count - 340) / 60)

    weights = {
        "readability": 8,
        "tone": 10,
        "grammar": 7,
        "length": 3,
        "cta": 20,
        "hashtags": 15,
        "mentions": 5,
        "emotional": 30,
        "emojis": 15,
        "hook": 12
    }

    weighted_score = (
        (readability / 100 * weights["readability"]) +
        ((tone + 1) / 2 * 10 * weights["tone"] / 10) +
        (spelling_score * weights["grammar"] / 10) +
        (length_score * weights["length"] / 10) +
        (10 * weights["cta"] / 10 if cta else 0) +
        (min(3, hashtags) / 3 * 10 * weights["hashtags"] / 10) +
        (min(2, mentions) / 2 * 10 * weights["mentions"] / 10) +
        (emotional_appeal * weights["emotional"] / 10) +
        (min(5, emojis) / 5 * 10 * weights["emojis"] / 10) +
        (hook * weights["hook"] / 10)
    )

    return round(weighted_score), {
        "Readability": readability,
        "Tone & Sentiment": round((tone + 1) / 2 * 10),
        "Grammar & Style": spelling_score,
        "Length & Structure": length_score,
        "Call-to-Action": 10 if cta else 0,
        "Hashtags": min(10, hashtags * 3.3),
        "Mentions": min(5, mentions * 2.5),
        "Emotional Appeal": emotional_appeal,
        "Engagement Hooks": min(5, emojis * 1.0),
        "Hook Strength": hook
    }

# Virality estimate
def predict_virality(score):
    return (
        "🔥 Likely to go viral" if score >= 90 else
        "📈 High engagement potential" if score >= 75 else
        "👍 Moderate engagement" if score >= 60 else
        "📉 Low engagement unless boosted"
    )
