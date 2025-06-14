"""LinkedIn Post Optimizer - Final Version with Safe NLTK Initialization and Streamlit UI"""

import streamlit as st # Import streamlit first

# ---- Streamlit UI Config ----
# This MUST be the first Streamlit command executed.
st.set_page_config(page_title="LinkedIn Post Optimizer", layout="centered")

# Set environment variable to avoid file watch issues
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Now import other libraries
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

# Safe NLTK resource initialization
# Added 'punkt_tab' to ensure it's downloaded if needed.
nltk_packages = ["punkt", "vader_lexicon", "stopwords", "punkt_tab"]
for pkg in nltk_packages:
    try:
        # Adjusted to correctly check for 'punkt_tab' as a tokenizer resource.
        if pkg in ["punkt", "punkt_tab"]:
            nltk.data.find(f'tokenizers/{pkg}')
        else:
            nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        # These st. calls are now after st.set_page_config
        st.info(f"Downloading NLTK package: {pkg}. This may take a moment...")
        nltk.download(pkg)
        st.success(f"NLTK package '{pkg}' downloaded successfully.")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Helper functions
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
        if corrected != text:
            return 3
        return 0
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
    words = word_tokenize(text.lower())
    count = sum(1 for word in words if word in positive_words)
    return min(10, count)

def generate_hashtags(post, top_n=3):
    words = nltk.word_tokenize(post.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(top_n)
    return ['#' + word for word, _ in common_words]

def is_similar_sentence(sent1, sent2, threshold=0.5):
    words1 = set(word_tokenize(sent1.lower()))
    words2 = set(word_tokenize(sent2.lower()))
    overlap = len(words1.intersection(words2))
    avg_len = (len(words1) + len(words2)) / 2
    return overlap / avg_len > threshold if avg_len > 0 else False

def strip_cta(text):
    sentences = sent_tokenize(text)
    cta_keywords = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view', 'discuss']
    emotional_words = ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success']
    non_cta_sentences = [
        s for s in sentences
        if not (any(keyword in s.lower() for keyword in cta_keywords) and not any(word in s.lower() for word in emotional_words))
    ]
    return ' '.join(non_cta_sentences)

def calculate_score(post):
    readability = flesch_kincaid(post)
    tone = analyze_tone_vader(post)
    spelling_issues = spelling_check(post)
    word_count = len(word_tokenize(post))
    cta = detect_call_to_action(post)
    hashtags, mentions = detect_hashtags_mentions(post)
    emojis = detect_emojis(post)
    emotional_appeal = detect_emotional_appeal(post)

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
        "emojis": 15
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
        (min(5, emojis) / 5 * 10 * weights["emojis"] / 10)
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
        "Engagement Hooks": min(5, emojis * 1.0)
    }

def predict_virality(score):
    if score >= 90:
        return "Likely to go viral"
    elif score >= 75:
        return "High engagement potential"
    elif score >= 60:
        return "Moderate engagement"
    else:
        return "Low engagement unless boosted"

def optimize_post(post, post_without_cta):
    emotional_intro = "I'm excited to share that"
    if emotional_intro.lower() in post.lower():
        return post

    optimized = post_without_cta
    optimized_sentences = sent_tokenize(optimized)

    context_additions = [
        "This tool leverages advanced NLP techniques to analyze content effectively.",
        "Did you know? Engaging posts can increase visibility by over 50% on LinkedIn!",
        "For example, adding a personal touch can make your post more relatable and boost engagement.",
        "I’m absolutely thrilled to share insights that help professionals grow their presence!",
        "As someone passionate about career growth, I’ve seen how optimized posts can make a difference."
    ]
    added_context = 0
    for addition in context_additions:
        if added_context >= 2:
            break
        if not any(is_similar_sentence(addition, s) for s in optimized_sentences):
            optimized += f" {addition}"
            optimized_sentences.append(addition)
            added_context += 1

    if not any(emotional_intro.lower() in s.lower() for s in optimized_sentences):
        optimized = f"{emotional_intro} {optimized}"

    mentions = "@Streamlit @Python"
    if not detect_hashtags_mentions(optimized)[1]:
        optimized += f" Built with tools like {mentions}."

    cta_options = [
        "What strategies have you used to improve engagement? Let me know in the comments!",
        "Let’s discuss in the comments below! What do you think?",
        "I’d love to hear your views—share them in the comments!",
        "How do you approach LinkedIn engagement? Share your tips below!",
        "What’s your take on this? Drop a comment to let me know!"
    ]
    random.shuffle(cta_options)
    optimized += f" {cta_options[0]}"

    return optimized

# ---- Streamlit UI Elements ----
st.title("🚀 LinkedIn Post Optimizer")
st.markdown("Enhance your LinkedIn post for maximum engagement using NLP techniques.")

post = st.text_area("✍️ Paste your LinkedIn post here:", height=250)

if st.button("🔍 Analyze and Optimize"):
    if not post.strip():
        st.warning("Please enter some text to analyze.")
    else:
        original_post = post
        cleaned_post = strip_cta(post)
        optimized = optimize_post(original_post, cleaned_post)
        score, breakdown = calculate_score(original_post)
        virality = predict_virality(score)
        hashtags = generate_hashtags(original_post)

        st.subheader("📊 Score Summary")
        st.metric("Engagement Score", f"{score} / 100")
        st.write("**Virality Prediction:**", virality)

        st.subheader("🔍 Breakdown")
        for k, v in breakdown.items():
            st.write(f"- **{k}:** {v}/10")

        st.subheader("🎯 Optimized Post")
        st.code(optimized, language="markdown")

        st.subheader("🏷️ Suggested Hashtags")
        st.write(" ".join(hashtags))

        st.success("Optimization complete!")