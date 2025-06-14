"""LinkedIn Post Optimizer - Final Clean Version with punkt_tab fix and smarter optimization"""

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
import shutil

# 🔒 Remove bogus punkt_tab directory if it exists
for path in nltk.data.path:
    bogus_path = os.path.join(path, "tokenizers", "punkt_tab")
    if os.path.exists(bogus_path):
        shutil.rmtree(bogus_path)

# 🔇 Download valid NLTK resources quietly
required_nltk = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "vader_lexicon": "sentiment/vader_lexicon"
}
for pkg, resource_path in required_nltk.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(pkg, quiet=True)

analyzer = SentimentIntensityAnalyzer()

# --- NLP Functions ---
def flesch_kincaid(text):
    words = len(word_tokenize(text))
    sentences = len(sent_tokenize(text))
    if sentences == 0 or words == 0:
        return 0
    avg = words / sentences
    syllables = sum(sum(1 for c in word if c in 'aeiou') for word in word_tokenize(text.lower()))
    return round(max(0, min(100, 206.835 - 1.015 * avg - 84.6 * (syllables / words))))

def analyze_tone_vader(text):
    return analyzer.polarity_scores(text)['compound']

def spelling_check(text):
    try:
        return 3 if str(TextBlob(text).correct()) != text else 0
    except:
        return 2

def detect_hashtags_mentions(text):
    return len(re.findall(r'#\w+', text)), len(re.findall(r'@\w+', text))

def detect_call_to_action(text):
    ctas = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view', 'discuss']
    return any(cta in text.lower() for cta in ctas)

def detect_emojis(text):
    return len(re.findall(r'[^\w\s,.!?]', text))

def detect_emotional_appeal(text):
    emotion_words = ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success']
    return min(10, sum(1 for w in word_tokenize(text.lower()) if w in emotion_words))

def generate_hashtags(text, top_n=3):
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    freq = Counter(words)
    return ['#' + w for w, _ in freq.most_common(top_n)]

def is_similar_sentence(s1, s2, threshold=0.5):
    w1, w2 = set(word_tokenize(s1.lower())), set(word_tokenize(s2.lower()))
    return len(w1 & w2) / ((len(w1) + len(w2)) / 2) > threshold if w1 and w2 else False

def strip_cta(text):
    cta = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view', 'discuss']
    emotion = ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success']
    return ' '.join([s for s in sent_tokenize(text) if not (any(c in s.lower() for c in cta) and not any(e in s.lower() for e in emotion))])

def calculate_score(post):
    readability = flesch_kincaid(post)
    tone = analyze_tone_vader(post)
    spell = spelling_check(post)
    words = len(word_tokenize(post))
    cta = detect_call_to_action(post)
    hashtags, mentions = detect_hashtags_mentions(post)
    emojis = detect_emojis(post)
    emotion = detect_emotional_appeal(post)

    length_score = 10 if 80 <= words <= 600 else max(0, 10 - abs(words - 340) / 60)
    weights = {
        "readability": 8, "tone": 10, "grammar": 7, "length": 3,
        "cta": 20, "hashtags": 15, "mentions": 5, "emotional": 30, "emojis": 15
    }
    score = (
        (readability / 100 * weights["readability"]) +
        ((tone + 1) / 2 * 10 * weights["tone"] / 10) +
        (max(0, 10 - spell) * weights["grammar"] / 10) +
        (length_score * weights["length"] / 10) +
        (10 * weights["cta"] / 10 if cta else 0) +
        (min(3, hashtags) / 3 * 10 * weights["hashtags"] / 10) +
        (min(2, mentions) / 2 * 10 * weights["mentions"] / 10) +
        (emotion * weights["emotional"] / 10) +
        (min(5, emojis) / 5 * 10 * weights["emojis"] / 10)
    )
    return round(score), {
        "Readability": readability,
        "Tone & Sentiment": round((tone + 1) * 5),
        "Grammar & Style": max(0, 10 - spell),
        "Length & Structure": round(length_score, 1),
        "Call-to-Action": 10 if cta else 0,
        "Hashtags": min(10, hashtags * 3.3),
        "Mentions": min(5, mentions * 2.5),
        "Emotional Appeal": emotion,
        "Engagement Hooks": min(5, emojis)
    }

def predict_virality(score):
    if score >= 90: return "Likely to go viral"
    if score >= 75: return "High engagement potential"
    if score >= 60: return "Moderate engagement"
    return "Low engagement unless boosted"

def optimize_post(post, post_wo_cta):
    optimized = post_wo_cta
    sentences = sent_tokenize(optimized)
    additions = [
        "This tool leverages advanced NLP techniques to analyze content effectively.",
        "Did you know? Engaging posts can increase visibility by over 50% on LinkedIn!",
        "Adding a personal touch makes posts more relatable and boosts engagement."
    ]
    for a in additions:
        if not any(is_similar_sentence(a, s) for s in sentences):
            optimized += " " + a
            if additions.index(a) >= 1: break

    first_sentence = sent_tokenize(post.strip())[0].lower()
    emotional = ['excited', 'thrilled', 'proud', 'amazing', 'pumped', 'delighted', 'grateful']
    if not any(word in first_sentence for word in emotional):
        optimized = f"I'm excited to share that {optimized}"

    if detect_hashtags_mentions(optimized)[1] == 0:
        optimized += " Built with tools like @Streamlit @Python."

    ctas = [
        "What strategies have you used to improve engagement? Let me know in the comments!",
        "Let’s discuss in the comments below! What do you think?",
        "I’d love to hear your views—share them in the comments!"
    ]
    random.shuffle(ctas)
    optimized += " " + ctas[0]
    return optimized

# --- Streamlit App ---
st.set_page_config(page_title="LinkedIn Post Optimizer", layout="centered")
st.title("🚀 LinkedIn Post Optimizer")
st.markdown("Enhance your LinkedIn post using NLP-powered suggestions.")

post = st.text_area("✍️ Paste your LinkedIn post here:", height=250)

if st.button("🔍 Analyze and Optimize"):
    if not post.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = strip_cta(post)
        optimized = optimize_post(post, cleaned)
        score, breakdown = calculate_score(post)
        virality = predict_virality(score)
        hashtags = generate_hashtags(post)

        st.subheader("📊 Engagement Score")
        st.metric("Score", f"{score} / 100")
        st.caption(f"📈 {virality}")

        st.subheader("📍 Breakdown")
        for k, v in breakdown.items():
            st.markdown(f"- **{k}**: {v}/10")

        st.subheader("✅ Optimized Post")
        st.code(optimized, language="markdown")

        st.subheader("🏷️ Suggested Hashtags")
        st.markdown(" ".join(hashtags))

        st.success("Post optimized successfully!")
