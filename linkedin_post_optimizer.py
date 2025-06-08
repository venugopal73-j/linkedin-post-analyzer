"""
💼 LinkedIn Post Optimizer – Fully Working Version for Streamlit Cloud
"""

import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

# 🔽 Download required NLTK resources at startup
nltk.download('punkt')           # Fixes 'punkt' or 'punkt_tab' error
nltk.download('vader_lexicon')   # For sentiment analysis

# 🧠 Initialize tools
analyzer = SentimentIntensityAnalyzer()

# 📊 Helper Functions
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
    ctas = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view']
    return any(cta in text.lower() for cta in ctas)

def detect_emojis(text):
    return len(re.findall(r'[^\w\s,.!?]', text))

def detect_emotional_appeal(text):
    positive_words = ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success']
    count = sum(1 for word in positive_words if word in text.lower())
    return min(10, count)

def generate_hashtags(post):
    r = Rake()
    r.extract_keywords_from_text(post)
    phrases = r.get_ranked_phrases()
    return ['#' + p.replace(' ', '') for p in phrases[:3]]

def calculate_score(post):
    readability = flesch_kincaid(post)
    tone = analyze_tone_vader(post)
    spelling_issues = spelling_check(post)
    word_count = len(word_tokenize(post))
    cta = detect_call_to_action(post)
    hashtags, mentions = detect_hashtags_mentions(post)
    emojis = detect_emojis(post)
    emotional_appeal = detect_emotional_appeal(post)

    # Normalize scores
    spelling_score = max(0, 10 - spelling_issues)
    length_score = 10 if 300 <= word_count <= 1000 else max(0, 10 - abs(word_count - 650) / 100)

    weights = {
        "readability": 10,
        "tone": 10,
        "grammar": 10,
        "length": 10,
        "cta": 10,
        "hashtags": 10,
        "mentions": 5,
        "emotional": 10,
        "emojis": 5
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
        return "🔥 Likely to go viral"
    elif score >= 75:
        return "📈 High engagement potential"
    elif score >= 60:
        return "👍 Moderate engagement"
    else:
        return "📉 Low engagement unless boosted"

# 🖥️ UI
st.set_page_config(page_title="💼 LinkedIn Post Optimizer", layout="centered")
st.title("💼 LinkedIn Post Optimizer")

post = st.text_area(
    "📝 Paste or write your LinkedIn post below...",
    height=300,
    placeholder="Type or paste your LinkedIn post here..."
)

if post.strip():
    total_score, details = calculate_score(post)
    virality = predict_virality(total_score)

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Metrics", "✨ Suggestions"])

    with tab1:
        st.subheader("📈 Summary")
        if total_score >= 90:
            st.success(f"Final Quality Score: {total_score}/100")
        elif total_score >= 75:
            st.info(f"Final Quality Score: {total_score}/100")
        else:
            st.warning(f"Final Quality Score: {total_score}/100")

        st.markdown(f"### 🔮 Virality Prediction: {virality}")

    with tab2:
        st.subheader("🔍 Parameter Breakdown")
        for key, value in details.items():
            st.progress(int(value), text=f"{key}: {value}/10")

    with tab3:
        st.subheader("💡 Optimization Suggestions")

        suggested_hashtags = generate_hashtags(post)
        st.markdown("#### 🏷️ Suggested Hashtags:")
        st.code(' '.join(suggested_hashtags))

        if not detect_call_to_action(post):
            st.markdown("#### 💬 Add a Call-to-Action:")
            st.code("What are your thoughts on this? Let me know in the comments!")

        @st.cache_resource
        def get_summarizer():
            from transformers import pipeline
            return pipeline("summarization", model="facebook/bart-large-cnn")

        if st.button("🧠 Generate AI-Optimized Version"):
            try:
                summarizer = get_summarizer()
                optimized = summarizer(post, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                st.markdown("#### ✨ Optimized Version:")
                st.markdown(optimized)
                st.download_button("📥 Download Optimized Version", data=optimized, file_name="optimized_linkedin_post.txt")
            except Exception as e:
                st.error(f"⚠️ Error generating rewrite: {e}")

else:
    st.info("Please enter your LinkedIn post above to begin the analysis.")