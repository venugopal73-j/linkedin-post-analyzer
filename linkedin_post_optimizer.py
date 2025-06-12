"""
LinkedIn Post Optimizer - Streamlit Cloud Version
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
import random  # For randomizing CTA selection

# Load environment variables (simplified, since Streamlit Cloud uses UI-set variables)
# Removed tomli dependency since it's not needed for Streamlit Cloud
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Initialize NLTK resources safely (delayed until needed)
def initialize_nltk_resources():
    try:
        for resource in ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords']:
            nltk.download(resource, quiet=True)
        return True
    except Exception as e:
        return f"Error initializing NLTK resources: {e}"

# Initialize tools
analyzer = SentimentIntensityAnalyzer()

# Helper Functions for LinkedIn Post Optimization
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
            return 3  # Some issues found
        return 0
    except:
        return 2  # Unknown error

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
    count = sum(1 for word in positive_words if word in text.lower())
    return min(10, count)

def generate_hashtags(post, top_n=3):
    words = nltk.word_tokenize(post.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(top_n)
    return ['#' + word for word, _ in common_words]

# Improved deduplication function
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
    # Keep sentences with emotional words, even if they contain CTA keywords
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
        return "🔥 Likely to go viral"
    elif score >= 75:
        return "📈 High engagement potential"
    elif score >= 60:
        return "👍 Moderate engagement"
    else:
        return "📉 Low engagement unless boosted"

# Optimization function
def optimize_post(post, post_without_cta):
    # Check if the post is already optimized (to prevent compounding content)
    emotional_intro = "I'm excited to share that"
    if emotional_intro.lower() in post.lower():
        return post  # Skip optimization if already optimized

    optimized = post_without_cta
    optimized_sentences = sent_tokenize(optimized)
    
    # Add context (limit to 2 sentences to avoid excessive length)
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

    # Add emotional intro
    if not any(emotional_intro.lower() in s.lower() for s in optimized_sentences):
        optimized = f"{emotional_intro} {optimized} 🌟🎉🚀"

    # Add CTA (randomized)
    cta_options = [
        "What strategies have you used to improve engagement? Let me know in the comments! 💬",
        "Let’s discuss in the comments below! What do you think? 🤔",
        "I’d love to hear your views—share them in the comments! 👇",
        "How do you approach LinkedIn engagement? Share your tips below! 💡",
        "What’s your take on this? Drop a comment to let me know! 📝"
    ]
    random.shuffle(cta_options)  # Randomize the list
    if not detect_call_to_action(optimized):
        for cta in cta_options:
            if not any(is_similar_sentence(cta, s) for s in sent_tokenize(optimized)):
                optimized += f"\n{cta}"
                break

    # Add hashtags
    hashtags = ' '.join(generate_hashtags(optimized))
    if not detect_hashtags_mentions(optimized)[0]:
        optimized += f"\n{hashtags}"

    # Trim to optimal length (target around 340 words for max length score)
    word_count = len(word_tokenize(optimized))
    if word_count > 400:  # Slightly above optimal to allow some buffer
        sentences = sent_tokenize(optimized)
        trimmed_sentences = []
        current_word_count = 0
        for s in sentences:
            sentence_words = len(word_tokenize(s))
            if current_word_count + sentence_words <= 340:
                trimmed_sentences.append(s)
                current_word_count += sentence_words
            else:
                break
        optimized = ' '.join(trimmed_sentences)
        # Re-add CTA and hashtags if they were trimmed
        if not detect_call_to_action(optimized):
            for cta in cta_options:
                if not any(is_similar_sentence(cta, s) for s in sent_tokenize(optimized)):
                    optimized += f"\n{cta}"
                    break
        if not detect_hashtags_mentions(optimized)[0]:
            optimized += f"\n{hashtags}"

    return optimized

# UI
st.set_page_config(page_title="LinkedIn Post Optimizer", layout="centered")
st.title("LinkedIn Post Optimizer")

# Initialize session state
if 'nltk_initialized' not in st.session_state:
    st.session_state.nltk_initialized = False
    st.session_state.nltk_error = None

# Delay NLTK initialization until the user interacts with the app
if not st.session_state.nltk_initialized:
    result = initialize_nltk_resources()
    if result is True:
        st.session_state.nltk_initialized = True
    else:
        st.session_state.nltk_error = result

# Check for NLTK initialization errors
if st.session_state.nltk_error:
    st.error(st.session_state.nltk_error)
    st.stop()

post = st.text_area(
    "Paste or write your LinkedIn post below...",
    height=300,
    placeholder="Type or paste your LinkedIn post here..."
)

if post.strip():
    total_score, details = calculate_score(post)
    virality = predict_virality(total_score)

    tab1, tab2, tab3 = st.tabs(["Overview", "Metrics", "Suggestions"])

    with tab1:
        st.subheader("Summary")
        if total_score >= 90:
            st.success(f"Original Quality Score: {total_score}/100")
        elif total_score >= 75:
            st.info(f"Original Quality Score: {total_score}/100")
        else:
            st.warning(f"Original Quality Score: {total_score}/100")
        st.markdown(f"### Virality Prediction: {virality}")

    with tab2:
        st.subheader("Parameter Breakdown")
        for key, value in details.items():
            st.progress(int(value), text=f"{key}: {value}/10")

    with tab3:
        st.subheader("Optimization Suggestions")
        suggested_hashtags = generate_hashtags(post)
        st.markdown("#### Suggested Hashtags:")
        st.code(' '.join(suggested_hashtags))
        if not detect_call_to_action(post):
            st.markdown("#### Add a Call-to-Action:")
            cta_suggestions = [
                "What are your thoughts? Let me know in the comments!",
                "Let’s discuss in the comments below! What do you think? 🤔",
                "I’d love to hear your views—share them in the comments! 👇",
                "How do you approach this? Share your tips below! 💡",
                "What’s your take on this? Drop a comment to let me know! 📝"
            ]
            selected_cta = random.choice(cta_suggestions)  # Randomly select a CTA
            st.code(selected_cta)
        
        if st.button("Generate AI-Optimized Version"):
            try:
                with st.spinner("Generating optimized version..."):
                    # Strip CTA from input to avoid duplication
                    post_without_cta = strip_cta(post)
                    # Optimize the post
                    optimized = optimize_post(post, post_without_cta)

                    # Post-process to enhance content (restoring emotional/example sentences)
                    optimized_sentences = sent_tokenize(optimized)
                    original_sentences = sent_tokenize(post)
                    emotional_sentences = [s for s in original_sentences if any(word in s.lower() for word in ['inspiring', 'amazing', 'excited', 'thrilled', 'proud', 'success'])]
                    example_sentences = [s for s in original_sentences if 'for example' in s.lower() or 'e.g.' in s.lower()]
                    if emotional_sentences:
                        emotional_sentence = emotional_sentences[0]
                        if not any(is_similar_sentence(emotional_sentence, s) for s in optimized_sentences):
                            optimized += f" {emotional_sentence}"
                            optimized_sentences.append(emotional_sentence)
                    if example_sentences:
                        example_sentence = example_sentences[0]
                        if not any(is_similar_sentence(example_sentence, s) for s in optimized_sentences):
                            optimized += f" {example_sentence}"
                            optimized_sentences.append(example_sentence)

                # Recalculate score for the optimized version
                optimized_score, optimized_details = calculate_score(optimized)
                optimized_virality = predict_virality(optimized_score)
                st.markdown("#### Optimized Version:")
                st.markdown(optimized)
                st.markdown(f"**Optimized Quality Score:** {optimized_score}/100")
                st.markdown(f"**Optimized Virality Prediction:** {optimized_virality}")
                # Show parameter breakdown for debugging
                st.markdown("#### Optimized Parameter Breakdown:")
                for key, value in optimized_details.items():
                    st.progress(int(value), text=f"{key}: {value}/10")
                st.download_button("Download Optimized Version", data=optimized, file_name="optimized_linkedin_post.txt")

            except Exception as e:
                st.error(f"Unexpected error during optimization: {e}")

else:
    st.info("Please enter your LinkedIn post above to begin the analysis.")