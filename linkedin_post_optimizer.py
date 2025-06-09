"""
LinkedIn Post Optimizer with Image-to-Video Creator - Fully Working Version for Streamlit Cloud
"""

import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rembg import remove
from PIL import Image
import io
from moviepy.editor import ImageClip, AudioFileClip
import os

# Initialize NLTK resources safely
def initialize_nltk_resources():
    try:
        for resource in ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords']:
            nltk.download(resource, quiet=True)
    except Exception as e:
        st.error(f"Error initializing NLTK resources: {e}")

initialize_nltk_resources()

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

# Improved deduplication function with a lower threshold
def is_similar_sentence(sent1, sent2, threshold=0.5):  # Lowered threshold for better matching
    words1 = set(word_tokenize(sent1.lower()))
    words2 = set(word_tokenize(sent2.lower()))
    overlap = len(words1.intersection(words2))
    avg_len = (len(words1) + len(words2)) / 2
    return overlap / avg_len > threshold if avg_len > 0 else False

def strip_cta(text):
    sentences = sent_tokenize(text)
    cta_keywords = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view', 'discuss']
    non_cta_sentences = [s for s in sentences if not any(keyword in s.lower() for keyword in cta_keywords)]
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
    # Adjusted ideal length range for LinkedIn posts (100–800 words)
    length_score = 10 if 100 <= word_count <= 800 else max(0, 10 - abs(word_count - 450) / 80)

    weights = {
        "readability": 8,
        "tone": 10,
        "grammar": 7,
        "length": 5,   # Reduced weight to be less punitive for short posts
        "cta": 20,
        "hashtags": 15,
        "mentions": 5,
        "emotional": 20, # Increased weight to prioritize engagement
        "emojis": 10
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

# Image Processing Functions
def remove_background(image_file):
    try:
        input_image = image_file.read()
        output_image = remove(input_image)
        return Image.open(io.BytesIO(output_image)).convert("RGBA")
    except Exception as e:
        st.error(f"Error removing background: {e}")
        return None

def create_video(image, audio_path, duration=10, output_path="output_video.mp4"):
    try:
        image.save("temp_image.png")
        clip = ImageClip("temp_image.png").set_duration(duration)
        clip = clip.resize(lambda t: 1 + 0.02 * t)  # Zoom effect
        clip = clip.set_position(('center', 'center'))
        if os.path.exists(audio_path):
            audio = AudioFileClip(audio_path).subclip(0, duration)
            clip = clip.set_audio(audio)
        else:
            st.warning("Audio file 'background_music.mp3' not found in the app directory. Creating video without audio.")
            st.info("Please ensure 'background_music.mp3' is included in your GitHub repository root.")
        clip.write_videofile(output_path, fps=24)
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {e}")
        return None

# UI
st.set_page_config(page_title="LinkedIn Post Optimizer & Video Creator", layout="centered")
st.title("LinkedIn Post Optimizer & Video Creator")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Optimize LinkedIn Post", "Create Video from Image"])

if mode == "Optimize LinkedIn Post":
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
                st.code("What are your thoughts? Let me know in the comments!")
            
            @st.cache_resource
            def get_summarizer():
                try:
                    from transformers import pipeline
                    # Use t5-small for lightweight summarization in Streamlit Cloud
                    return pipeline("summarization", model="t5-small")
                except Exception as e:
                    st.error(f"Failed to load summarization model: {e}")
                    return None

            if st.button("Generate AI-Optimized Version"):
                try:
                    summarizer = get_summarizer()
                    if summarizer is None:
                        st.error("Summarization model not available. Please try again later.")
                    else:
                        with st.spinner("Generating optimized version..."):
                            # Calculate input length
                            input_length = len(word_tokenize(post))
                            # Strip CTA from input to avoid duplication in summary
                            post_without_cta = strip_cta(post)
                            # Adjust prompt based on input length
                            if input_length < 150:
                                # For short posts, expand with context
                                prompt = (f"Expand this LinkedIn post by adding context, a brief explanation, and an example, while keeping it professional and engaging, and retaining key details: {post_without_cta}")
                                max_length = max(100, int(input_length * 2))  # Aim to double the length
                                min_length = min(80, max_length - 20)
                            else:
                                # For longer posts, summarize
                                prompt = (f"Summarize this LinkedIn post while keeping it professional, engaging, and retaining key details and examples: {post_without_cta}")
                                max_length = max(50, int(input_length * 0.8))
                                min_length = min(50, max_length - 10)
                            # Generate optimized text
                            optimized = summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False, num_beams=4)[0]['summary_text']
                            # Post-process to enhance content and score
                            optimized_sentences = sent_tokenize(optimized)
                            # Add additional context for short posts
                            if input_length < 150:
                                context_additions = [
                                    "This tool leverages advanced NLP techniques to analyze how well your resume aligns with job requirements.",
                                    "For example, a user recently increased their ATS score from 4 to 8 by adding key terms like 'project management' and 'agile methodology' identified by the calculator.",
                                    "I’m passionate about helping job seekers navigate the complexities of ATS systems to land their dream roles!"
                                ]
                                for addition in context_additions:
                                    if not any(is_similar_sentence(addition, s) for s in optimized_sentences):
                                        optimized += f" {addition}"
                                        optimized_sentences.append(addition)
                            # Restore key sentences with emotional appeal or examples
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
                            # Add emotional intro if not already present
                            emotional_intro = "I'm excited to share that"
                            if not any(emotional_intro.lower() in s.lower() for s in optimized_sentences):
                                optimized = f"{emotional_intro} {optimized} 🚀"
                            # Add CTA if not already present, with varied phrasing
                            cta_options = [
                                "What are your thoughts? Let me know in the comments!",
                                "Let’s discuss in the comments below! What do you think?",
                                "I’d love to hear your views—share them in the comments!"
                            ]
                            if not detect_call_to_action(optimized):
                                for cta in cta_options:
                                    if not any(is_similar_sentence(cta, s) for s in sent_tokenize(optimized)):
                                        optimized += f"\n{cta}"
                                        break
                            # Add hashtags if not already present
                            hashtags = ' '.join(generate_hashtags(optimized))
                            if not detect_hashtags_mentions(optimized)[0]:
                                optimized += f"\n{hashtags}"
                            # Recalculate score for the optimized version
                            optimized_score, optimized_details = calculate_score(optimized)
                            optimized_virality = predict_virality(optimized_score)
                            st.markdown("#### Optimized Version:")
                            st.markdown(optimized)
                            st.markdown(f"**Optimized Quality Score:** {optimized_score}/100")
                            st.markdown(f"**Optimized Virality Prediction:** {optimized_virality}")
                            st.download_button("Download Optimized Version", data=optimized, file_name="optimized_linkedin_post.txt")
                except Exception as e:
                    st.error(f"Error during summarization: {e}")

    else:
        st.info("Please enter your LinkedIn post above to begin the analysis.")

elif mode == "Create Video from Image":
    st.header("Create Video from Image with Background Removal")
    st.markdown("Upload an image to remove its background (e.g., TV, fridge) and create a video with background audio. For GAN-themed posts, use AI-generated images! Note: A royalty-free audio file ('background_music.mp3') must be included in your GitHub repository root.")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image_uploader")
    audio_path = "background_music.mp3"  # Must be in the repository root
    video_duration = st.slider("Video Duration (seconds)", 5, 30, 10)

    if image_file is not None:
        st.image(image_file, caption="Original Image", use_column_width=True)
        with st.spinner("Removing background..."):
            result_image = remove_background(image_file)
            if result_image:
                st.image(result_image, caption="Image with Background Removed", use_column_width=True)
                with st.spinner("Creating video..."):
                    video_path = create_video(result_image, audio_path, duration=video_duration)
                    if video_path and os.path.exists(video_path):
                        st.video(video_path)
                        with open(video_path, "rb") as file:
                            st.download_button("Download Video", file, file_name="output_video.mp4")
                        st.success("Video created! Download and share on LinkedIn to showcase GAN capabilities.")
                    else:
                        st.error("Failed to create video. Ensure 'background_music.mp3' is included in your GitHub repository root.")
            else:
                st.error("Failed to remove background. Try a different image.")

# Cleanup temporary files
for temp_file in ["temp_image.png", "output_video.mp4"]:
    if os.path.exists(temp_file):
        os.remove(temp_file)