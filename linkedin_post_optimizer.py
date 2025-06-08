"""
💼 LinkedIn Post Optimizer with Image-to-Video Creator – Fully Working Version for Streamlit Cloud
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

# 🔽 Download required NLTK resources at startup
nltk.download('punkt')           # Sentence tokenization
nltk.download('punkt_tab')       # Fixes 'punkt_tab' error
nltk.download('vader_lexicon')   # For sentiment analysis
nltk.download('stopwords')       # For keyword extraction

# 🧠 Initialize tools
analyzer = SentimentIntensityAnalyzer()

# 📊 Helper Functions for LinkedIn Post Optimization
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
    ctas = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view']
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

# 📷 Image Processing Functions
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
            st.warning("Audio file 'background_music.mp3' not found. Creating video without audio.")
        clip.write_videofile(output_path, fps=24)
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {e}")
        return None

# 🖥️ UI
st.set_page_config(page_title="💼 LinkedIn Post Optimizer & Video Creator", layout="centered")
st.title("💼 LinkedIn Post Optimizer & Video Creator")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Optimize LinkedIn Post", "Create Video from Image"])

if mode == "Optimize LinkedIn Post":
    post = st.text_area(
        "📝 Paste or write your LinkedIn post below...",
        height=300,
       ව

System: Thank you for sharing the error details. The `ModuleNotFoundError: No module named 'onnxruntime'` occurs because the `onnxruntime` package, a dependency required by `rembg` for background removal, is not included in your `requirements.txt`. This prevents the `rembg` library from functioning in your Streamlit app (`linkedin_post_optimizer.py`) when deployed on Streamlit Cloud.

Additionally, your goal is to:
1. Optimize LinkedIn posts using the existing functionality in `linkedin_post_optimizer.py`.
2. Remove the background from an image (e.g., objects like a TV or fridge) and create a video with background audio, aligned with a GAN-themed LinkedIn post.

I’ll provide updated versions of `linkedin_post_optimizer.py` and `requirements.txt` to fix the error by adding `onnxruntime` and ensure compatibility with Streamlit Cloud. The updated code retains the NLTK `punkt_tab` fix from your previous issue and includes robust error handling for the image-to-video feature. I’ll also ensure the app works seamlessly for both LinkedIn post optimization and video creation.

---

### Updated Files

#### 1. `linkedin_post_optimizer.py`
**Changes**:
- Retained the NLTK `punkt_tab` fix with `nltk.download('punkt_tab')`.
- Kept the dual-mode functionality (LinkedIn post optimization and image-to-video creation).
- Enhanced error handling for the audio file (`background_music.mp3`) to provide clearer user guidance.
- Ensured compatibility with Streamlit Cloud using relative file paths.
- Added a note about including `background_music.mp3` in the repository for video creation.

```python
"""
💼 LinkedIn Post Optimizer with Image-to-Video Creator – Fully Working Version for Streamlit Cloud
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

# 🔽 Download required NLTK resources at startup
nltk.download('punkt')           # Sentence tokenization
nltk.download('punkt_tab')       # Fixes 'punkt_tab' error
nltk.download('vader_lexicon')   # For sentiment analysis
nltk.download('stopwords')       # For keyword extraction

# 🧠 Initialize tools
analyzer = SentimentIntensityAnalyzer()

# 📊 Helper Functions for LinkedIn Post Optimization
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
    ctas = ['comment', 'let me know', 'thoughts?', 'what do you think', 'share your view']
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

# 📷 Image Processing Functions
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

# 🖥️ UI
st.set_page_config(page_title="💼 LinkedIn Post Optimizer & Video Creator", layout="centered")
st.title("💼 LinkedIn Post Optimizer & Video Creator")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Optimize LinkedIn Post", "Create Video from Image"])

if mode == "Optimize LinkedIn Post":
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

elif mode == "Create Video from Image":
    st.header("🎥 Create Video from Image with Background Removal")
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