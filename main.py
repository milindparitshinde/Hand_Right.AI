import cv2
import pytesseract
from nltk.corpus import words
import nltk
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
from transformers import pipeline

# Download the NLTK words corpus
nltk.download('words')
# nltk.data.path.append('/Users/milind/Documents/ML_Project/Project/Hand_Right/env/')


# Set the path to Tesseract executable (update this path based on your system)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
#'/usr/local/bin/tesseract'

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# OCR Extraction function
def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image)
    return text

# Confidence score calculation function
def calculate_legibility_score(extracted_text):
    word_list = words.words()
    recognized_words = extracted_text.split()
    matches = sum(1 for word in recognized_words if word.lower() in word_list)
    match_rate = matches / len(recognized_words) if recognized_words else 0
    return match_rate * 100  # Confidence score as a percentage

# Use GPT model for feedback generation
# feedback_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# def generate_feedback(text):
#     prompt = f"Provide constructive handwriting improvement feedback based on this text:\n\n{text}"
#     response = feedback_pipeline(prompt, max_length=150)
#     return response[0]['generated_text']

st.title("Handwriting Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a handwriting image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract text and calculate confidence score
    text = extract_text_from_image(temp_file_path)
    confidence_score = calculate_legibility_score(text)

    # feedback = generate_feedback(text)

    st.write("### Extracted Text:")
    st.write(text)
    # st.write("### Feedback:")
    # st.write(feedback)
    st.write(f"### Confidence Score: {confidence_score:.2f}%")
