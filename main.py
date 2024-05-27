import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64

def set_background(image_file, font_color='black'):
    """
    Sets the background image and font color for the Streamlit app.

    Parameters:
        image_file (str): The path to the image file to be used as the background.
        font_color (str): The color to be used for the font. Default is 'black'.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp li, .stApp a {{
            color: {font_color} !important;
        }}
        .segment-box {{
            border: 2px solid black;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 20px;
            background-color: white;
            color: black;
            margin-top: 20px;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

set_background('./backg.jpg', font_color="black")

# Set title
st.title('Rice Leaf Disease Classification')

# Set header
st.header('Please upload a Rice Leaf image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./my_model.h5')

# Load class names
with open('./labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    class_name, conf_score = classify(image, model, class_names)
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(round(conf_score * 100, 1)))

# Create a fill-up box with bold text "Segment"
st.markdown('<div class="segment-box">Disease Affected Region</div>', unsafe_allow_html=True)
