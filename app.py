import streamlit as st
import tensorflow as tf
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# Load VGG16 model for feature extraction
base_model = VGG16()
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# File uploader
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
if uploaded_image is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)

 

    # Load caption data and mapping
    try:
        with open('./vars/captions.txt', 'r') as f:
            next(f)
            captions_doc = f.read()
        mapping = load("./vars/Mapping.joblib")
    except FileNotFoundError:
        st.error("Required files (captions.txt or Mapping.joblib) not found.")
        mapping = {}

    tokenizer = load("./vars/Tokenizer.joblib")
    max_len = max(len(caption.split()) for captions in mapping.values() for caption in captions)

    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(model, image_feature, tokenizer, max_length):
        in_text = 'startseq'
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([image_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idx_to_word(yhat, tokenizer)
            if word is None or word == 'endseq':
                break
            in_text += f" {word}"
        return in_text

    model3 = tf.keras.models.load_model("./model/best_model_70_epochs.keras")

    def generate_caption(model):
        
        
        
        y_pred = predict_caption(model, feature, tokenizer, 35)
        y_pred_refined = ' '.join(y_pred.split()[1:])  # Exclude the first and last tokens
        st.write("**Predicted Caption:**", y_pred_refined)
        

    # Replace with actual image name for testing
    generate_caption(model3)
