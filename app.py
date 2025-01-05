import streamlit as st
import pickle 
import tensorflow as tf
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np

features={}
directory='./Images'

#load features from pickle
try:
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
        print("Features loaded successfully.")
except EOFError:
    print("The file is empty or corrupted. Please regenerate the pickle file.")



#load captions data
with open('captions.txt','r') as f:
    next(f)
    captions_doc=f.read()

mapping=load("Mapping.joblib")


all_captions=[]
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


tokenizer=load("Tokenizer.joblib")


max_len=max(len(caption.split()) for caption in all_captions)

def idx_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def predict_caption(model,image,tokenizer,max_length):
    #add start tag for generation
    in_text='startseq'
    #iterate over maxlength of sequence
    for i in range(max_length):
        #incode input sequence
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        #pad the sequence
        sequence=pad_sequences([sequence],max_length)
        #predict next word
        yhat=model.predict([image,sequence],verbose=0)
        #convert index with high probability
        yhat=np.argmax(yhat)
        #convert index to word
        word=idx_to_word(yhat,tokenizer)
        #stop if word not found
        if word is None:
            break
        #append word as input for generation next word
        in_text +=" "+word
        #stop if we reach end tag
        if word =='endseq':
            break
    return in_text

model3=tf.keras.models.load_model("./model/best_model_70_epochs.keras")
from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name,model):
    #load Image
    #image_name="1002674143_1b742ab4b8.jpg"
    image_id=image_name.split('.')[0]
    image_path=os.path.join("Images",image_name)
    image=Image.open(image_path)
    
    captions=mapping[image_id]
    print("---------------------------Original Caption--------------------------")
    for caption in captions:
        print(caption)
    
    #predict the caption
    y_pred=predict_caption(model,features[image_id],tokenizer,max_len)
    
    print("--------------------------Predicted Caption----------------")
    st.write(y_pred)
    st.image(image)

generate_caption("3747543364_bf5b548527.jpg",model3)