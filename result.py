import numpy as np 
import pickle 
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

def bg_img():
    
    st.markdown(
        f""" 
    <style>
    .stApp
    {{
        background-image: url('https://png.pngtree.com/background/20230416/original/pngtree-fruit-cartoon-background-picture-image_2443618.jpg');
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """,unsafe_allow_html=True)
    
img_height= 100
img_width = 100
batch_size = 32
dataset_url = "dataset/train"
train_ds = tf.keras.utils.image_dataset_from_directory( 
    dataset_url, 
    validation_split=0.2, 
    subset= 'training', 
    seed = 256, 
    image_size=(img_height,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height,img_width),
  batch_size=batch_size
)

# Print labels

class_names = train_ds.class_names

pickle_in=open('history.pkl','rb')
loaded_model=pickle.load(pickle_in)
print(class_names)
def fruit(data):
    img = tf.keras.utils.load_img(data, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # create a batch

    predictions_apple = loaded_model.predict(img_array)
    score_apple = tf.nn.softmax(predictions_apple[0])

    if(class_names[np.argmax(score_apple)][:6]=="rotten"):
        return ("This",class_names[np.argmax(score_apple)][6:]," is {:.2f}".format(100-(100 * np.max(score_apple))),"% healthy. ")
    else:
        return ("This",class_names[np.argmax(score_apple)][5:]," is {:.2f}".format(100 * np.max(score_apple)),"% healthy. This is rotten fruit") 
def main():
    bg_img()
    st.title("Health of fruits")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:   
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fruit Image")
        result = fruit(uploaded_file)
        
        st.header(f"{result[0]} {result[1]} {result[2]}% healthy")

if __name__== '__main__':
    main()
