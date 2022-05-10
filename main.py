import io
import streamlit as st
from PIL import Image
import requests
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import tqdm
from pexels_api import API

DEFAULT_IMG1_URL = (
    "http://example.com"
)


def get_sim_image(query):
    RESULTS_PER_PAGE = 10

    PEXELS_API_KEY = "563492ad6f917000010000019edfdc2e715c4c618681cf160b065bce "
    api = API(PEXELS_API_KEY)
    photos_dict = {}

    api.search(query, page=1, results_per_page=RESULTS_PER_PAGE)
    photos = api.get_entries()
    for photo in tqdm.tqdm(photos):
        photos_dict[photo.id] = vars(photo)['_Photo__photo']
        return (photos_dict[photo.id]['src']['original'])

print(get_sim_image('cat'))

def load_model():
    model = EfficientNetB0(weights='imagenet')
    return model

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


def fetch_img_from_url(url: str) -> Image:
    img = Image.open(requests.get(url, stream=True).raw)
    return img

def load_image(form):
    uploaded_file = form.file_uploader(label='Choose an image for recognition')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        return Image.open(io.BytesIO(image_data))
    else:
        return None

st.title('Image recognition')
model = load_model()
form = st.form(key="Image comparison")
img_load = load_image(form)
img_url = form.text_input("Or provide a link to an image", value=DEFAULT_IMG1_URL)
img = None
images = []
if img_load:
    img = img_load
    images.append(img)
elif img_url != DEFAULT_IMG1_URL and img_url:
    img = fetch_img_from_url(img_url)
    images.append(img)

submit = form.form_submit_button('Recognize Image')



if submit:
    if img:
        x = preprocess_image(img)
        preds = model.predict(x)
        classes = decode_predictions(preds, top=1)[0]
        for c1 in classes:
            st.markdown(f'A ***{c1[1]}*** is recognized in the image with a probability of ***{c1[2]}%***')
        img1 = fetch_img_from_url(get_sim_image(c1[1]))

        cols = st.columns(3)

        cols[0].image(img,  use_column_width=True,width=70, caption='Origin Image')
        cols[1].text("")
        cols[2].image(img1,  use_column_width=True,width=70, caption='Similar Image')

    else:
        st.write('**Image not selected!**')
