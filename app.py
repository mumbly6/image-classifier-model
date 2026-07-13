
import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Necessary for the pkl to load properly
def get_yolo_label(fname): return 'threat'

# Use cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    return load_learner('threat_classifier_v1.pkl')

# Initialize the model
try:
    learn = load_my_model()
    categories = learn.dls.vocab
except Exception as e:
    st.error(f'Error loading model: {e}')
    categories = ['safe', 'threat']

st.title('🛡️ X-Ray Contraband Detector')
st.write('Upload an X-ray image to identify potential threats using our trained Fast.ai model.')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_container_width=True)
    
    if st.button('Classify'):
        with st.spinner('Analyzing...'):
            # Convert PIL to Fastai image
            fast_img = PILImage.create(uploaded_file)
            pred, pred_idx, probs = learn.predict(fast_img)
            
            st.subheader(f'Prediction: {pred.upper()}')
            
            # Display probabilities as a bar chart
            prob_dict = {categories[i]: float(probs[i]) for i in range(len(categories))}
            st.bar_chart(prob_dict)
            
            # Specific confidence highlight
            conf = float(probs[pred_idx]) * 100
            st.info(f'Confidence Level: {conf:.2f}%')
