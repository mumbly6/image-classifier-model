
import streamlit as st
import platform
import pathlib

# ── Fix: Patch pathlib for cross-platform .pkl compatibility ──────────────
# fastai models trained on Linux embed PosixPath objects. When loaded on
# Windows (or with newer Python 3.12+ where pathlib internals changed),
# pickle fails with "'Resolver' object has no attribute 'dict'".
# This patch ensures PosixPath resolves to the current OS path type.
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
from PIL import Image

# ── Fix: Custom label function MUST be in scope before load_learner ──────
# fastai serialises the label function reference; it must be importable
# at load time with the exact same name used during training.
def get_yolo_label(fname):
    """Label function used during training — required for unpickling."""
    return 'threat'

# Use cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    return load_learner('threat_classifier_v1.pkl')

# Initialize the model
model_loaded = False
try:
    learn = load_my_model()
    categories = learn.dls.vocab
    model_loaded = True
except Exception as e:
    st.error(f'Error loading model: {e}')
    categories = ['safe', 'threat']

st.title('🛡️ X-Ray Contraband Detector')
st.write('Upload an X-ray image to identify potential threats using our trained Fast.ai model.')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_container_width=True)

    if not model_loaded:
        st.warning('Model failed to load — classification is unavailable.')
    elif st.button('Classify'):
        with st.spinner('Analyzing...'):
            # Reset stream position after PIL read
            uploaded_file.seek(0)
            fast_img = PILImage.create(uploaded_file)
            pred, pred_idx, probs = learn.predict(fast_img)

            st.subheader(f'Prediction: {pred.upper()}')

            # Display probabilities as a bar chart
            prob_dict = {categories[i]: float(probs[i]) for i in range(len(categories))}
            st.bar_chart(prob_dict)

            # Specific confidence highlight
            conf = float(probs[pred_idx]) * 100
            st.info(f'Confidence Level: {conf:.2f}%')
