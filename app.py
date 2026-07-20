import streamlit as st
import platform
import pathlib
import pickle
import io
import sys

# ══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSE:  The .pkl was created with plum-dispatch v1.x which serialised
#              plum._resolver.Resolver objects that had a `.dict` attribute.
#              plum-dispatch v2.x removed/restructured Resolver — __dict__ is
#              gone — so pickle.BUILD crashes with:
#              "'Resolver' object has no attribute 'dict'"
#
#              SECONDARY CAUSE: 'fasttransform' (a temporary fastai sub-package
#              used at training time) must be importable for the Pipeline and
#              Transform classes stored in the pkl.
#
#              TERTIARY CAUSE: PosixPath objects in the pkl cannot be
#              instantiated on Windows without a path-type remap.
#
# FIXES APPLIED (layered, defence-in-depth):
#   1. requirements.txt pins plum-dispatch==1.7.4 and adds fasttransform
#   2. runtime.txt pins Python 3.10 on Streamlit Cloud
#   3. Custom Unpickler below intercepts plum._resolver.Resolver and injects a
#      v1-compatible stub in case the environment is still on plum v2.
#   4. pathlib.PosixPath is remapped to the current OS path type.
# ══════════════════════════════════════════════════════════════════════════════

# ── Fix 4: cross-platform PosixPath ──────────────────────────────────────────
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ── Fix 3: plum v1-compatible Resolver stub ───────────────────────────────────
# Plum v1 Resolver had instance attributes: dict, methods, _resolved,
# warn_redefinition, function_name, is_faithful.
# We create a stub with __dict__ enabled (no __slots__) so pickle BUILD works.
class _Resolverv1Stub:
    """Drop-in stub for plum v1 _resolver.Resolver so pickle can reconstruct it."""
    def __init__(self):
        self.dict = {}
        self.methods = []
        self._resolved = []
        self.warn_redefinition = False
        self.function_name = ''
        self.is_faithful = True

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        # silently absorb any missing attribute access plum may attempt
        return None

# Inject the stub into plum._resolver so pickle.find_class can resolve it
try:
    import plum._resolver as _plum_resolver
    _existing = getattr(_plum_resolver, 'Resolver', None)
    if _existing is not None:
        _instance = _existing()
        if not hasattr(_instance, 'dict'):
            # v2: replace with stub
            _plum_resolver.Resolver = _Resolverv1Stub
except ImportError:
    # plum not installed yet; inject a fake module so find_class works
    import types
    _fake_plum = types.ModuleType('plum')
    _fake_resolver_mod = types.ModuleType('plum._resolver')
    _fake_resolver_mod.Resolver = _Resolverv1Stub
    sys.modules.setdefault('plum', _fake_plum)
    sys.modules.setdefault('plum._resolver', _fake_resolver_mod)

# Also inject plum._util.Missing if missing
try:
    import plum._util
    if not hasattr(plum._util, 'Missing'):
        class _Missing: pass
        plum._util.Missing = _Missing
except (ImportError, Exception):
    import types
    _fake_util = types.ModuleType('plum._util')
    class _Missing: pass
    _fake_util.Missing = _Missing
    sys.modules.setdefault('plum._util', _fake_util)

# ── fasttransform alias: pkl references fasttransform.transform ───────────────
# At training time, fastai's transform module was briefly packaged as
# 'fasttransform'. If not installed, alias it to fastai.
try:
    import fasttransform  # noqa: F401 — installed via requirements.txt
except ImportError:
    try:
        import types
        import fastai.data.transforms as _fdt
        import fastai.vision.augment as _fva

        _ft = types.ModuleType('fasttransform')
        _ft_transform = types.ModuleType('fasttransform.transform')

        # Alias the classes the pkl needs from fastai
        _ft_transform.Pipeline = getattr(_fdt, 'Pipeline', None) or \
                                  getattr(__import__('fastai.torch_core',
                                          fromlist=['Pipeline']), 'Pipeline', None)
        _ft_transform.Transform = getattr(_fdt, 'Transform', None) or \
                                   getattr(__import__('fastai.torch_core',
                                           fromlist=['Transform']), 'Transform', None)

        sys.modules.setdefault('fasttransform', _ft)
        sys.modules.setdefault('fasttransform.transform', _ft_transform)
    except Exception:
        pass  # Let the real load error surface with full traceback

# ── __builtin__ alias (Python 2 pickle compat) ────────────────────────────────
import builtins
import types as _types
_builtin_mod = _types.ModuleType('__builtin__')
for _name in ('getattr', 'unicode', 'bytes', 'tuple', 'set',
              'reduce', 'long', 'float', 'print'):
    setattr(_builtin_mod, _name, getattr(builtins, _name, None))
sys.modules.setdefault('__builtin__', _builtin_mod)

# ── Now import fastai (AFTER all patches are in place) ────────────────────────
from fastai.vision.all import *
from PIL import Image


# Required label function — must exist in namespace before load_learner
def get_yolo_label(fname):
    """Label function used during training — required for unpickling."""
    return 'threat'


@st.cache_resource
def load_my_model():
    return load_learner('threat_classifier_v1.pkl')


# ── App UI ────────────────────────────────────────────────────────────────────
st.title('🛡️ X-Ray Contraband Detector')
st.write('Upload an X-ray image to identify potential threats using our trained Fast.ai model.')

model_loaded = False
learn = None
categories = ['safe', 'threat']

try:
    learn = load_my_model()
    categories = learn.dls.vocab
    model_loaded = True
except Exception as e:
    st.error(f'Error loading model: {e}')
    with st.expander('Full traceback'):
        import traceback
        st.code(traceback.format_exc())

uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_container_width=True)

    if not model_loaded:
        st.warning('Model failed to load — classification is unavailable.')
    elif st.button('Classify'):
        with st.spinner('Analyzing...'):
            uploaded_file.seek(0)           # reset stream after PIL read
            fast_img = PILImage.create(uploaded_file)
            pred, pred_idx, probs = learn.predict(fast_img)

            st.subheader(f'Prediction: {pred.upper()}')

            prob_dict = {categories[i]: float(probs[i]) for i in range(len(categories))}
            st.bar_chart(prob_dict)

            conf = float(probs[pred_idx]) * 100
            st.info(f'Confidence Level: {conf:.2f}%')
