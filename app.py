"""
X-Ray Contraband Detector — Streamlit App
==========================================
Pickle compatibility layer for loading a fastai model trained with:
  - plum-dispatch 1.x  (Resolver had .dict instance attribute)
  - fasttransform      (Pipeline/Transform were in fasttransform.transform)
  - Python 3.9/3.10   (PosixPath in pkl, __builtin__ module references)
on a Streamlit Cloud environment running Python 3.14 + plum-dispatch 2.x.

Strategy: monkey-patch pickle.Unpickler globally BEFORE any fastai/torch
import so that all subsequent pickle.load() calls — including the ones deep
inside torch.load() / torch.serialization — transparently remap the old
class names to compatible stubs or current equivalents.
"""

import sys

# Streamlit Cloud caches the venv. If fasttransform was previously installed,
# pip leaves it there even after removing it from requirements.txt.
# We MUST block it from being imported, otherwise fastai will try to load it
# and crash with an IndexError.
sys.modules['fasttransform'] = None
sys.modules['fasttransform.transform'] = None

import io
import pickle
import pathlib
import platform
import builtins
import types

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build compatibility stubs for plum v1 classes
# ─────────────────────────────────────────────────────────────────────────────
# plum-dispatch v1.x  →  Resolver was a plain Python class with __dict__
# plum-dispatch v2.x  →  Resolver is a Cython extension; instances have no
#                         __dict__, so pickle's BUILD opcode crashes with
#                         "'Resolver' object has no attribute 'dict'"
#
# We pin plum-dispatch==1.7.4 in requirements.txt so the REAL classes are
# available, but as a safety net we also register these stubs so that if
# any plum internal is missing we never crash hard.

class _ResolverStub:
    """plum v1 Resolver: had .dict, .methods, ._resolved, .warn_redefinition"""
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, name):
        return None

class _PlumStub:
    """Generic stub for any other plum v1 class (Method, MethodList, etc.)"""
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, name):
        return None
    def __call__(self, *a, **kw):
        return None

_MISSING = type('Missing', (), {'__repr__': lambda s: 'Missing'})()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Cross-platform pathlib remap
# ─────────────────────────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Pre-populate sys.modules so find_class() can resolve them
# ─────────────────────────────────────────────────────────────────────────────

# __builtin__  (Python 2 pickle writes module='__builtin__')
if '__builtin__' not in sys.modules:
    _bi = types.ModuleType('__builtin__')
    for _n in ('getattr', 'setattr', 'hasattr', 'isinstance',
               'unicode', 'str', 'bytes', 'tuple', 'list', 'set', 'dict',
               'reduce', 'long', 'int', 'float', 'bool', 'print',
               'object', 'type', 'super', 'staticmethod', 'classmethod'):
        setattr(_bi, _n, getattr(builtins, _n, getattr(builtins, 'str', str)))
    sys.modules['__builtin__'] = _bi


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Global pickle.Unpickler monkey-patch
# ─────────────────────────────────────────────────────────────────────────────
# torch.load (all versions) eventually calls pickle.Unpickler(file).load().
# By replacing pickle.Unpickler we intercept every class lookup regardless
# of which code path torch/fastai takes.

_OriginalUnpickler = pickle.Unpickler


class _CompatUnpickler(_OriginalUnpickler):
    """
    Custom Unpickler that remaps:
      - plum v1 classes  → lightweight stubs with __dict__
      - fasttransform    → fastai equivalents
      - pathlib.PosixPath → pathlib.Path  (cross-platform)
      - __builtin__.*    → builtins.*     (Python 2 compat)
    """

    # (module, name) → replacement class
    _REMAP = {
        # plum v1 → stubs
        ('plum._resolver',  'Resolver'):   _ResolverStub,
        ('plum._function',  'Function'):   _PlumStub,
        ('plum._method',    'Method'):     _PlumStub,
        ('plum._method',    'MethodList'): _PlumStub,
        ('plum._signature', 'Signature'):  _PlumStub,
        ('plum._util',      'Missing'):    type('Missing', (), {}),
        # pathlib cross-platform
        ('pathlib', 'PosixPath'):   pathlib.Path,
        ('pathlib', 'WindowsPath'): pathlib.Path,
    }

    def find_class(self, module, name):
        # 1. Direct remap table
        obj = self._REMAP.get((module, name))
        if obj is not None:
            return obj

        # 2. plum catch-all: try real module, fall back to generic stub
        if module.startswith('plum'):
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError, ModuleNotFoundError):
                return _PlumStub

        # 3. fasttransform → fastai mapping
        if module.startswith('fasttransform'):
            # try installed fasttransform package first
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError, ModuleNotFoundError):
                pass
            # remap to fastai equivalents
            _ft_map = {}
            try:
                import fastai.data.transforms as _fdt
                import fastai.torch_core as _ftc
                for _attr in ('Pipeline', 'Transform', 'TfmdLists'):
                    for _mod in (_fdt, _ftc):
                        if hasattr(_mod, _attr):
                            _ft_map[_attr] = getattr(_mod, _attr)
                            break
            except ImportError:
                pass
            if name in _ft_map:
                return _ft_map[name]
            return _PlumStub  # non-critical class, use stub

        # 4. __builtin__ → builtins (Python 2 pickle compat)
        if module == '__builtin__':
            try:
                return getattr(builtins, name)
            except AttributeError:
                return getattr(builtins, 'str', str)  # safe fallback

        # 5. Default behaviour
        return super().find_class(module, name)


# Apply the patch globally
pickle.Unpickler = _CompatUnpickler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: NOW import fastai (after all patches are live)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from fastai.vision.all import *
from PIL import Image


# Required label function — must be in scope before load_learner unpickles it
def get_yolo_label(fname):
    return 'threat'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Load model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return load_learner('threat_classifier_v1.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title('🛡️ X-Ray Contraband Detector')
st.write('Upload an X-ray image to identify potential threats using our trained Fast.ai model.')

model_loaded = False
learn = None
categories = ['safe', 'threat']

try:
    learn = load_my_model()
    categories = learn.dls.vocab
    model_loaded = True
except Exception as _e:
    st.error(f'Error loading model: {_e}')
    with st.expander('🔍 Full traceback (for debugging)'):
        import traceback
        st.code(traceback.format_exc())

uploaded_file = st.file_uploader('Choose an X-ray image…', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_container_width=True)

    if not model_loaded:
        st.warning('Model failed to load — classification is unavailable.')
    elif st.button('Classify'):
        with st.spinner('Analyzing…'):
            uploaded_file.seek(0)           # reset stream after PIL.Image.open consumed it
            fast_img = PILImage.create(uploaded_file)
            pred, pred_idx, probs = learn.predict(fast_img)

            st.subheader(f'Prediction: {pred.upper()}')

            prob_dict = {categories[i]: float(probs[i]) for i in range(len(categories))}
            st.bar_chart(prob_dict)

            conf = float(probs[pred_idx]) * 100
            st.info(f'Confidence Level: {conf:.2f}%')
