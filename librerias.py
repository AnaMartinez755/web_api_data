from flask import Flask, render_template, request
import urllib.request
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from flask import jsonify
import uuid
from scipy import signal
import json
import pandas as pd

def get_library_version(library_name):
    try:
        library = __import__(library_name)
        if hasattr(library, '__version__'):
            return library.__version__
        elif hasattr(library, 'version'):
            return library.version
        else:
            return "Unknown"
    except ImportError:
        return "Not installed"

libraries = {
    'Flask': Flask,
    'urllib': urllib.request,
    'tensorflow.keras': load_model,
    'keras.applications.vgg16': decode_predictions,
    'numpy': np,
    'dotenv': load_dotenv,
    'cloudinary': cloudinary.uploader,
    'scipy': signal,
    'pandas': pd,
}

for library_name, library in libraries.items():
    version = get_library_version(library_name)
    print(f"{library_name}: {version}")
