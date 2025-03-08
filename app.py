import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load TensorFlow Hub Style Transfer Model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Image processing functions
def crop_center(image):
    """Crops the image to a square shape."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

def load_image(image_path, image_size=(512, 512)):
    """Loads and preprocesses an image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=False)
    return img

def stylize_image(content_path, style_path):
    """Applies style transfer and saves the output image."""
    content_image = load_image(content_path)
    style_image = load_image(style_path, image_size=(256, 256))
    
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Convert to uint8 format and save
    stylized_image = np.array(stylized_image[0] * 255, dtype=np.uint8)
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'stylized_output.png')
    Image.fromarray(stylized_image).save(result_path)
    
    return result_path

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        content_file = request.files['content']
        style_file = request.files['style']

        if content_file and style_file:
            content_filename = secure_filename(content_file.filename)
            style_filename = secure_filename(style_file.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            # Save images
            content_file.save(content_path)
            style_file.save(style_path)

            # Generate stylized image
            result_path = stylize_image(content_path, style_path)

            return render_template('index.html', content_img=content_filename, style_img=style_filename, result_img=os.path.basename(result_path))

    return render_template('index.html', content_img=None, style_img=None, result_img=None)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
