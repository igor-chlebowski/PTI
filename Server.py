import random
from flask import Flask, request, jsonify
from PIL import Image
from main import run    
import uuid
import os

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

def handle_file(image_file) -> str:
    ext = os.path.splitext(image_file.filename)[1]
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        raise ValueError("Unsupported file type. Please upload an image file.")

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save image to the upload folder
    image_file.save(filepath)
    return filepath

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    try:
        image_path = handle_file(image_file)

        output_dir = 'output' + random.randint(0, 1000).__str__()
        result = run(image_path, output_dir)

        if result == "error":
            return jsonify({'error': 'Error processing the image'}), 500
        else:
            print(f"Image processed successfully. Output saved in {output_dir}")
            return jsonify({'message': 'Image processed successfully', "res": result, 'output_dir': output_dir})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
