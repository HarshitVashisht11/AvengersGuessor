from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename 
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)

model = load_model("trained_model.h5")

# Function to preprocess image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

# Function to load character data from JSON file
def load_character_data(file_path):
    try:
        with open(file_path, 'r') as file:
            character_data = json.load(file)
        return character_data
    except Exception as e:
        print("Error loading character data:", e)
        return None

# Function to find character details from loaded data
def find_character_details(character_name, data):
    try:
        characters = data['characters']
        for character in characters:
            if character['name'] == character_name:
                return character
    except Exception as e:
        print("Error finding character details:", e)
    return None

# Route to index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload page
@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['imageFile']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            upload_dir = 'uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            uploaded_file_path = os.path.join(upload_dir, filename)
            uploaded_file.save(uploaded_file_path)
            print("Uploaded file saved as:", uploaded_file_path)
            processed_image = preprocess_image(uploaded_file_path)
            if processed_image is not None:
                print("Processed image shape:", processed_image.shape)
                prediction = model.predict(processed_image)
                print("Prediction:", prediction)
                classes = ['Black Widow', 'Captain America', 'Dr. Strange', 'Hulk', 'Iron Man', 'Loki', 'Spider-Man', 'Thanos']  # Adjust classes as needed
                predicted_class_index = np.argmax(prediction)
                predicted_class = classes[predicted_class_index]
                character_data = load_character_data('marvel_data.json')  # Load character data from JSON file
                character_details = find_character_details(predicted_class, character_data)  # Find character details
                return render_template('result.html', predicted_class=predicted_class, uploaded_image=filename, character_details=character_details)
            else:
                print("Error: Preprocessed image is None")
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
