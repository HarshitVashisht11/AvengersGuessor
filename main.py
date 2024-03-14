from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename 
from tensorflow.keras.preprocessing import image
from marvel import Marvel
import numpy as np
import os
import requests

app = Flask(__name__)

model = load_model("trained_model.h5")

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

def fetch_character_details(character_name):
    PUBLIC_KEY=""
    PRIVATE_KEY=""
        
    marvel = Marvel(PUBLIC_KEY, PRIVATE_KEY)
    characters = marvel.characters
    characters_data = characters.all(nameStartsWith=character_name)["data"]["results"]
    
    if characters_data:
        character_details = []
        for char in characters_data:
            comics = [comic["name"] for comic in char["comics"]["items"]]
            series = [serie["name"] for serie in char["series"]["items"]]
            movies = [movie["title"] for movie in char["movies"]["items"]]
            character_details.append({
                "name": char["name"],
                "description": char["description"],
                "comics": comics,
                "series": series,
                "movies": movies
            })
    else:
        character_details = None

@app.route('/')
def index():
    return render_template('index.html')

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
                classes = ['Black Widow', 'Captain America', 'Dr. Strange', 'Hulk', 'Iron-Man', 'Loki', 'Spider-Man', 'Thanos']
                predicted_class_index = np.argmax(prediction)
                predicted_class = classes[predicted_class_index]
                character_details = fetch_character_details(predicted_class)
                return redirect(url_for('result', predicted_class=predicted_class, uploaded_image=filename, character_details=character_details))
            else:
                print("Error: Preprocessed image is None")
    return render_template('upload.html')

@app.route('/result')
def result():
    predicted_class = request.args.get('predicted_class')
    uploaded_image = request.args.get('uploaded_image')
    character_details = request.args.get('character_details')
    return render_template('result.html', predicted_class=predicted_class, uploaded_image=uploaded_image, character_details=character_details)

if __name__ == '__main__':
    app.run(debug=True)
