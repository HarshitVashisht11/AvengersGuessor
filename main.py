from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained Keras model
model = load_model('')

# Define a function to preprocess the image before feeding it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming your model expects input shape of (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['imageFile']

        # Save the uploaded file
        uploaded_file.save('uploaded_image.jpg')

        # Preprocess the image
        processed_image = preprocess_image('uploaded_image.jpg')

        # Use the model to make a prediction
        prediction = model.predict(processed_image)
        # Assuming you have a list of classes
        classes = ['Iron Man', 'Captain America', 'Thanos', 'Hulk', 'Black Widow', 'Spiderman']

        # Get the predicted class
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]

        return render_template('result.html', predicted_class=predicted_class)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
