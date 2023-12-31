from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pickle
import requests  # Ajoutez cette ligne

app = Flask(__name__)

# Chargez le modèle lors du démarrage de l'application
model_url = 'http://smart.marocmeteo.ma/build/model_classification_images.pkl'
response = requests.get(model_url)

if response.status_code == 200:
    with open('model_classification_images.pkl', 'wb') as file:
        file.write(response.content)
        
    # Chargez le modèle
    model = pickle.load(open('model_classification_images.pkl', 'rb'))
else:
    raise Exception(f"Échec du téléchargement du modèle. Code d'état : {response.status_code}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='Aucun fichier sélectionné')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='Aucun fichier sélectionné')

    if file:
        # Chargez et affichez l'image téléchargée
        image = Image.open(file)
        image_array = np.array(image.resize((150, 150)))
        img_tensor = np.expand_dims(image_array, axis=0) / 255.0

        # Faites des prédictions
        prediction = model.predict(img_tensor)

        # Affichez le résultat
        if prediction[0][0] > 0.5:
            result = 'Chien'
        else:
            result = 'Chat'

        return render_template('result.html', prediction=result, confidence=prediction[0][0])

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
