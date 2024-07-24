import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
from keras.models import load_model

# Load data
disease_info = pd.read_csv('C:/Users/vaibh/Desktop/Skin_Diseases_detection/Flask Deployed App/disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('C:/Users/vaibh/Desktop/Skin_Diseases_detection/Flask Deployed App/supplement_info.csv', encoding='cp1252')

# Load model
model = load_model("C:/Users/vaibh/Desktop/Skin_Diseases_detection/Flask Deployed App/skin_diseases_detection.h5")

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  
    image = np.array(image)  
    if image.shape[2] == 4: 
        image = image[:, :, :3]
    image = image / 255.0  
    input_data = np.expand_dims(image, axis=0) 
    output = model.predict(input_data) 
    index = np.argmax(output)
    return index



app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        upload_folder = 'static/uploads'
        
        # Ensure the upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)
        try:
            pred = prediction(file_path)
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                    image_url=image_url, pred=pred, sname=supplement_name,
                                    simage=supplement_image_url, buy_link=supplement_buy_link)
        except Exception as e:
            return f"An error occurred: {e}", 500

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                            supplement_name=list(supplement_info['supplement name']),
                            disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
