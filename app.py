from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

features = pickle.load(open("encodings.pkl", "rb"))
model = load_model('model_3.h5', compile = False)
images = "images/"
max_length = 124
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))

def Image_Caption(picture):

    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Modify the following line to ensure that both arrays have the same number of samples
        yhat = model.predict([np.repeat(picture, len(sequence), axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq but', '')
    #final = in_text.split()
    #final = final[1:-1]
    #final = ' '.join(final)
    return in_text

@app.route('/')
def index():

    return render_template('index.html', appName="Image Captioning Model")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        z = 1000
        pic = list(features.keys())[z]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = features[pic].reshape((1,14))
        x = plt.imread(images+pic)
        print("Caption:", Image_Caption(image))
        print("Model predicting ...")
        result = Image_Caption(image)
        print("Model predicted")
        print(result)
        return jsonify({'prediction': result})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        z = 1000
        pic = list(features.keys())[z]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = features[pic].reshape((1,14))
        x = plt.imread(images+pic)
        print("Caption:", Image_Caption(image))
        print("Model predicting ...")
        result = Image_Caption(image)
        print("Model predicted")
        print(result)

        return render_template('index.html', prediction=result, image='static/IMG/', appName="Image Captioning Model")
    else:
        return render_template('index.html',appName="Image Captioning Model")


if __name__ == '__main__':
    app.run(debug=True)