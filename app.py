from flask import Flask, flash, jsonify, render_template, redirect, request, url_for, send_file
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = 'super secret key'
ML_MODEL_FILENAME = 'models/custom_cifar10_model2.h5'

# categories
X = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# samples
mySampleX = ['./static/class0.jpeg','./static/class1.jpeg','./static/class2.jpeg','./static/class3.jpeg','./static/class4.jpeg','./static/class5.jpeg','./static/class6.jpeg','./static/class7.jpeg','./static/class8.jpeg','./static/class9.jpeg']
airplane = './static/class0.jpeg'
automobile = './static/class1.jpeg'
bird= './static/class2.jpeg'
cat= './static/class3.jpeg'
deer= './static/class4.jpeg'
dog= './static/class5.jpeg'
frog= './static/class6.jpeg'
horse= './static/class7.jpeg'
ship= './static/class8.jpeg'
truck= './static/class9.jpeg'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

results = []


def getPrediction(filename):
    myModel = load_model(ML_MODEL_FILENAME)

    image = load_img('static/uploads/'+filename, target_size=(32, 32))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = myModel.predict(image)
    label = X[np.argmax(yhat)]
    acc = max(yhat.tolist()[0])*100

    return label, acc

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    ''' webpage load and uploading of files'''
    if request.method == 'GET':
        return render_template('index.html', myX=X, mySampleX=mySampleX)
    else:
        if 'file' not in request.files:
            flash('No file found')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash(f'Not an allowed file type, please provide one of following types {ALLOWED_EXT}')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            getPrediction(filename)
            label, acc = getPrediction(filename)
            flash(label)
            flash(acc)
            flash(filename)
            # return redirect(url_for('upload_file', filename=filename))
            return redirect('/')


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print('Num GPUs Availables: ', len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
    app.run()

if __name__ == '__main__':
    app.debug = True
    main()