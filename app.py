import numpy as np
from flask import Flask, flash, jsonify, render_template, redirect, request, url_for, send_file
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('.\\models\\custom_cifar10_cbs.h5')
app.secret_key = 'super secret key'
ML_MODEL_FILENAME = 'models/vgg16_cifar10_model_vgg16prepro.h5'

# categories
X = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# samples
mySampleX = ['./static/class0.jpeg','./static/class1.jpeg','./static/class2.jpeg','./static/class3.jpeg','./static/class4.jpeg','./static/class5.jpeg','./static/class6.jpeg','./static/class7.jpeg','./static/class8.jpeg','./static/class9.jpeg']
airplane = './static/class0.jpeg'
automobile = './static/class0.jpeg'
bird= './static/class0.jpeg'
cat= './static/class0.jpeg'
deer= './static/class0.jpeg'
dog= './static/class0.jpeg'
frog= './static/class0.jpeg'
horse= './static/class0.jpeg'
ship= './static/class0.jpeg'
truck= './static/class0.jpeg'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

results = []

def load_model_from_file():

    mySession = tf.compat.v1.Session()
    K.set_session(mySession)
    myModel = load_model(ML_MODEL_FILENAME)
    myGraph = tf.compat.v1.get_default_graph()
    return (mySession, myModel, myGraph)

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
            return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(os.path.join(UPLOAD_FOLDER, filename), target_size=(32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGRAPH = app.config['GRAPH']

    # with myGRAPH.as_default():
    K.set_session(mySession)
    result = np.argmax(myModel.predict(test_image), axis=-1)
    img_src = os.path.join(UPLOAD_FOLDER, filename)
    answer = "<div class='col text-center'><img width='150' height='150' src='" + img_src + "' class='img-thumbnail' /><h4>guess:" + X + " " + str(
        result) + "</h4></div><div class='col'></div><div class='w-100'></div>"
    results.append(answer)
    return render_template('index.html', myX=X, mySampleX=mySampleX, len=len(results), results=results)

def main():
    (mySession, myModel, myGraph) = load_model_from_file()

    tf.compat.v1.enable_eager_execution()
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
    app.run()

if __name__ == '__main__':
    app.debug = True
    main()