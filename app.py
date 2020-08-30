from flask import Flask, render_template,request
import os
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.models import load_model
import tensorflow as tf
global graph
graph = tf.get_default_graph()
import numpy as np
model = load_model(r"E:\VEC-Rsip\skindisease identification/skindidease.h5")
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("base.html", methods = ['GET'])
@app.route('/predict',methods = ['GET','POST'])
def pred():
    if request.method == "POST":
        f = request.files["image"]
        print("hie")
        """take the path of  the current
        running prog,and concatenate to the
        folder where you wouldlike to save the file"""
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        with graph.as_default():
            p = model.predict_classes(x)
            print(p)
        index = ["Acne","Melanoma","Psoriasis","Rosacea","Vitiligo"]
        text = "the prediction is :" +index[p[0]]
        return text
if __name__ == "__main__":
    app.run(debug = True)
