import os
from PIL import Image
import glob
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath

from PIL import Image, ImageOps
import torchvision as tv
from torchvision import models
import torch
from function import MedNet, pred_body
import matplotlib.pyplot as mp

model = torch.load('saved_model.sav')
classNames = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
app = Flask(__name__)

@app.route('/' ,methods = ['GET', 'POST'])
def home():
    return render_template("index.html")

def pred_body(filename, model, classNames):
    img = Image.open(filename)
    gray = ImageOps.grayscale(img)
    toTensor = tv.transforms.ToTensor()
    y = toTensor(gray)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    y = z.reshape([1,1,64,64])
    pred = model(y)
    classNames[pred.max(1)[1]]
    return classNames[pred.max(1)[1]]

@app.route('/uploader', methods = ['GET', 'POST'])
def result():  
    
    #upload_path = join(dirname(realpath(__file__)), "static\\img")
    #IMAGE_UPLOADS = upload_path
    #app.config["IMAGE_UPLOADS"] = upload_path
    
    if request.method == "POST":
        file = request.files["file1"]
        
        print(redirect(request.url))
        tt=Image.open(file)
        #file.save(os.path.join(app.config['IMAGE_UPLOADS'], file.filename))
        pred = pred_body(file, model, classNames)
        return render_template("index.html",img = file, predict= 'prediction class is {}'.format(pred))

if __name__ == '__main__':
    app.run(debug=True)