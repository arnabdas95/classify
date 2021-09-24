
import PIL
from joblib  import load
import io
from PIL import Image, ImageOps
from urllib.request import urlopen
from flask import Flask , request, jsonify,render_template
import numpy as np
import scipy


def image_preprocessing(result,int_position):
        # strip the result
        with urlopen(result) as response:
                result = response.read()

        # convert it numpy array
        img = Image.open(io.BytesIO(result))
        arr = np.asarray(img)

        # make pillow image object
        arr = Image.fromarray(arr)

        #crop image to fit the digit to the whole
        arr = arr.crop((int_position[0], int_position[1], int_position[2], int_position[3]))

        # resize images to 28,28
        arr = arr.resize((28, 28))
        #arr.show()
        # make white background png
        arr = arr.convert("RGBA")
        arr1 = Image.new("RGBA", arr.size, "WHITE")
        arr1.paste(arr, mask=arr)
        arr = arr1.convert("RGB")

        # make greyscale image
        arr = arr.convert("L")
        # invert the image
        arr = PIL.ImageOps.invert(arr)
        # make numpy image array
        n_arr = np.array(arr)
        #convert into float
        n_arr = n_arr.astype(float)
        n_arr = n_arr.reshape(784)
        #inteoduce comma seperated as mnist data set
        n_arr=list(n_arr)

        return n_arr

app = Flask(__name__)
trained_model=load("ml_model/best_KNN_clf")
@app.route("/")
def index():

        return render_template('index.html')

@app.route("/predict",methods=['post'])
def predict():

        #get uri from javascript canvas as result
        result=request.form.get('text')
        position = request.form.get('text_two')

        position= position.split(',')
        int_position=[]
        for item in position:
                print(item)
                item=int(item)

                int_position.append(item)
        #try to fix outside of the canvas line
        if int_position[0] <0:
                int_position[0]=0
        if int_position[1] <0:
                int_position[1]=0
        if int_position[2] >300:
                int_position[2]=300
        if int_position[3] >300:
                int_position[3]=300
       # print(int_position)
        #preprocessing
        n_arr=image_preprocessing(result,int_position)
        predicted_number = trained_model.predict([n_arr])

        return render_template('index.html',a="PREDICTED NUMBER = {}".format(predicted_number[0]))

@app.route("/about")
def about():
        return render_template('about.html')

if __name__ == '__main__':
   app.run()