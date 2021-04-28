from flask import Flask, request, render_template
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from urllib import parse

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = '/uploads'
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = tensorflow.keras.models.load_model('model/keras_model.h5', compile = False)
image_path = 'data/test_photo.jpg'

@app.route('/upload')
def upload():
   return render_template('/upload.html')

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    img_b64 = request.args.get('img')
    image = Image.open(io.BytesIO(base64.b64decode(img_b64)))

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return str(prediction[0])
    
    
@app.route('/b64',methods = ['POST', 'GET'])
def image_to_b64():
    with open(image_path, "rb") as img_file:
        str_b64 = base64.b64encode(img_file.read())
        return parse.quote_plus(str_b64)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('model.h5')
      return 'file uploaded successfully'