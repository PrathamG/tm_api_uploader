from flask import Flask, request, render_template
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from urllib import parse
import hashlib
import json
import requests
import h5py

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
image_path = 'data/test_photo.jpg'
key = 'eabe9b121192:000351b2c4b29f9c323533b828974fce8357b0013a'

def get_auth_resp(key):
    # Authorize account and get api URL
    basic_auth_string = 'Basic ' + str(base64.b64encode(key.encode('utf-8')))[2:-1]
    headers = { 'Authorization': basic_auth_string }
    url = 'https://api.backblazeb2.com/b2api/v2/b2_authorize_account'
    response = requests.get(
        url = url,
        headers = headers
        )
    auth_resp = response.json()
    return auth_resp

def get_urlreq_resp(auth_resp):
    api_url = auth_resp['apiUrl'] # Provided by b2_authorize_account
    account_authorization_token = auth_resp['authorizationToken'] # Provided by b2_authorize_account
    bucket_id = "2ecadb9e794b11d271910912" # The ID of the bucket you want to upload your file to

    url = '%s/b2api/v2/b2_get_upload_url' % api_url
    headers = { 'Authorization': account_authorization_token }
    params = { 'bucketId' : bucket_id }

    response = requests.get(url = url, params = params, headers = headers)
    urlreq_resp = response.json()
    return urlreq_resp

@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'GET':
        return render_template('/upload.html')
    elif request.method == 'POST':
        uploaded_model_file = request.files['model']
        uploaded_label_file = request.files['labels']
        auth_resp = get_auth_resp(key)
        urlreq_resp = get_urlreq_resp(auth_resp)

        # Encrypt files with SHA1. Encrypted file is also necessary to upload to b2
        model_file = uploaded_model_file.read()
        model_sha1Hash = hashlib.sha1(model_file)
        model_sha1Hashed = model_sha1Hash.hexdigest()

        label_file = uploaded_label_file.read()
        label_sha1Hash = hashlib.sha1(label_file)
        label_sha1Hashed = label_sha1Hash.hexdigest()

        #Retrieve header params for the b2 API calls
        upload_url = urlreq_resp['uploadUrl'] # Provided by b2_get_upload_url
        upload_authorization_token = urlreq_resp['authorizationToken'] # Provided by b2_get_upload_url
        
        # Upload both files
        file_name = "model.h5"
        content_type = "application/x-hdf5"
        headers = {
            'Authorization' : upload_authorization_token,
            'X-Bz-File-Name' :  file_name,
            'Content-Type' : content_type,
            'X-Bz-Content-Sha1' : model_sha1Hashed
            }
        response = requests.post(upload_url, model_file, headers = headers)
        model_file_id = response.json()['fileId']

        file_name = "labels.txt"
        content_type = "text/plain"
        headers = {
            'Authorization' : upload_authorization_token,
            'X-Bz-File-Name' :  file_name,
            'Content-Type' : content_type,
            'X-Bz-Content-Sha1' : label_sha1Hashed
            }
        response = requests.post(upload_url, label_file, headers = headers)
        label_file_id = response.json()['fileId']

        return "Label ID: " + label_file_id + "<br>" + "Model ID: " + model_file_id + "<br><br> Please keep a copy of these IDs. They will be required to use the API"

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    auth_resp = get_auth_resp(key)

    img_b64 = request.args.get('img')
    img_b64 = parse.unquote(img_b64, encoding='ascii', errors='replace')
    
    download_url = auth_resp['apiUrl']
    auth_token = auth_resp['authorizationToken']
    headers = {'Authorization': auth_token}
    
    model_id = request.args.get('model') 
    model_url = download_url + '/b2api/v2/b2_download_file_by_id?fileId=' + model_id
    model_resp = requests.get(url = model_url, headers = headers)
    model_bytes = io.BytesIO(model_resp.content)
    model_file = h5py.File(model_bytes,'r')
    model = tensorflow.keras.models.load_model(model_file, compile = False)

    label_id = request.args.get('label') 
    label_url = download_url + '/b2api/v2/b2_download_file_by_id?fileId=' + label_id
    label_resp = requests.get(url = label_url, headers = headers)
    labels = label_resp.text
    labels_list = labels.split('\n')
    for idx, label in enumerate(labels_list):
        labels_list[idx] = label[2:]
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
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
    prediction = model.predict(data)[0]
    prediction = list(map(str, prediction))
    pred_dict = dict(zip(labels_list, prediction))

    response = app.response_class(
        response=json.dumps(pred_dict),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/b64',methods = ['POST', 'GET'])
def image_to_b64():
    with open(image_path, "rb") as img_file:
        str_b64 = base64.b64encode(img_file.read())
        return parse.quote_plus(str_b64)