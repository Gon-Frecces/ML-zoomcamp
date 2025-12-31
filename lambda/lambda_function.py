
import numpy as np
import onnxruntime as ort

from io import BytesIO
from urllib import request
from PIL import Image


# ---------- Image helpers ----------

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(img):
    img = np.array(img, dtype=np.float32)

    # scale to [0, 1]
    img = img / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    return img.astype('float32')


# ---------- Model loading ----------

model_path = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# ---------- Lambda handler ----------

def lambda_handler(event, context):
    image_url = event['url']

    img = download_image(image_url)
    img = prepare_image(img)
    img_np = preprocess(img)

    prediction = session.run(
        [output_name],
        {input_name: img_np}
    )

    score = float(prediction[0][0][0])

    return {
        'score': score
    }
