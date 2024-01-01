import base64
from io import BytesIO
from flask import Flask, render_template, request, url_for, jsonify, send_file
import torch 
import numpy as np
from models.model_plain import ModelPlain
from utils.model import define_model
from utils.utils_image import single2tensor3, single2tensor4, tensor2img
from utils import utils_option as option
import torchvision.transforms as transforms
import argparse
import cv2
import PIL.Image


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

device = torch.device('cuda')

def im2tensor(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
    return img

model = define_model()
model.eval()
model = model.to(device)

def get_image(image_tensor):
    window_size = 8
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = image_tensor.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        image_tensor = torch.cat([image_tensor, torch.flip(image_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        image_tensor = torch.cat([image_tensor, torch.flip(image_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(image_tensor)
        output = output[..., :h_old * 1, :w_old * 1]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    image = transforms.ToPILImage("RGB")(output[:, :, [2, 1, 0]])
    return image

def serve_pil_image64(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG')
    return img_io
    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/denoise', methods=['GET','POST'])
def denoise():
     if request.method == 'POST':
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

       
        tensor = im2tensor(file)
        new_image = get_image(tensor)
        response = serve_pil_image64(new_image)
        response.seek(0)
        return send_file(response, mimetype='image/jpeg')
        

if __name__ == '__main__':
    app.run(debug=True)