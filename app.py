from flask import *
import os
from werkzeug.utils import secure_filename
import label_image
import image_fuzzy_clustering as fem
import numpy as np
from PIL import Image
from flask import current_app
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

app = Flask(__name__)
model = None

# Upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -------------------- FUNCTIONS --------------------

def load_image(image):
    text = label_image.main(image)
    return text


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def save_img(img, filename):
    folder_path = os.path.join(current_app.root_path, 'static', 'images')

    # create folder if not exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    picture_path = os.path.join(folder_path, filename)

    i = Image.open(img)
    i.save(picture_path)

    return picture_path


# -------------------- ROUTES --------------------

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/upload')
def upload():
    return render_template('index1.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        i = request.form.get('cluster')
        f = request.files['file']

        original_pic_path = save_img(f, f.filename)

        # clustering
        fem.plot_cluster_img(original_pic_path, i)

    return render_template('success.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':

        f = request.files['file']

        # secure path
        file_path = os.path.join("static", secure_filename(f.filename))
        f.save(file_path)

        # prediction
        result = load_image(file_path)
        result = result.title()

        d = {
            "1": " → Stage1 - The disease is only in ducts and lobules (noninvasive).",
            "2": " → Stage2 - Cancer has spread to nearby tissue or lymph nodes.",
            "3": " → Stage3 - Cancer spread to multiple lymph nodes.",
            "4": " → Stage4 - Cancer spread beyond breast to other body parts.",
            "0": " → No Breast Cancer detected. You are healthy."
        }

        if result in d:
            result = result + d[result]

        print(result)

        # remove uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    return None


# -------------------- MAIN --------------------

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT)