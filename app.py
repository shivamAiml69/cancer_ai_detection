from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
import joblib
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =========================
# 📁 CONFIG
# =========================
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================
# 🧠 LOAD MODELS
# =========================

# CNN model
cnn_model = tf.keras.models.load_model("cancer_model.h5")

# RF model
rf_model = joblib.load("rf_model.pkl")

# Feature extractor
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224,224,3)
)

# =========================
# 🔥 GRAD-CAM FUNCTION
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def save_gradcam(img_path, heatmap):

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    cam_path = os.path.join("static/uploads", "cam.jpg")
    cv2.imwrite(cam_path, superimposed)

    return cam_path


# =========================
# 🏠 HOME ROUTE
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# 📂 PREDICT ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    # Secure filename
    if not file.filename:
        return render_template('index.html', error="No file selected")
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ================= PREPROCESS =================
    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ================= CNN =================
    pred = cnn_model.predict(img_array)[0][0]

    if pred > 0.5:
        cnn_label = "Cancer"
        cnn_conf = pred * 100
    else:
        cnn_label = "No Cancer"
        cnn_conf = (1 - pred) * 100

    # ================= RF =================
    features = base_model.predict(img_array, verbose=0)[0]
    rf_pred = rf_model.predict([features])[0]

    rf_label = "Cancer" if rf_pred == 1 else "No Cancer"

    # ================= GRAD-CAM =================
    heatmap = make_gradcam_heatmap(img_array, cnn_model)
    cam_path = save_gradcam(filepath, heatmap)

    # ================= RETURN =================
    return render_template(
        'index.html',
        cnn_result=f"{cnn_label} ({cnn_conf:.2f}%)",
        rf_result=rf_label,
        img_path=filepath,
        cam_path=cam_path
    )


# =========================
# 🚀 RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)