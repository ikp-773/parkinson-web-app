import io
import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'nii.gz'}

num_slices = 5
class_labels = ['pd']
model_path = 'mri_model.h5'
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_images_from_zip(zip_path, folder_name):
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for i in range(1, 6):
            image_path = f'{folder_name}/mid_{i}_slice.png'
            with zip_file.open(image_path) as file:
                image = Image.open(file)
                image = image.resize((256, 256))
                image = image.convert('RGB')
                image = np.array(image) / 255.0
                images.append(image)
    images = np.array(images)
    return images

def test_model_from_zip(zip_path, folder_name):
    images = load_and_preprocess_images_from_zip(zip_path, folder_name)
    predictions = model.predict(images)
    conf_avg = np.mean(np.max(predictions, axis=1) * 100)
    return conf_avg

def process_mri(file):
    filename = file.filename
    if file and allowed_file(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = nib.load(file_path)
        img_data = img.get_fdata()

        output_folder_name = 'temp_slices'
        output_path = os.path.join(output_folder_name)
        os.makedirs(output_path, exist_ok=True)

        mid_slice = img_data.shape[2] // 2

        for i in range(-(num_slices//2), (num_slices//2) if (num_slices % 2 == 0) else (num_slices//2)+1):
            slice_data = np.squeeze(img_data[:, :, mid_slice+i])
            slice_data = (slice_data - np.min(slice_data)) / \
                (np.max(slice_data) - np.min(slice_data)) * 255
            slice_data = slice_data.astype(np.uint8)

            output_filename = os.path.join(
                output_path, 'mid_'+str(i+(num_slices//2)+1)+'_slice.png')
            plt.imsave(output_filename, slice_data, cmap="gray")

        zipf = zipfile.ZipFile('temp_slices.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('temp_slices', zipf)
        zipf.close()

        folder_name = 'temp_slices'
        folder_name = folder_name[:21]

        with open('temp_slices.zip', 'rb') as f:
            zipf_bytes = io.BytesIO(f.read())

        result_str = test_model_from_zip(zipf_bytes, folder_name)
        confidence = f"{result_str:.2f}%"
        if result_str > 50:
            result_label = 'The person may have Parkinson\'s disease.'
        else:
            result_label = 'The person may not have Parkinson\'s disease.'

        return confidence, result_label
    else:
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    confidence, result_label = process_mri(file)
    if confidence is not None and result_label is not None:
        return render_template('result.html', confidence=confidence, result_label=result_label)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()