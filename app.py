import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify , send_file
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from flask import send_file
import joblib
from flask_cors import CORS
app = Flask(__name__)

CORS(app)

# Load data and train models
path_no_tumor = 'no_tumor'
path_pituitary_tumor = 'pituitary_tumor'
path_meningioma_tumor = 'meningioma_tumor'
path_glioma_tumor = 'glioma_tumor'
tumor_check = {'no_tumor': 0, 'pituitary_tumor': 1, 'meningioma_tumor': 2, 'glioma_tumor': 3}

# x = []
# y = []
# for cls in tumor_check:
#     path = globals()[f'path_{cls}']
#     for j in os.listdir(path):
#         image = cv2.imread(os.path.join(path, j), 0)
#         image = cv2.resize(image, (200, 200))
#         x.append(image)
#         y.append(tumor_check[cls])

# x = np.array(x)
# y = np.array(y)

# x_update = x.reshape(len(x), -1)
# x_train, x_test, y_train, y_test = train_test_split(x_update, y, random_state=10, test_size=0.3)
# x_train = x_train / 255
# x_test = x_test / 255

# pca = PCA(.98)
# pca_train = pca.fit_transform(x_train)
# pca_test = pca.transform(x_test)

# logistic = LogisticRegression(C=0.1, max_iter=500)
# logistic.fit(pca_train, y_train)

# sv = SVC()
# sv.fit(pca_train, y_train)


logistic = joblib.load('logistic_model.pkl')
sv = joblib.load('svc_model.pkl')
pca = joblib.load('pca_model.pkl')



@app.route("/")
def home():
    return render_template('home.html')

@app.route("/execute_python_function", methods=["POST"])
def execute_python_function():
    # Get the uploaded file
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (200, 200))
    img_flattened = img_resized.reshape(1, -1) / 255

    # Predict tumor type
    pca_transformed = pca.transform(img_flattened)
    tumor_type = sv.predict(pca_transformed)[0]

    # # Create a highlighted image
    # highlighted_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    # if tumor_type != 0:  # Assuming non-zero means there's a tumor
    #     # Dummy highlight: Draw a rectangle (adjust this to your actual detection logic)
    #     cv2.rectangle(highlighted_img, (0, 0), (0, 0), (0, 0, 0), 0)  # Red box

    # # Save the highlighted image
    # highlighted_image_path = 'static/highlighted_image.jpg'
    # cv2.imwrite(highlighted_image_path, highlighted_img)

    # Return response
    dec = {0: 'no_tumor', 1: 'pituitary_tumor', 2: 'meningioma_tumor', 3: 'glioma_tumor'}
    tumor_label = dec[tumor_type]
    return jsonify({'tumor_type': tumor_label})

# def generate_report():
#     # Get form data (for example: tumor type and image path)
#     tumor_type = request.form.get('tumor_type')
#     highlighted_image_url = request.form.get('highlighted_image_url')

#     # Create a PDF file
#     pdf_file_path = "static/tumor_report.pdf"
#     c = canvas.Canvas(pdf_file_path, pagesize=letter)
#     width, height = letter

#     # Add text to the PDF
#     c.setFont("Helvetica", 12)
#     c.drawString(100, height - 100, f"Tumor Detection Report")
#     c.drawString(100, height - 120, f"Tumor Type: {tumor_type}")

#     # Add the highlighted tumor image to the PDF
#     if os.path.exists(highlighted_image_url):
#         c.drawImage(highlighted_image_url, 100, height - 400, width=300, height=200)

#     # Save the PDF
#     c.save()

#     # Send the PDF as a downloadable file
#     return send_file(pdf_file_path, as_attachment=True)

@app.route("/details")
def details():
    return render_template("details.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
