# Image Classification Web App using Flask & ResNet50

This project is a deep learning–based image classification web application built using Flask and a pre-trained ResNet50 model.

# Features
- Upload an image through a web interface
- Uses ResNet50 (pre-trained on ImageNet)
- Predicts the object present in the image
- Displays prediction label with confidence score

# Model Used
- **ResNet50**
- Pre-trained on **ImageNet dataset**
- No custom training required

# Tech Stack
- Python
- Flask
- TensorFlow / Keras
- NumPy
- HTML (Frontend)

# Workflow
1. User uploads an image
2. Image is resized to 224×224
3. Image is preprocessed for ResNet50
4. Model predicts the class
5. Result is shown with confidence percentage
