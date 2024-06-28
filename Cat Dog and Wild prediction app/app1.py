import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page configuration
st.set_page_config(page_title="Image Recognition", layout="wide")

# Load the trained model
# @st.cache(allow_output_mutation=True)
def load_model(model_path):
    return tf.keras.models.load_model("train_model.h5")

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((255, 255))  # Resize the image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Get the predicted class
def get_predicted_class(prediction, class_names):
    predicted_index = np.argmax(prediction)
    return class_names[predicted_index]

# Main function to run the app
def main():
    # Sidebar for uploading images
    st.sidebar.title("Upload Image")
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Load the trained model
    model = load_model('new_model.h5')  # Replace 'new_model.h5' with the path to your trained model

    # Define class names for the prediction
    class_names = ['Cat', 'Dog', 'Wild']

    # Main content
    st.title("Image Recognition: Cat, Dog, or Wild Animal?")
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    # Display the uploaded image and prediction
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1.image(image, caption="Uploaded Image")

        # Predict button
        predict_button = col1.button("Predict")

        # Perform prediction
        if predict_button:
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Predict the class
            prediction = model.predict(processed_image)

            # Display the prediction result
            predicted_class = get_predicted_class(prediction, class_names)
            col1.success(f"The uploaded image is predicted to be: {predicted_class}")

# Run the app
if __name__ == "__main__":
    main()
