# Image recongnition Cat, Dog, and Wild Animal Image Classifier using CNN Deep Learning Model
This repository contains code for training a Convolutional Neural Network (CNN) deep learning model to classify images into three categories: cat, dog, or something wild. The model is trained on a dataset obtained from Kaggle, which consists of a large collection of images featuring various animals.

# About the Dataset
The dataset used in this project is sourced from Kaggle and contains images of cats, dogs, and various wild animals. The dataset is split into training and testing sets for model training and evaluation.

Kaggle Dataset: https://www.kaggle.com/datasets/andrewmvd/animal-faces
# Dependencies
To run the code in this repository and interact with the Streamlit app, you'll need the following Python libraries:

pandas
numpy
tensorflow
streamlit
Pillow
You can install these dependencies using pip:

 pip install pandas numpy tensorflow streamlit Pillow
# Usage
To train the CNN deep learning model and deploy a Streamlit app for users to interact with the trained model, follow these steps:

Clone this repository to your local machine.
Navigate to the repository directory.
Run the Streamlit app using the following command:

  streamlit run app.py
Open your web browser and go to the provided URL (usually http://localhost:8501) to access the user interface.
Upload an image containing either a cat, a dog, or a wild animal to the interface.
Click the 'Predict' button to see the model's classification for the uploaded image.
 Streamlit App
The Streamlit app (app.py) included in this repository provides a simple interface for users to upload images and receive predictions from the trained CNN deep learning mode

# Acknowledgements
The dataset used for training the CNN model is sourced from Kaggle. Citation details are provided in the dataset link above.
This project is inspired by the need to create an image classifier capable of distinguishing between images of cats, dogs, and wild animals, and providing a user-friendly interface for users to interact with the model.
# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
