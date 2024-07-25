# Document Classifier

Short description of the project.

## Overview

This project aims to classify text documents into different categories using a machine learning model trained on datasets collected from various sources. The classification model is built using the creme library, which allows for online/incremental learning, and the final model is serialized into a pickle file for later use.

## Files

- **test.ipynb**: Jupyter Notebook file for testing the accuracy of the model on a test dataset.
- **auto-retrain.ipynb**: Jupyter Notebook file for training the model and serializing it into a pickle file.
- **README.md**: This file containing project overview and instructions.

## Usage

1. **Testing the Model**:
   - Open `test.ipynb` in Jupyter Notebook.
   - Run the notebook to evaluate the accuracy of the model on a test dataset.

2. **Training and Serializing the Model**:
   - Open `auto-retrain.ipynb` in Jupyter Notebook.
   - Run the notebook to train the model on the provided datasets and serialize it into a pickle file.

## Requirements

- Python 3.x
- Jupyter Notebook
- PyPDF2
- creme
- scikit-learn
- pandas
- matplotlib
- seaborn

## Usage Example

Here's an example of how to use the trained model:

```python
import pickle

# Load the model from the file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the transformation function from the file
with open('transform-doc.pkl', 'rb') as f:
    loaded_transform_text = pickle.load(f)

# Example text document
document_text = "Sample text document to classify."

# Transform the text using the loaded transformation function
transformed_text = loaded_transform_text(document_text)

# Predict the category using the loaded model
predicted_category = loaded_model.predict_one(transformed_text)

print("Predicted category:", predicted_category)
