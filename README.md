# Iris Flower Type Prediction API Project

This project is a simple API that uses a Logistic Regression model to predict the type of iris flower based on its features. The API is built with FastAPI and utilizes the **Iris** dataset.

## Project Features
- Uses a Logistic Regression model to classify iris flower types
- Implements an API to accept inputs and return predictions
- Includes documentation in Swagger UI for easy API usage

## Prerequisites
- Python 3.7 or higher
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas

## Installation and Setup

### 1. Clone the Project
First, clone the repository:
```bash

git clone https://github.com/Yasaman-Sattari/Iris-Prediction-API.git

2. Install Dependencies
Install all dependencies using pip:
pip install -r requirements.txt

3. Train the Model
Before running the API, train and save the model:
python train_model.py

4. Run the API
To start the API, use the following command:
uvicorn main:app --reload
Your API will be available at http://127.0.0.1:8000.

5. Using the API
Navigate to Swagger UI to view the documentation and usage instructions:

Swagger UI
Sample Request to the API
To predict the flower type, you can send a POST request to /predict:

Endpoint: /predict

Method: POST

Body:
Copy code
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Response:
Copy code
{
  "prediction": 0
}


The output prediction returns one of the following values:

0: Setosa
1: Versicolor
2: Virginica


Project Structure
train_model.py : Contains the code for loading, processing, and training the machine learning model.
main.py : Implements the API using FastAPI and provides endpoints for predictions.
model.pkl : Saved model file created after training.
