import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved GRU model
model = load_model('model.h5')

# Define a function to preprocess the input data for the model
def preprocess_data(data):
    # Apply the same normalization and differencing transformations as you did in your project
    # For example, you can use a MinMaxScaler from scikit-learn to normalize the data between 0 and 1
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    # Then, you can use the same differencing method you used in your project to make the data stationary
    data_diff = np.diff(data_norm, axis=0)
    return data_diff.reshape(1, -1, 1)

# Define a function to make predictions using the loaded model
def make_predictions(model, input_data):
    # Preprocess the input data using the preprocess_data function
    input_data = preprocess_data(input_data)
    # Use the predict method of the loaded model to make predictions
    predictions = model.predict(input_data)
    # Postprocess the predictions as necessary
    # For example, you can use the inverse_transform method of the MinMaxScaler to denormalize the predictions
    scaler = MinMaxScaler()
    scaler.fit_transform(input_data[0])
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Define the Streamlit app
def app():
    # Add a title and subtitle
    st.title("Traffic Prediction using GRU Neural Network")
    st.markdown("This app uses a GRU Neural Network to predict traffic on four junctions.")

    # Load the test data and plot the true values
    test_data = pd.read_csv("test_data.csv", index_col=0)
    fig, ax = plt.subplots()
    sns.lineplot(data=test_data)
    ax.set_title("True Values")
    ax.set_xlabel("DateTime")
    ax.set_ylabel("Number of Vehicles")
    st.pyplot(fig)

    # Add a section to allow the user to input new data
    st.header("Enter New Data")
    input_data = st.text_input("Enter new data as a comma-separated list of numbers")
    if input_data:
        # Convert the input data to a numpy array
        input_data = np.array(input_data.split(","), dtype=float)
        # Make predictions using the loaded model and the input data
        predictions = make_predictions(model, input_data)
        # Display the predictions using a line chart
        fig, ax = plt.subplots()
        sns.lineplot(data=predictions.reshape(-1))
        ax.set_title("Predictions")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Number of Vehicles")
        st.pyplot(fig)
        # Display a message to the user indicating the predicted traffic volume
        st.write(f"The predicted traffic volume is {int(predictions[-1][0])} vehicles.")

# Run the Streamlit app
if __name__ == '__main__':
    app()
