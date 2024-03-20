"""
Author: Manshu Huang
Date: 2024 March 19

This Flask web application serves as a platform for neural network models for option pricing evaluation. Users can upload two CSV files containing financial data, preprocess the data, and evaluate the option pricing using a pre-trained convolutional neural network model.

The main functionalities include:
1. Upload and preprocess financial data.
2. View uploaded csv files.
3. View details of the uploaded csv files.
4. Evaluate option pricing and display evaluation results.

The application consists of the following routes:

1. '/' (GET, POST): Renders the index page where users can upload CSV files and evaluate option pricing. Upon successful upload and evaluation, it displays the evaluation results.
2. '/show_prediction' (GET): Renders the page to display the prediction results.
3. '/view_uploaded_data' (GET): Renders the page to view the uploaded csv files.
4. '/view_data' (GET): Renders the page to view the details of the uploaded csv files.

The application utilizes a convolutional neural network model for option pricing evaluation, preprocesses the uploaded data, and calculates evaluation metrics such as MSE loss, RMSE loss, and R2 score.

"""
# Importing Flask class and several functions from the flask module to handle web application tasks
from flask import Flask, render_template, url_for, request, flash, redirect

# Importing NumPy, a library for numerical operations, especially on arrays and matrices
import numpy as np
# Importing pandas, a library used for data manipulation and analysis
import pandas as pd

# Importing torch, a deep learning framework that provides a wide range of algorithms for machine learning
import torch
# Import secure_filename from werkzeug.utils to sanitize filenames to prevent security vulnerabilities or invalid file names during file uploads
from werkzeug.utils import secure_filename

# Importing nn module from torch for building neural network layers
from torch import nn
# Importing the functional module from torch, which contains functions used in building neural networks like activation functions
import torch.nn.functional as F

# Importing jsonify to convert complex data types to JSON (JavaScript Object Notation) before sending them to clients
from flask import jsonify

# Importing sqlite3 to provide a SQL interface to interact with SQLite databases
import sqlite3
# Importing math, a module that provides access to mathematical functions
import math

# Importing StandardScaler from sklearn.preprocessing to standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# Importing plotly.graph_objects for creating a wide variety of interactive plots and figures
import plotly.graph_objects as go

# Importing os, a module that provides a way of using operating system dependent functionality like reading or writing to a file system
import os

# Initialize a Flask application instance
app = Flask(__name__)
"""
This line creates an instance of the Flask class. 
The `__name__` parameter determines the root path of the application 
so that Flask can find resource files relative to the location of the application.
"""
# Set the upload folder in the app's configuration
app.config['UPLOAD_FOLDER'] = 'datab/'
# Configure the secret key for the application
app.secret_key = os.environ.get('SECRET_KEY') or 'fallback_key'
"""
Sets the secret key for the Flask application, which is used to maintain sessions and other 
security features. It tries to get the key from an environment variable `SECRET_KEY`; 
if not found, it defaults to 'fallback_key'. It's crucial for the key to be secret and 
complex in a production environment to ensure security.
"""
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
"""
Creates the upload directory specified in `app.config['UPLOAD_FOLDER']` if it doesn't already exist, 
using `os.makedirs()`. The `exist_ok=True` parameter means that `os.makedirs()` won't throw an error 
if the directory already exists. This ensures that the application has a place to store uploaded files.
"""

class Convolution1D(nn.Module):
    """
    1D Convolutional Neural Network (CNN) model for option pricing evaluation.

    The model architecture consists of convolutional layers, batch normalization layers,
    residual connections, dropout, global average pooling, and fully connected layers.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer with 64 output channels.
        bninit (nn.BatchNorm1d): Batch normalization for the initial convolutional layer.
        res1_conv1, res1_conv2 (nn.Conv1d): Convolutional layers in the first residual block.
        res1_bn1, res1_bn2 (nn.BatchNorm1d): Batch normalization layers in the first residual block.
        res2_conv1, res2_conv2 (nn.Conv1d): Convolutional layers in the second residual block.
        res2_bn1, res2_bn2 (nn.BatchNorm1d): Batch normalization layers in the second residual block.
        res2_shortcut (nn.Conv1d): Shortcut connection in the second residual block.
        dropout (nn.Dropout): Dropout layer for regularization.
        global_pool (nn.AdaptiveAvgPool1d): Global average pooling layer.
        fc1, fc2, fc3, fc4 (nn.Linear): Fully connected layers for final output.
    """
    def __init__(self):
        super(Convolution1D, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Batch normalization for the initial convolutional output
        self.bninit = nn.BatchNorm1d(64)
        
        # First residual block
        self.res1_conv1 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(64)
        self.res1_conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(64)
        
        # Second residual block with increased channel depth and stride for down-sampling
        self.res2_conv1 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.res2_bn1 = nn.BatchNorm1d(128)
        self.res2_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(128)
        # Shortcut connection to match the increased depth in the second residual block
        self.res2_shortcut = nn.Conv1d(64, 128, 1, stride=2)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        # Global pooling to reduce the spatial dimensions to a single scalar
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layers for the final prediction
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        # Final layer with one output, suitable for regression or binary classification
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        This method defines how the input tensor flows through the network. The network architecture
        includes convolutional layers with residual connections, batch normalization, dropout for
        regularization, and a global average pooling followed by fully connected layers to produce
        the final output.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
                          It represents the data that will be passed through the neural network.

        Returns:
        torch.Tensor: The output tensor of shape (batch_size, 1), which is the result of the model's
                      prediction for each item in the batch.
    """
    # Initial convolution layer followed by batch normalization and ReLU activation
        x = F.relu(self.bninit(self.conv1(x)))  # Apply ReLU after batch normalization of conv1's output

        # First residual block
        res1 = self.res1_conv1(x)  # First convolution in residual block
        res1 = F.relu(self.res1_bn1(res1))  # Apply ReLU after batch normalization
        res1 = self.res1_conv2(res1)  # Second convolution in residual block
        x = F.relu(x + res1) # Add the input to the block (residual connection) and apply ReLU
        
        # Second residual block with a shortcut connection
        res2 = self.res2_conv1(x) # Convolution with increased channels and down-sampling
        res2 = F.relu(self.res2_bn1(res2))   # Batch normalization and ReLU activation
        res2 = self.res2_conv2(res2)  # Second convolution in the block
        shortcut = self.res2_shortcut(x) # Shortcut connection to match the channel dimensions
        x = F.relu(res2 + shortcut) # Combine the shortcut and the residual block output

        # Global average pooling reduces each channel to a single scalar value
        x = self.global_pool(x)   # Apply global average pooling
        
        # Flatten the tensor to prepare it for the linear layer
        x = torch.flatten(x, 1)  # Flatten starting from the first dimension
        
        # Fully connected layers with dropout and ReLU activations to form the output
        x = self.dropout(x)  # Apply dropout for regularization
        x = F.relu(self.fc1(x)) # First fully connected layer with ReLU activation
        x = self.dropout(F.relu(self.fc2(x)))  # Apply dropout after second fully connected layer with ReLU
        x = self.dropout(F.relu(self.fc3(x)))  # Apply dropout after third fully connected layer with ReLU
        x = self.fc4(x)   # Final fully connected layer to produce the output

        return x

# Check for available GPU, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
This line checks if CUDA (a parallel computing platform and API by NVIDIA) is available for use.
If CUDA is available, it sets 'cuda' as the device, meaning PyTorch will use the GPU for computations.
Otherwise, it falls back to 'cpu', meaning computations will be done on the central processing unit (CPU).
"""
model_loaded = Convolution1D().to(device)
"""
This line creates an instance of the `Convolution1D` class (a 1D convolutional neural network model)
and moves it to the designated device (GPU or CPU). This is necessary to ensure that the model's 
tensors are stored on the same device to prevent errors during computation.
"""

# Load the saved model parameters into the model instance
model_loaded.load_state_dict(torch.load('model/best_model_ultimate.pt', map_location=device))
"""
Loads the model's state dictionary from the file 'best_model_ultimate.pt'. The `map_location` argument
ensures that the loaded state dict is moved to the appropriate device. This step initializes the model
with previously trained parameters, allowing for further training, evaluation, or inference without 
retraining from scratch.
"""

# Set the model to evaluation mode
model_loaded.eval()
"""
Switches the model to evaluation mode. This is an important step because it tells the model that it is
going to be used for inference, not training. In evaluation mode, certain operations like dropout and 
batch normalization will behave differently, ensuring consistent performance across different evaluations.
"""

def preprocess_data(file1_path, file2_path):
    """
    Preprocesses the uploaded CSV files containing financial data. The function reads two CSV files, 
    normalizes selected columns, combines them, and then splits the data into training, validation, 
    and test sets.

    Args:
        file1_path (str): Path to the first CSV file containing the features.
        file2_path (str): Path to the second CSV file containing the target variable.

    Returns:
        Tuple: Contains preprocessed data tensors for training, validation, and test sets, 
               which includes both features (X) and target variables (y).
    """
     # Load datasets from the given file paths
    ds = pd.read_csv(file1_path)  # Load the first CSV file as a pandas DataFrame
    target = pd.read_csv(file2_path)  # Load the second CSV file as a pandas DataFrame
    
    # Copy the dataset to avoid modifying the original data
    ds_new = ds.copy()
    
    # Initialize a StandardScaler to normalize the data
    scaler = StandardScaler()
    # List of columns to be normalized
    tbd = ['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]']
    # Apply scaling to the specified columns
    ds_new[tbd] = scaler.fit_transform(ds[tbd])
    
    # Concatenate the features and target dataframes along columns (axis=1)
    ds_new = pd.concat([ds_new, target], axis=1)

    # Select features for the model
    features = ds_new[['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]', '[STRIKE]', '[UNDERLYING_LAST]', '[C_DELTA]', '[C_GAMMA]',
                       '[C_VEGA]', '[C_THETA]', '[C_RHO]', '[C_IV]', '[C_VOLUME]', '[C_BID]', '[C_ASK]', '[P_DELTA]',
                       '[P_GAMMA]', '[P_VEGA]', '[P_THETA]', '[P_RHO]', '[P_IV]', '[P_VOLUME]', '[P_BID]', '[P_ASK]']].values
    # Select the target variable
    target_1 = ds_new['discounted_price'] 

    # Split the data into training (80%), validation (10%), and test (10%) sets
    X_train = features[:int(0.8 * len(features))]
    X_val = features[int(0.8 * len(features)):int(0.9 * len(features))]
    X_test = features[int(0.9 * len(features)):]
    
    y_train = target_1[:int(0.8 * len(target_1))]
    y_val = target_1[int(0.8 * len(target_1)):int(0.9 * len(target_1))]
    y_test = target_1[int(0.9 * len(target_1)):]

    # Convert the numpy arrays to PyTorch tensors and set the datatype to tensor
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_val = torch.from_numpy(X_val).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    
    y_train = torch.from_numpy(y_train.values).type(torch.Tensor)
    y_val = torch.from_numpy(y_val.values).type(torch.Tensor)
    y_test = torch.from_numpy(y_test.values).type(torch.Tensor)

    # Add a new dimension to the tensors to match the expected input shape for convolutional layers
    X_train = X_train.unsqueeze(2)
    X_val = X_val.unsqueeze(2)
    X_test = X_test.unsqueeze(2)

    return X_train, X_val, X_test, y_train, y_val, y_test

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    This function serves as the endpoint for the root URL ('/'). It handles both GET and POST requests,
    rendering the index page of the web application. During a POST request, it processes uploaded files
    for option pricing evaluation, performs predictions, and returns the results along with rendering
    the HTML template.

    Returns:
        render_template: Renders the HTML template based on the request method and the operations performed.
    """
    # Check if the request method is POST, which indicates that data has been submitted to the server
    if request.method == 'POST':
        # Check if both files are present in the request
        if 'file1' not in request.files or 'file2' not in request.files:
             # If either file is missing, flash a message to the user and reload the page
            flash('No file part')
            return redirect(request.url)
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Check if file names are not empty, meaning that the user has selected files
        if file1.filename == '' or file2.filename == '':
            # If no file is selected, flash a message and reload the page
            flash('No selected file')
            return redirect(request.url)
        
         # Check if both files exist and proceed with processing
        if file1 and file2:
            # Secure the file names and prepare the file paths
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            # Save the files to the server
            file1.save(file1_path)
            file2.save(file2_path)
            # Notify the user that files have been uploaded successfully
            flash('Files successfully uploaded')

            # Preprocess the data from the uploaded files
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file1_path, file2_path)

            # Evaluate the model on the uploaded data
            criterion = nn.MSELoss() # Mean Squared Error Loss function
            with torch.no_grad(): # No gradient computation for evaluation to save memory and computations
                output = model_loaded(X_test.to(device)).squeeze(-1) # Model prediction
                predictions = output
                test_loss = criterion(predictions, y_test.to(device)) # Calculate the test loss

             # Compute the Root Mean Square Error (RMSE) for the test data
            mse_loss = criterion(predictions, y_test)
            rmse_loss = mse_loss.item() ** (0.5)
            
            # Compute the R-squared (R2) score to measure the goodness of fit
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, predictions)
            # Create a plot of the predictions against the true values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(output)), y=output.squeeze().numpy(), mode='lines', name='Predicted',
                                      line=dict(color='red')))
            fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.numpy(), mode='lines', name='True',
                                      line=dict(color='blue')))
            fig.update_layout(title='Test Set Predictions', xaxis_title='Index', yaxis_title='Value')
            # Save the plot output to a file
            plot_output_path = os.path.join('static', 'Test_Set_Predictions.html')
            fig.write_html(plot_output_path)
            
            # Render the index HTML template with the results and plot path
            return render_template('index.html', test_loss=test_loss.item(), mse_loss=mse_loss.item(),
                                   rmse_loss=rmse_loss, r2_score=r2, plot=plot_output_path)
    # If the request method is GET, render the index HTML template without any prediction or plot
    return render_template('index.html', prediction=None, plot=None)

@app.route('/show_prediction', methods=['GET'])
def show_prediction():
    """
    Handles the GET request to display the prediction results on a dedicated page.

    This function checks if the prediction results file exists and then renders a page to display
    these results. If the results file does not exist, it redirects the user back to the index page.

    Returns:
        render_template: Renders the HTML template to show the prediction plot if the file exists.
        redirect: Redirects to the index page if the prediction results file does not exist.
    """
    
   # Define the path to the prediction results file, assumed to be in the 'static' directory
    plot_output_path = os.path.join('static', 'Test_Set_Predictions.html')
    
    # Check if the prediction results file exists
    if not os.path.exists(plot_output_path):
        # If the file does not exist, redirect the user to the index page
        return redirect(url_for('index'))

    # If the file exists, render the template to display the prediction results,
    # passing the path of the prediction plot file to the template
    return render_template('Test_Set_Predictions.html', plot_output_path=plot_output_path)

def get_table_info(file1_path, file2_path):
    """
    Extracts and returns the filenames from the full file paths.

    This function uses `os.path.basename` to strip the directory path from the full file paths, 
    returning only the filenames. It's useful for cases where only the file names are needed, 
    without their full paths.

    Args:
        file1_path (str): The full path to the first file.
        file2_path (str): The full path to the second file.

    Returns:
        list of str: A list containing just the filenames of the two files.
    """
    return [os.path.basename(file1_path), os.path.basename(file2_path)]

@app.route('/view_uploaded_data', methods=['GET'])
def view_uploaded_data():
    """
    Handles the GET request to display the uploaded database files.

    This endpoint fetches the list of files present in the upload directory and displays them
    on the 'view_uploaded_data.html' page. This allows users to see which files have been
    uploaded to the application.

    The function retrieves the filenames from the specified upload folder set in the app's configuration
    and passes these filenames to the rendering template.

    Returns:
        render_template: This function renders the 'view_uploaded_data.html' template, 
                         passing the list of uploaded files to it. This enables the webpage to
                         display the names of all files currently stored in the upload folder.
    """
    
    # Retrieve the list of files in the upload directory
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    # Render the HTML template, passing the list of files for display on the webpage
    return render_template('view_uploaded_data.html', files=files)

@app.route('/view_data', methods=['GET'])
def view_data():

    """
    Endpoint to view the details of the uploaded csv files.

    This function is triggered via a GET request and is responsible for rendering a page that
    displays the contents of the two uploaded CSV files. It expects 'filename1' and
    'filename2' as query parameters in the request.

    The function checks for the existence of the specified files in the upload directory.
    If both files exist, it reads them as CSVs and prepares the data for viewing. If either
    file is missing or an error occurs during file reading, the user is redirected to the
    file upload view with an appropriate error message.

    Args:
        filename1 (str): Query parameter specifying the first file to view.
        filename2 (str): Query parameter specifying the second file to view.

    Returns:
        render_template: Renders the 'view_data.html' template with the table data and filenames
                         if both files are found and readable.
        redirect: Redirects to the file upload view if any file is missing or an error occurs.
    """
    # Retrieve filenames from the request's query parameters
    filename1 = request.args.get('filename1')
    filename2 = request.args.get('filename2')
    
    # Validate that both filenames are provided
    if not filename1 or not filename2:
        flash('No data file selected.')
        return redirect(url_for('view_uploaded_data'))
    
    # Construct full file paths
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    # Check if both files exist in the specified upload folder
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        flash('Data file not found.')
        return redirect(url_for('view_uploaded_data'))

    try:
        # Attempt to read the files as CSVs and store their contents
        ds = pd.read_csv(file1_path)
        target = pd.read_csv(file2_path)
        table_data = {'Dataset 1': ds, 'Dataset 2': target}
    except Exception as e:
        # Handle any error that occurs during file reading and flash an error message
        flash(f'Error accessing CSV files: {e}')
        return redirect(url_for('view_uploaded_data'))
    # Render the view template with the loaded table data and filenames
    return render_template('view_data.html', tables=table_data, filename1=filename1, filename2=filename2)

# This conditional statement checks if the script is run as the main program.
# Ensure that code is only executed when the script is run directly, and not when imported as a module.
if __name__ == '__main__':
    # The app.run(debug=True) command starts the Flask application server.
    # The debug=True argument enables debug mode, which provides useful feedback in the browser for development, including detailed tracebacks and live reloading on code changes.
    app.run(debug=True)
