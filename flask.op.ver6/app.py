"""
This Flask web application serves as a platform for neural network models for option pricing evaluation. Users can upload two CSV files containing financial data, preprocess the data, and evaluate the option pricing using a pre-trained convolutional neural network model.

The main functionalities include:
1. Upload and preprocess financial data.
2. View uploaded database files.
3. View details of the uploaded database files.
4. Evaluate option pricing and display evaluation results.

The application consists of the following routes:

1. '/' (GET, POST): Renders the index page where users can upload CSV files and evaluate option pricing. Upon successful upload and evaluation, it displays the evaluation results.
2. '/show_prediction' (GET): Renders the page to display the prediction results.
3. '/view_uploaded_database' (GET): Renders the page to view the uploaded database files.
4. '/view_database' (GET): Renders the page to view the details of the uploaded database files.

The application utilizes a convolutional neural network model for option pricing evaluation, preprocesses the uploaded data, and calculates evaluation metrics such as MSE loss, RMSE loss, and R2 score.

Author: Manshu Huang
Date: 2024 March 17
"""

from flask import Flask, render_template, url_for, request, flash, redirect
import numpy as np
import pandas as pd
import torch
from werkzeug.utils import secure_filename
from torch import nn
import torch.nn.functional as F
from flask import jsonify
import sqlite3
import math
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datab/'
app.secret_key = os.environ.get('SECRET_KEY') or 'fallback_key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Convolution1D(nn.Module):
    def __init__(self):
        super(Convolution1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bninit = nn.BatchNorm1d(64)

        self.res1_conv1 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(64)
        self.res1_conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(64)

        self.res2_conv1 = nn.Conv1d(64, 128, 3, padding=1, stride=2)
        self.res2_bn1 = nn.BatchNorm1d(128)
        self.res2_conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(128)
        self.res2_shortcut = nn.Conv1d(64, 128, 1, stride=2)

        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bninit(self.conv1(x)))

        res1 = self.res1_conv1(x)
        res1 = F.relu(self.res1_bn1(res1))
        res1 = self.res1_conv2(res1)
        x = F.relu(x + res1)

        res2 = self.res2_conv1(x)
        res2 = F.relu(self.res2_bn1(res2))
        res2 = self.res2_conv2(res2)
        shortcut = self.res2_shortcut(x)
        x = F.relu(res2 + shortcut)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

# Check for available GPU, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loaded = Convolution1D().to(device)
model_loaded.load_state_dict(torch.load('model/best_model_new.pt', map_location=device))
model_loaded.eval()

def preprocess_data(file1_path, file2_path):
    """
    Preprocesses the uploaded CSV files containing financial data.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.

    Returns:
        Tuple: Contains preprocessed data including training, validation, and test sets.
    """
    # Read two CSV files
    ds = pd.read_csv(file1_path)
    target = pd.read_csv(file2_path)
    ds_new = ds.copy()
    scaler = StandardScaler()
    tbd = ['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]']
    ds_new[tbd] = scaler.fit_transform(ds[tbd])
    ds_new = pd.concat([ds_new, target], axis=1)
    ds_new['-rt'] = -0.04 * (ds['[EXPIRE_UNIX]'] - ds['[QUOTE_UNIXTIME]']) / (3600 * 365 * 24)
    ds_new['price_diff'] = ds_new['[STRIKE]'] - ds_new['discounted_price']
    ds_new['-rt'] = pd.to_numeric(ds_new['-rt'])
    ds_new['exp(-rt)'] = ds_new['-rt'].apply(lambda x: math.exp(x))

    ds_new = ds_new.loc[:, ~ds_new.columns.str.contains('^Unnamed')]
    ds = ds.loc[:, ~ds.columns.str.contains('^Unnamed')]

    features = ds_new[['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]', '[STRIKE]', '[UNDERLYING_LAST]', '[C_DELTA]', '[C_GAMMA]',
                       '[C_VEGA]', '[C_THETA]', '[C_RHO]', '[C_IV]', '[C_VOLUME]', '[C_BID]', '[C_ASK]', '[P_DELTA]',
                       '[P_GAMMA]', '[P_VEGA]', '[P_THETA]', '[P_RHO]', '[P_IV]', '[P_VOLUME]', '[P_BID]', '[P_ASK]']].values

    target_1 = ds_new['price_diff'] * ds_new['exp(-rt)']

    X_train = features[:int(0.8 * len(features))]
    X_val = features[int(0.8 * len(features)):int(0.9 * len(features))]
    X_test = features[int(0.9 * len(features)):]
    y_train = target_1[:int(0.8 * len(target_1))]
    y_val = target_1[int(0.8 * len(target_1)):int(0.9 * len(target_1))]
    y_test = target_1[int(0.9 * len(target_1)):]

    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_val = torch.from_numpy(X_val).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train.values).type(torch.Tensor)
    y_val = torch.from_numpy(y_val.values).type(torch.Tensor)
    y_test = torch.from_numpy(y_test.values).type(torch.Tensor)

    X_train = X_train.unsqueeze(2)
    X_val = X_val.unsqueeze(2)
    X_test = X_test.unsqueeze(2)

    return X_train, X_val, X_test, y_train, y_val, y_test

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the index page and handles file upload and option pricing evaluation.

    Returns:
        render_template: Renders the HTML template.
    """
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file1 = request.files['file1']
        file2 = request.files['file2']
        if file1.filename == '' or file2.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file1 and file2:
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file1.save(file1_path)
            file2.save(file2_path)
            flash('Files successfully uploaded')

            # Call preprocess_data function to process the data
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file1_path, file2_path)

            criterion = nn.MSELoss()
            with torch.no_grad():
                output = model_loaded(X_test.to(device)).squeeze(-1)
                predictions = output
                test_loss = criterion(predictions, y_test.to(device))

            # Calculate RMSE Loss
            mse_loss = criterion(predictions, y_test)
            rmse_loss = mse_loss.item() ** (0.5)
            # Calculate R2 Score
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, predictions)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(output)), y=output.squeeze().numpy(), mode='lines', name='Predicted',
                                      line=dict(color='red')))
            fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.numpy(), mode='lines', name='True',
                                      line=dict(color='blue')))
            fig.update_layout(title='Test Set Predictions', xaxis_title='Index', yaxis_title='Value')

            # plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
            plot_output_path = os.path.join('static', 'Test_Set_Predictions.html')
            fig.write_html(plot_output_path)

            return render_template('index.html', test_loss=test_loss.item(), mse_loss=mse_loss.item(),
                                   rmse_loss=rmse_loss, r2_score=r2, plot=plot_output_path)
    return render_template('index.html', prediction=None, plot=None)

@app.route('/show_prediction', methods=['GET'])
def show_prediction():
    """
    Renders the page to display the prediction results.

    Returns:
        render_template: Renders the HTML template.
    """
    # plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
    plot_output_path = os.path.join('static', 'Test_Set_Predictions.html')
    if not os.path.exists(plot_output_path):
        return redirect(url_for('index'))

    return render_template('Test_Set_Predictions.html', plot_output_path=plot_output_path)

def get_table_info(file1_path, file2_path):
    """
    Retrieves the filenames of the uploaded database files.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.

    Returns:
        List: Contains the filenames of the uploaded database files.
    """
    return [os.path.basename(file1_path), os.path.basename(file2_path)]

@app.route('/view_uploaded_database', methods=['GET'])
def view_uploaded_database():
    """
    Renders the page to view the uploaded database files.

    Returns:
        render_template: Renders the HTML template.
    """
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('view_uploaded_database.html', files=files)

@app.route('/view_database', methods=['GET'])
def view_database():
    """
    Renders the page to view the details of the uploaded database files.

    Returns:
        render_template: Renders the HTML template.
    """
    filename1 = request.args.get('filename1')
    filename2 = request.args.get('filename2')
    if not filename1 or not filename2:
        flash('No database file selected.')
        return redirect(url_for('view_uploaded_database'))

    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        flash('Database file not found.')
        return redirect(url_for('view_uploaded_database'))

    try:
        ds = pd.read_csv(file1_path)
        target = pd.read_csv(file2_path)
        table_data = {'Dataset 1': ds, 'Dataset 2': target}
    except Exception as e:
        flash(f'Error accessing CSV files: {e}')
        return redirect(url_for('view_uploaded_database'))

    return render_template('view_database.html', tables=table_data, filename1=filename1, filename2=filename2)

if __name__ == '__main__':
    app.run(debug=True)
