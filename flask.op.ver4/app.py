from flask import Flask, render_template, url_for,request, flash
import numpy as np
import pandas as pd
import torch
from flask import redirect
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
# Check if there is a GPU available, otherwise use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_loaded = Convolution1D()
# model_loaded.load_state_dict(torch.load('model/best_model.pth'))
model_loaded = Convolution1D().to(device)
model_loaded.load_state_dict(torch.load('model/best_model_final.pt', map_location=device))
model_loaded.eval()
# Load the model and move it to the device
# model_loaded = Convolution1D().to(device)
# model_loaded.load_state_dict(torch.load('saved_models/best_model_final.pt', map_location=device))
def preprocess_data(data):
    conn = sqlite3.connect(data)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    ds = pd.read_sql_query("SELECT * from df_2023_h1_feature", conn)
    target = pd.read_sql_query("SELECT * from df_2023_h1_target", conn)

    ds_new = ds.copy()
    scaler = StandardScaler()
    tbd = ['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]']
    ds_new[tbd] = scaler.fit_transform(ds[tbd])

    ds_new = pd.concat([ds_new, target], axis=1)

    ds_new['-rt'] = -0.04*(ds['[EXPIRE_UNIX]'] - ds['[QUOTE_UNIXTIME]'])/(3600*365*24)
    ds_new['price_diff'] = ds_new['[STRIKE]'] - ds_new['discounted_price']
    ds_new['-rt'] = pd.to_numeric(ds_new['-rt'])
    ds_new['exp(-rt)'] = ds_new['-rt'].apply(lambda x: math.exp(x))

    ds_new = ds_new.loc[:, ~ds_new.columns.str.contains('^Unnamed')]
    ds = ds.loc[:, ~ds.columns.str.contains('^Unnamed')]

    features = ds_new[['[QUOTE_UNIXTIME]', '[EXPIRE_UNIX]', '[STRIKE]', '[UNDERLYING_LAST]', '[C_DELTA]', '[C_GAMMA]', '[C_VEGA]',
           '[C_THETA]', '[C_RHO]', '[C_IV]', '[C_VOLUME]','[C_BID]', '[C_ASK]', '[P_DELTA]', '[P_GAMMA]', '[P_VEGA]', '[P_THETA]',
           '[P_RHO]', '[P_IV]', '[P_VOLUME]', '[P_BID]', '[P_ASK]']].values

    target_1= ds_new['price_diff']*ds_new['exp(-rt)']

    X_train = features[:int(0.8*len(features))]
    X_val = features[int(0.8*len(features)):int(0.9*len(features))]
    X_test = features[int(0.9*len(features)):]
    y_train = target_1[:int(0.8*len(target_1))]
    y_val = target_1[int(0.8*len(target_1)):int(0.9*len(target_1))]
    y_test = target_1[int(0.9*len(target_1)):]

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

# @app.route('/predict', methods=['POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded')

            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path)

            criterion = nn.MSELoss()
            with torch.no_grad():
                # output = model_loaded(X_test)
                output = model_loaded(X_test.to(device)).squeeze(-1)
                predictions=output
                test_loss = criterion(predictions, y_test.to(device))
                # predictions = output.squeeze(-1)
                # test_loss = criterion(predictions, y_test)
            criterion = nn.MSELoss()
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

            # plot_output_path = "static/Test_Set_Predictions.html"
            # fig.write_html(plot_output_path)
            plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
            fig.write_html(plot_output_path)
            plot_output_path = "./static/Test_Set_Predictions.html"

            return render_template('index.html', test_loss=test_loss.item(), mse_loss=mse_loss.item(), rmse_loss=rmse_loss, r2_score=r2, plot=plot_output_path)
    return render_template('index.html', prediction=None, plot=None)

@app.route('/show_prediction', methods=['GET'])
def show_prediction():
    plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
    if not os.path.exists(plot_output_path):
        return redirect(url_for('index'))

    return render_template('Test_Set_Predictions.html', plot_output_path=plot_output_path)

def get_table_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    table_names = [table[0] for table in tables]
    return table_names

@app.route('/view_uploaded_database', methods=['GET'])
def view_uploaded_database():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('view_uploaded_database.html', files=files)

@app.route('/view_database', methods=['GET'])
def view_database():
    filename = request.args.get('filename')
    if not filename:
        flash('No database file selected.')
        return redirect(url_for('view_uploaded_database'))

    db_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(db_path):
        flash('Database file not found.')
        return redirect(url_for('view_uploaded_database'))

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_data = {}
        for table in tables:
            table_name = table[0]
            table_data[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
        conn.close()
        return render_template('view_database.html', tables=table_data, filename=filename)
    except sqlite3.Error as e:
        flash(f'Error accessing database: {e}')
        return redirect(url_for('view_uploaded_database'))

@app.route('/get_table_data', methods=['GET'])
def get_table_data():
    table_name = request.args.get('table_name')
    if not table_name:
        return jsonify({'error': 'Table name not provided'})

    db_path = request.args.get('db_path')
    if not db_path:
        return jsonify({'error': 'Database file path not provided'})

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name};")
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        conn.close()
        return jsonify({'columns': columns, 'data': data})
    except sqlite3.Error as e:
        return jsonify({'error': str(e)})

@app.route('/plot', methods=['POST'])
def plot():
    table_name = request.form['table']
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with sqlite3.connect(file_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)

    fig = go.Figure()
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
    fig.write_html(plot_path)

    return render_template('plot.html', plot_path=plot_path)

@app.route('/evaluate_and_visualize', methods=['POST'])
def evaluate_and_visualize():
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path)

    criterion = nn.MSELoss()
    with torch.no_grad():
        output = model_loaded(X_test)
        predictions = output.squeeze(-1)
        test_loss = criterion(predictions, y_test)

    # Calculate RMSE Loss
    mse_loss = criterion(predictions, y_test)
    rmse_loss = mse_loss.item()**(0.5)

    # Calculate R2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)

    # Creating Chart Objects
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(predictions)), y=predictions.numpy(), mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test.numpy(), mode='lines', name='True',
                             line=dict(color='blue')))
    fig.update_layout(title='Test Set Predictions', xaxis_title='Index', yaxis_title='Value')

    # Save the chart as an HTML file
    plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_Set_Predictions.html')
    fig.write_html(plot_output_path)

    # Return to evaluation results & charts'html pages
    return render_template('evaluate_and_visualize.html', test_loss=test_loss.item(), mse_loss=mse_loss.item(),
                           rmse_loss=rmse_loss, r2_score=r2, plot=plot_output_path)



if __name__ == '__main__':
    app.run(debug=True)
