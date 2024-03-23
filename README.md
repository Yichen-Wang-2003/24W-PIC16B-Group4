
# 24W-PIC16B-Group4
Members: Yichen Wang, Xipeng Du, Manshu Huang 
(ALL activities done by user name called 'Ning Sam' refer to the student named Yichen Wang)

Description:
Option is a financial contract that give the buyer the right to buy or sell certain quantity of assets as specific strike price on or before maturity date, and the buyer needs to pay premium or “option price” in this context to the seller of this contract. Option pricing is a way to evaluate the fair value of an option which corresponds to its striking price, maturity time and risk involved with the stock.

In this project, we aim to use one-dimension convolutional neural network to predict the intrinsic value of the call and put options with regard to its final payoff from the contract. Without loss of generality, we used the S&P 500 index as the stock of choice.

The whole project contains three parts: Data preprocessing, CNN model construction and result analysis, and Flask Web Application creation. We first utilize the API and Yahoo Finance to get rudimentary data, do systematic data preprocessing, and then construct the 1D CNN model to predict the option price. Finally, we create a web application to visualize the result using Flask.

For data preprocessing part, the data source is [https://www.optionsdx.com/product/spy-option-chains/] and yfinance api for python. For CNN construction and training, please refer to the blog post for more details. The flask web app instructions are kind of complex, so the procedures are attached below.

# Flask Web App Instructions

This guide will help you set up and run the Flask web application for our project. Follow the steps below to get started.

## Step 0: Download the Application Folder

1. Download the entire "flask.op.PerfectFINALver" folder, which contains all the necessary files for the Flask application.

2. The folder includes the following key components:
   - `app.py`: The main application file that contains crucial functions such as `Convolution1D`, `preprocess_data`, `Index`, `show_prediction`, `get_table_info`, `view_uploaded_database`, and `view_database`. These functions are essential for the app's operation.
   - `templates` folder: This directory contains all the HTML files required for the app's user interface, enabling interactions and data display. It includes `index.html`, `plot.html`, `view_uploaded_data.html`, `show_prediction.html`, `Test_Set_Predictions.html`, and `view_data.html`.
   - `data` folder: This folder contains sample CSV files, `df_2023_h1_feature.csv` and `df_2023_h1_target.csv`, which you can use for data upload and result demonstration in the app.
   - `model` folder: This folder stores the pre-trained model file necessary for making data predictions.
   - `__pycache__` folder: A system-generated directory that caches bytecode, enhancing the program's execution speed.
   - `static` folder: This folder houses static files that are vital for the app's styling and interactive features.

## Step 1: Install Flask

1. Ensure that you have Flask installed on your system. If you haven't installed it yet, refer to the official Flask installation guide: [Flask Installation](https://flask.palletsprojects.com/en/3.0.x/installation/)

## Step 2: Run the Application

1. Open a terminal or command prompt and navigate to the "flask.op.PerfectFINALver" folder.

2. Run the following command to start the Flask application:
   ```
   python app.py
   ```

3. Upon successful execution, you will see a link similar to `http://127.0.0.1:5000`. Copy and paste this link into your web browser to access the welcome page of the application.

## Step 3: Using the Application

1. On the welcome page, you will find `"Browse" buttons` that allow you to quickly upload local CSV files.

2. After selecting your CSV files, click the `"Upload & Save" button`. The application will efficiently process the data and generate evaluation results within a few seconds.

3. To view your uploaded dataset (CSV files) online, click on the `"View Data"` option. This will display the contents of your uploaded files.

4. For a comprehensive analysis, click on the `"View Prediction"` feature. This will present interactive prediction graphs, enabling you to interpret and analyze data trends effectively.

That's it! You are now ready to use the Flask web application to upload, view, and analyze your data. If you encounter any issues or have further questions, please refer to the documentation or reach out to the project team for assistance.
