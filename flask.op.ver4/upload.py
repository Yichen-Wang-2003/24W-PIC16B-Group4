from flask import Flask, render_template, request

app = Flask(__name__)

# Upload files
@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('pb')
    fname = f.filename
    with open(f'./data/{fname}', 'wb') as tf:
        tf.write(f.read())
    return f'Upload Succeed, File Name: {fname}'



if __name__ == '__main__':
    app.run(debug=True)
