from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["DEBUG"] = True
ALLOWED_EXTENSIONS = {'csv','xlsx','xls'}
app.config["UPLOAD_FOLDER"] = "static\\uploads"
basedir = os.path.abspath(os.path.dirname(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/dataset', methods=['POST'])
def dataset():
    # get the uploaded file
    if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
    # if uploaded_file.filename != '':
    #     file_path = os.path.join(basedir, app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    #     # set the file path
    #     uploaded_file.save(file_path)
    #     # save the file
    # return redirect(url_for('/'))
    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
        file_url = app.config['UPLOAD_FOLDER'] + '/' + filename
        return render_template('dataset.html', file_url=file_url , filename=filename)
    return redirect(url_for('index'))
    


if __name__ == '__main__':
    app.run(debug=True)
    
    