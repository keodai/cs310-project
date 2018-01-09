import os
from flask import Flask, jsonify, request, redirect, url_for, flash, render_template, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from functools import wraps, update_wrapper
from datetime import datetime

import recommender

UPLOAD_FOLDER = "/Volumes/expansion/project/data/uploads/"
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.secret_key = 'key'


@app.route('/')
def index():
    return render_template("index.html", current_song=None, recommendations=None, scroll=None, error=None)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             # return redirect(url_for('uploaded_file', filename=filename))
#             return redirect(request.url)
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <p><input type=file name=file>
#          <input type=submit value=Upload>
#     </form>
#     '''
# return 'Done'


@app.route('/recommend', methods=['GET', 'POST'])
def upload_file():
    # if request.method == 'POST':
    #     f = request.files['file']
    #     mode = request.form['mode']
    #     filename = secure_filename(f.filename)
    #     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     f.save(path)
    #     args = [path, mode]
    #     recommendations = recommender.recommend(args=args)
    #     return render_template("index.html", current_song=filename, recommendations=recommendations, scroll="app")

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("index.html", current_song=None, recommendations=None, scroll="app", error="No file part")
        f = request.files['file']
        mode = request.form['mode']
        if f.filename == '':
            return render_template("index.html", current_song=None, recommendations=None, scroll="app", error="No selected file")
        if f and mode and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            args = [path, mode]
            recommendations = recommender.recommend(args=args)
            return render_template("index.html", current_song=filename, recommendations=recommendations, scroll="app", error=None)
    return render_template("index.html", current_song=None, recommendations=None, scroll="app", error=None)


if __name__ == "__main__":
    app.run(debug=True)
