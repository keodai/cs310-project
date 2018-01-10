import os
from flask import Flask, request, url_for, render_template, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

import recommender

UPLOAD_FOLDER = "/Volumes/expansion/project/data/uploads/"
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
previous_upload_path = None
previous_filename = None

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.secret_key = 'key'


@app.route('/')
def index():
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll=None, error=None, warning=None)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/recommend', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global previous_upload_path
        global previous_filename
        use_previous_path = False
        # check if the post request has the file part
        if 'file' not in request.files:
            use_previous_path = True
        if use_previous_path:
            if previous_upload_path is None or previous_filename is None:
                return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="No file part", warning=None)
        else:
            f = request.files['file']
            if f.filename == '':
                use_previous_path = True
                if previous_upload_path is None or previous_filename is None:
                    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="No selected file", warning=None)
        mode = request.form['mode']
        if use_previous_path:
            args = [previous_upload_path, mode]
            recommendations, predicted, warning = recommender.recommend(args=args)
            return render_template("index.html", current_song=previous_filename, recommendations=recommendations, predicted=predicted, scroll="app", error=None, warning=warning)
        elif f and mode and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            args = [path, mode]
            previous_upload_path = path
            previous_filename = filename
            recommendations, predicted, warning = recommender.recommend(args=args)
            return render_template("index.html", current_song=filename, recommendations=recommendations, predicted=predicted, scroll="app", error=None, warning=warning)
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error=None, warning=None)


if __name__ == "__main__":
    app.run(debug=True)
