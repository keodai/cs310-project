import os
from flask import Flask, request, url_for, render_template, make_response, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from datetime import datetime
import errno

import paths
import recommender
import sound_recording

UPLOAD_FOLDER = paths.upload_folder
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
previous_upload_dir = None
previous_filenames = None

app = Flask(__name__, static_url_path='')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.secret_key = 'key'


def make_string(list):
    return ", ".join(list)


@app.route('/')
def index():
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll=None, error=None, warning=None)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/recommend', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global previous_upload_dir
        global previous_filenames
        use_previous_path = False
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        directory = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

        if request.form['input-method'] == 'mic':
            d, f = os.path.split(sound_recording.record(directory))
            filenames = [f]
            mode = request.form['mode']
            vector_type = request.form['features']
            args = [directory, mode, vector_type]
            previous_upload_dir = directory
            previous_filenames = filenames
            recommendations, predictions, warning = recommender.recommend(args=args)
            return render_template("index.html", current_song=filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
        elif request.form['input-method'] == 'file':
            # check if the post request has the file part
            if 'file[]' not in request.files:
                use_previous_path = True
            if use_previous_path:
                if previous_upload_dir is None or previous_filenames is None:
                    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="No file part", warning=None)
            else:
                file_list = request.files.getlist("file[]")
                for f in file_list:
                    if f.filename == '':
                        use_previous_path = True
                        if previous_upload_dir is None or previous_filenames is None:
                            return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="No selected file", warning=None)
                mode = request.form['mode']
                vector_type = request.form['features']
                if use_previous_path:
                    args = [previous_upload_dir, mode, vector_type]
                    recommendations, predictions, warning = recommender.recommend(args=args)
                    return render_template("index.html", current_song=previous_filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
                elif mode and all(allowed_file(f.filename) for f in file_list) and all(f for f in file_list):
                    filenames = []
                    for f in file_list:
                        filename = secure_filename(f.filename)
                        path = os.path.join(directory, filename)
                        f.save(path)
                        filenames.append(filename)
                    args = [directory, mode, vector_type]
                    previous_upload_dir = directory
                    previous_filenames = filenames
                    recommendations, predictions, warning = recommender.recommend(args=args)
                    return render_template("index.html", current_song=filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error=None, warning=None)


@app.route('/file')
def send_file():
    return send_from_directory('audio_data', request.args.get('path'))


@app.route('/record', methods=['GET', 'POST'])
def record():
    sound_recording.record()
    return render_template("index.html", current_song='Recorded audio', recommendations=None, predicted=None, scroll="app", error=None, warning='Audio recorded. Continue to get your recommendations for this audio sample.')


if __name__ == "__main__":
    app.run(debug=True)
