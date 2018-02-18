import os
from flask import Flask, request, url_for, render_template, make_response, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from datetime import datetime
import errno

import initialise
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
data = initialise.init()


# Utility function for formatting predicted genres
def make_string(list):
    return ", ".join(list)


# Ensure uploaded file has one of the allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# The single-page web-app before recommendations have been generated
@app.route('/')
def index():
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll=None, error=None, warning=None)


# Route for retrieving files from dataset for recommendation preview functionality
@app.route('/file')
def send_file():
    return send_from_directory('audio_data', request.args.get('path'))


# Route to perform recommendation and return results
@app.route('/recommend', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global previous_upload_dir
        global previous_filenames
        use_previous_path = False

        # Create directory with unique name for uploaded files
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        directory = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise OSError('Error creating directory for uploaded files')

        if request.form['input-method'] == 'mic':
            d, f = os.path.split(sound_recording.record(directory)) # Record audio and get path components
            # Retrieve form info
            filenames = [f]
            mode = request.form['mode']
            vector_type = request.form['features']
            # Update arguments and previous upload information to current upload
            args = [directory, mode, vector_type, data]
            previous_upload_dir = directory
            previous_filenames = filenames
            recommendations, predictions, warning = recommender.recommend(args=args)  # Perform recommendation
            # Return the template with the included information from the recommendation process
            return render_template("index.html", current_song=filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
        elif request.form['input-method'] == 'file':
            # Check if post request has file part
            if 'file[]' not in request.files:
                use_previous_path = True
            if use_previous_path:
                if previous_upload_dir is None or previous_filenames is None:
                    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="No file selected", warning=None)
            # Retrieve files from request
            file_list = request.files.getlist("file[]")
            for f in file_list:
                # Check uploaded files are valid
                if f.filename == '':
                    use_previous_path = True
                    if previous_upload_dir is None or previous_filenames is None:
                        return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error="Invalid filename", warning=None)
            # Retrieve other information from request
            mode = request.form['mode']
            vector_type = request.form['features']
            if use_previous_path:
                args = [previous_upload_dir, mode, vector_type, data]
                recommendations, predictions, warning = recommender.recommend(args=args)
                return render_template("index.html", current_song=previous_filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
            elif mode and all(allowed_file(f.filename) for f in file_list) and all(f for f in file_list):
                # Ensure filenames are secure and save the uploaded files
                filenames = []
                for f in file_list:
                    filename = secure_filename(f.filename)
                    path = os.path.join(directory, filename)
                    f.save(path)
                    filenames.append(filename)
                # Update arguments and previous upload information to current upload
                args = [directory, mode, vector_type, data]
                previous_upload_dir = directory
                previous_filenames = filenames
                recommendations, predictions, warning = recommender.recommend(args=args)  # Perform recommendation
                # Return the template with the included information from the recommendation process
                return render_template("index.html", current_song=filenames, recommendations=recommendations, predicted=make_string(predictions), scroll="app", error=None, warning=warning)
    return render_template("index.html", current_song=None, recommendations=None, predicted=None, scroll="app", error=None, warning=None)


if __name__ == "__main__":
    app.run(debug=True)
