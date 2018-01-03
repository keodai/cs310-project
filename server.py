from flask import Flask, jsonify, request

import recommender

app = Flask(__name__)


@app.route('/recommend')
def recommend():
    return jsonify(result=recommender.main(**request.args))


if __name__ == "__main__":
    app.run(debug=True)