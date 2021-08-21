
# ----- flask
from flask import Flask, jsonify
import os
app = Flask(__name__)
@app.route("/<name>", methods=['GET'])
def home(name):
    return jsonify({"message": f"Hello! {name}"})
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4455)

# run in terminal with: FLASK_APP=10a_api_flask.py flask run
# list all running processes in terminal: ps -a
# r shiny with flask api's: https://predictivehacks.com/how-to-share-flask-apis-with-shiny-applications/



