from flask import Flask, request, url_for, jsonify
from flask import render_template


app = Flask(__name__)

@app.route("/")
def index():
    # return "hello world"
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/hyperparameters")
def hyperparameters():
    return render_template("hyperparameters.html")


if __name__ == "__main__":
    app.run()