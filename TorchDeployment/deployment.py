from crypt import methods
from flask import Flask, Blueprint
from flask import render_template, redirect, jsonify, request

app = Flask(__name__)

@app.route('/prediction',methods=['GET','POST'])
def predict():
    pass


# Adding blueprint functionality
model_api = Blueprint('model_api', __name__, url_prefix='/model')
app.register_blueprint(model_api)

@model_api.errorhandler(404)
def errhandler(err):
    return 'there is an error'
