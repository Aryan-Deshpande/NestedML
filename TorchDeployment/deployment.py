from crypt import methods
from flask import Flask, Blueprint
from flask import render_template, redirect, jsonify, request, abort

app = Flask(__name__)

@app.route('/prediction',methods=['GET','POST'])
def predict():
    pass


# Adding blueprint functionality, used for organizing routes ( http://localhost:8080/api/members) => gives an error 

# Parent API
api = Blueprint('api', __name__, url_prefix='/api')

# Child API
model_api = Blueprint('model_api', __name__, url_prefix='/model')

app.register_blueprint(api)
api.register_blueprint(model_api)

@model_api.route('/')
def afunction():
    abort(404)
#    | |
#    | |
#    | |
#     v
@model_api.errorhandler(404)
def errhandler(err):
    return 'there is an error'
