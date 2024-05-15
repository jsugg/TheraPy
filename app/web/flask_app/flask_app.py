
from flask import Flask, Blueprint
from flask_socketio import SocketIO

def create_app() -> Flask:
    app: Flask = Flask(import_name=__name__)
    app.config['SECRET_KEY'] = 'my_secret_key'
    return app
