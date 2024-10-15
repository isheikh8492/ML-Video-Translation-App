from flask import Flask
from flask_cors import CORS
import os


def create_app():
    app = Flask(__name__)

    # Enable CORS
    CORS(app)

    # Configuration setup
    app.config.from_object("app.config.Config")

    # Import and register the routes
    from app.routes import main

    app.register_blueprint(main)

    # Ensure upload and processed folders exist
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    if not os.path.exists(app.config["PROCESSED_FOLDER"]):
        os.makedirs(app.config["PROCESSED_FOLDER"])

    return app
