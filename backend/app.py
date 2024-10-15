import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# Define the upload folder path
UPLOAD_FOLDER = "uploads"


def create_app():
    # Initialize the Flask app
    app = Flask(__name__)

    # Enable CORS
    CORS(app)

    # Set the folder for uploaded files
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    return app


# Create the Flask app instance
app = create_app()


@app.route("/")
def home():
    """Home route to test the API"""
    return jsonify(name="ML Video Translation App")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Route to handle video uploads"""
    # Check if a video file is in the request
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Get the video file from the request
    video = request.files["video"]

    # Save the video file to the upload folder
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    # Process the video here (e.g., OCR, translation, etc.)
    # For now, just return a mock processed video path
    processed_video_url = f"/uploads/{video.filename}"

    return jsonify({"outputVideoUrl": processed_video_url})


if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(debug=True)
