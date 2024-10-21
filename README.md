# ML Video Translation App

This is a web application that processes video files to detect text, translate it, and overlay the translated text back onto the video. The app consists of two main components:

1. **Frontend**: A web interface for uploading video files and interacting with the backend.
2. **Backend**: A Flask-based API that handles the video processing tasks, such as frame extraction, text detection (OCR), inpainting, and translation.

---

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Frontend Setup](#frontend-setup)
- [Backend Setup](#backend-setup)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Upload video files via the web interface.
- Extract video frames and perform Optical Character Recognition (OCR) on each frame.
- Translate the detected text into a target language.
- Overlay the translated text back onto the video.
- Reassemble the processed frames into a new video.
  
---

## Technologies Used

### Frontend
- **React.js**
- **HTML/CSS**
  
### Backend
- **Flask**
- **MoviePy** for video processing
- **Google Cloud Vision API** for OCR
- **Google Cloud Translate API** for text translation
- **OpenCV** for inpainting text from video frames

---

## Frontend Setup

1. **Install Dependencies**
   Navigate to the `frontend` directory and install the required dependencies.

   ```bash
   cd frontend
   npm install
   ```

2. **Run the Development Server**
   After installing dependencies, you can run the React development server with:

   ```bash
   npm start
   ```

   This will start the frontend on `http://localhost:3000`.

---

## Backend Setup

### Prerequisites
- Python 3.11 or above
- Google Cloud account (for Vision and Translate APIs)
- Set up environment variables for your Google Cloud API keys.

### Steps

1. **Clone the Repository**
   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/ml-video-translation-app.git
   cd ml-video-translation-app/backend
   ```

2. **Create a Virtual Environment**
   Set up a Python virtual environment to manage dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Backend Dependencies**

   Install the required Python libraries for the backend:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   
   Set up the following environment variables:

   - `UPLOAD_FOLDER`: Path to the folder where uploaded videos will be saved.
   - `PROCESSED_FOLDER`: Path to the folder where processed videos will be stored.

   For example:

   ```bash
   export UPLOAD_FOLDER=/path/to/uploaded/videos
   export PROCESSED_FOLDER=/path/to/processed/videos
   ```

5. **Run the Flask Server**
   
   Run the Flask development server:

   ```bash
   python app.py
   ```

   This will start the backend on `http://localhost:8080`.

---

## API Endpoints

- **POST `/upload`**
   - Upload a video to process.
   - **Request**: A `multipart/form-data` request containing the video file.
   - **Response**: JSON response with the URL to the processed video.

   **Example Request**:

   ```bash
   curl -X POST -F video=@your_video.mp4 http://localhost:8080/upload
   ```

---

## Contributing

Contributions are welcome! Please follow the standard GitHub fork, feature branch, pull request workflow.

---

## License

This project is licensed under the MIT License.

---

Let me know if you'd like to add or modify any section!
