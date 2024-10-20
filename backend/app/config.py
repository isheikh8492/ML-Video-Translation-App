import os


class Config:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    PROCESSED_FOLDER = os.path.join(os.getcwd(), "processed")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # Limit upload size to 16MB
