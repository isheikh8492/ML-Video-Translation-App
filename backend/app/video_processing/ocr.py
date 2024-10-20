# ocr.py
import io
from google.cloud import vision
import numpy as np
import cv2


# Detect text using Google Vision API
def detect_text_google_vision_from_frame(frame):
    client = vision.ImageAnnotatorClient()

    # Convert numpy array to bytes
    _, encoded_image = cv2.imencode(".jpg", frame)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(f"{response.error.message}")

    if not texts:
        print("No text detected")
        return None

    print(f"Detected {len(texts)} text elements.")

    text_info = []

    for i, text in enumerate(
        texts[1:], 1
    ):  # Skipping the first element that contains the entire block
        vertices = text.bounding_poly.vertices
        width = np.linalg.norm(
            np.array([vertices[1].x, vertices[1].y])
            - np.array([vertices[0].x, vertices[0].y])
        )
        height = np.linalg.norm(
            np.array([vertices[3].x, vertices[3].y])
            - np.array([vertices[0].x, vertices[0].y])
        )

        delta_x = vertices[1].x - vertices[0].x
        delta_y = vertices[1].y - vertices[0].y
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        text_info.append(
            {
                "description": text.description,
                "width": width,
                "height": height,
                "angle": angle,
                "bounding_box": vertices,
            }
        )

        print(f"Text {i}: '{text.description}'")
        print(f"  Size (width, height): {width:.2f}, {height:.2f}")
        print(f"  Orientation: {angle:.2f} degrees")
        print(f"  Bounding Box: {vertices}")

    return text_info
