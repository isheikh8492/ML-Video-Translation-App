# inpaint.py
import cv2
import numpy as np
from app.video_processing.translate import translate_text

# Create mask based on text bounding boxes
def create_mask_from_frame(image, texts):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for text in texts:  # Skipping the first element as it contains the entire block
        vertices = [(vertex.x, vertex.y) for vertex in text["bounding_box"]]
        points = np.array(vertices, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)  # Fill the region of text with white (255)

    return mask


# Improve mask by dilating it
def improve_mask(mask):
    kernel = np.ones(
        (5, 5), np.uint8
    )  # Change kernel size to make mask larger or smaller
    improved_mask = cv2.dilate(mask, kernel, iterations=10)  # Expand the mask slightly
    return improved_mask


# Remove text using OpenCV2 inpainting
def remove_text_opencv_from_frame(frame, mask):
    inpainted_image = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
    return inpainted_image


def overlay_translated_text_on_frame(
    image, text_info, source_language, target_language
):
    """
    Overlays translated text onto an inpainted image.

    Args:
        image (np.array): Input image.
        text_info (list): Text information (bounding boxes, descriptions).
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        image (np.array): Image with translated text overlaid.
    """

    # Translate text and overlay onto inpainted image
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    thickness = 2
    for info in text_info:
        translated_text = translate_text(
            info["description"], source_language, target_language
        )
        x, y = info["bounding_box"][0].x, info["bounding_box"][0].y
        width = info["width"]
        height = info["height"]
        angle = info["angle"]

        # Auto Font Scaling
        font_scale = min(height / 20, 5)  # Dynamic font scaling

        # Calculate text positioning within bounding box
        text_x = x + 5  # Add a small padding
        text_y = y + height - 5  # Align text to bottom of bounding box

        # Rotate text for non-horizontal orientations
        if abs(angle) > 10:
            M = cv2.getRotationMatrix2D((text_x, text_y), angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        cv2.putText(
            image,
            translated_text,
            (int(text_x), int(text_y)),
            font,
            font_scale,
            color,
            thickness=4,
        )

        # Reverse rotation if applied
        if abs(angle) > 10:
            M = cv2.getRotationMatrix2D((text_x, text_y), -angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image
