import cv2
import numpy as np


def remove_text_with_inpainting(image, ocr_data):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i in range(len(ocr_data["level"])):
        (x, y, w, h) = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image


def overlay_translated_text(image, translated_text, ocr_data):
    for i in range(len(ocr_data["level"])):
        (x, y, w, h) = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        cv2.putText(
            image,
            translated_text,
            (x, y + h),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image
