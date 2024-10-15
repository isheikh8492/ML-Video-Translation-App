import cv2
import pytesseract


def detect_text(frame_path):
    image = cv2.imread(frame_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return image, data
