import cv2
from app.video_processing.ocr import detect_text_google_vision
from app.video_processing.inpaint import create_mask, improve_mask, overlay_translated_text, remove_text_opencv


# Main test function
def test_single_frame(image_path, output_path):
    # Step 1: Detect text using Google Vision API
    texts = detect_text_google_vision(image_path)
    print(texts)

    # Step 2: Load the image
    image = cv2.imread(image_path)

    # Step 3: Create mask from bounding boxes
    mask = create_mask(image, texts)

    # Step 4: Improve mask
    improved_mask = improve_mask(mask)

    # Step 5: Remove text using OpenCV2 inpainting
    inpainted_image_path = "inpainted_" + output_path
    remove_text_opencv(image_path, improved_mask, inpainted_image_path)

    source_language = "en"
    target_language = "es"
    # Step 6: Overlay translated text onto the inpainted image
    overlay_translated_text(inpainted_image_path, texts, output_path, 
                            source_language, target_language)


# Example usage
if __name__ == "__main__":
    input_image = "frames/frame_0157.png"  # Input file path
    output_image = "translated_inpainted_frame.png"
    print("Running Test Frame")
    # Test the single frame
    test_single_frame(input_image, output_image)
