from google.cloud import translate_v2 as translate
from io import BytesIO
import base64

def translate_text(text, source_language="en", target_language="es"):
    """Translate text from source language to target language using Google Translate API."""
    translate_client = translate.Client()

    # Translate the text
    result = translate_client.translate(
        text, source_language=source_language, target_language=target_language
    )

    return result["translatedText"]


# Convert image to base64
def image_to_base64(image, format="JPEG"):
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_str
