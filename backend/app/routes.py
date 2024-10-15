from flask import Blueprint, jsonify, request
import os
from app.video_processing import extract, ocr, translate, inpaint, reassemble

main = Blueprint("main", __name__)

@main.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    video_path = os.path.join(current_app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    # Step 1: Extract frames
    frames = extract.extract_frames(video_path, current_app.config["UPLOAD_FOLDER"])

    # Step 2: Process each frame (OCR, translate, inpaint)
    for frame_path in frames:
        image, ocr_data = ocr.detect_text(frame_path)
        original_text = ocr_data["text"]
        translated_text = translate.translate_text(original_text, "en", "es")
        inpainted_image = inpaint.remove_text_with_inpainting(image, ocr_data)
        final_image = inpaint.overlay_translated_text(
            inpainted_image, translated_text, ocr_data
        )

    # Step 3: Reassemble the video
    output_video_path = os.path.join(
        current_app.config["PROCESSED_FOLDER"], f"processed_{video.filename}"
    )
    reassemble.reassemble_video(current_app.config["UPLOAD_FOLDER"], output_video_path)

    return jsonify(
        {"outputVideoUrl": f"/processed/{os.path.basename(output_video_path)}"}
    )
