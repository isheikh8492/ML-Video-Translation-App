from flask import Blueprint, jsonify, request, current_app
import os
from app.video_processing import extract, ocr, inpaint
import moviepy.editor as mp

main = Blueprint("main", __name__)


@main.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    video_path = os.path.join(current_app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    # Step 1: Extract frames in memory (as numpy arrays)
    frames = extract.extract_frames_in_memory(video_path)

    # Step 2: Process each frame (OCR, translate, inpaint)
    for i, frame in enumerate(frames):
        print("Frame " + str(i + 1))
        # OCR detection and text extraction
        texts = ocr.detect_text_google_vision_from_frame(frame)

        if not texts:
            # If no text is found, keep the original frame in place
            continue

        # Step 3: Process the frame with inpainting
        mask = inpaint.create_mask_from_frame(frame, texts)
        improved_mask = inpaint.improve_mask(mask)

        # Inpaint to remove text
        inpainted_image = inpaint.remove_text_opencv_from_frame(frame, improved_mask)

        # Overlay the translated text onto the inpainted image
        final_image = inpaint.overlay_translated_text_on_frame(
            inpainted_image, texts, source_language="en", target_language="es"
        )

        # Replace the original frame with the processed frame in the list
        frames[i] = final_image

    # Step 3: Reassemble the video from the processed frames
    output_video_path = os.path.join(
        current_app.config["PROCESSED_FOLDER"], f"processed_{video.filename}"
    )
    clip = mp.ImageSequenceClip.List(frames, fps=30)
    clip.write_videofile(output_video_path)

    return jsonify(
        {"outputVideoUrl": f"/processed/{os.path.basename(output_video_path)}"}
    )
