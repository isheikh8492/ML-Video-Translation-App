import cv2
import os


def extract_frames(video_path, output_folder):
    video_capture = cv2.VideoCapture(video_path)
    frame_number = 0
    frames = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_number}.png")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        frame_number += 1

    video_capture.release()
    return frames
