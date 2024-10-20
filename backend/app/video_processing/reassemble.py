import cv2
import os


def reassemble_video(frames_folder, output_path, fps=30):
    images = [img for img in os.listdir(frames_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(frames_folder, image)))

    video.release()
