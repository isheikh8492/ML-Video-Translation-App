# extract.py
import cv2
import os


import cv2
import os

def extract_frames_in_memory(video_path):
    """
    Extracts frames from a video and stores them in memory as numpy arrays.
    Args:
        video_path (str): Path to the video file.
    Returns:
        frames (list): List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Store frame as numpy array
    
    cap.release()
    return frames

def reassemble_frames_in_memory(frames, output_video, fps=30):
    """
    Reassembles frames stored in memory (numpy arrays) into a video file.
    Args:
        frames (list): List of frames as numpy arrays.
        output_video (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    if not frames:
        print("No frames found for reassembly!")
        return

    frame_size = frames[0].shape[1::-1]  # Get frame size from the first frame

    # Define video codec and output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

