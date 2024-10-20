import React, { useState } from "react";
import axios from "axios";
import ProgressBar from "./ProgressBar/ProgressBar";
import VideoPlayer from "./VideoPlayer/VideoPlayer";
import "./VideoUpload.css";

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [outputVideo, setOutputVideo] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadedVideo(URL.createObjectURL(e.target.files[0]));
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_FLASK_URL}/upload`,
        formData,
        {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );

      setOutputVideo(response.data.outputVideoUrl);
    } catch (err) {
      console.error("Error uploading video", err);
    }
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Your Video</h2>

      {/* Video Upload Section */}
      <div className="upload-section">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="upload-input"
        />
        <button onClick={handleUpload} className="upload-btn">
          Upload
        </button>
        {progress > 0 && (
          <div className="progress-wrapper">
            <ProgressBar progress={progress} />
          </div>
        )}
      </div>

      {/* Video Players */}
      <div className="video-section">
        <div className="video-card">
          <VideoPlayer title="Uploaded Video" videoSrc={uploadedVideo} />
        </div>
        <div className="separator"></div>
        <div className="video-card">
          <VideoPlayer title="Processed Video" videoSrc={outputVideo} />
        </div>
      </div>
    </div>
  );
};

export default VideoUpload;
