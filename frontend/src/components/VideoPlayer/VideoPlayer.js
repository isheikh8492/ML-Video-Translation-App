import React from "react";

const VideoPlayer = ({ title, videoSrc }) => {
  return (
    <div className="video-player">
      <h3>{title}</h3>
      {videoSrc ? (
        <video width="400" controls>
          <source src={videoSrc} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      ) : (
        <div className="video-placeholder">No video available</div>
      )}
    </div>
  );
};

export default VideoPlayer;
