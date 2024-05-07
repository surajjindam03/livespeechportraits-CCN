import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Results = () => {
  const [modelLoading, setModelLoading] = useState(true);
  const [videoUrl, setVideoUrl] = useState(null);

  useEffect(() => {
    // Fetch model loading status from the backend
    axios.get('http://localhost:3001/modelstatus')
      .then(response => {
        setModelLoading(response.data.model_loading);
        // If model_loading is false, fetch the video URL from the backend
        console.log(response.data.model_loading)
        if (!response.data.model_loading) {
          axios.get('http://localhost:3001/getvideo')
            .then(response => {
                console.log(response.data);
              setVideoUrl(response.data);
            })
            .catch(error => {
              console.error('Error fetching video URL:', error);
            });
        }
      })
      .catch(error => {
        console.error('Error fetching model status:', error);
      });
  }, []); // Run once on component mount

  return (
    <div>
      {modelLoading ? (
        <p>Process is still ongoing...</p>
      ) : (
        <div>
          <br/><br/><br/>
          <center>
          {videoUrl ? (
            <video controls>
              <source src={videoUrl} type="video/x-msvideo" />
              Your browser does not support the video tag.
            </video>
          ) : (
            <p>No video available</p>
          )}
          </center>
        </div>
      )}
    </div>
  );
};

export default Results;
