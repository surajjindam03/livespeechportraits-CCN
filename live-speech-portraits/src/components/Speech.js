import React, { useState } from 'react';
import axios from 'axios';

const Speech = () => {
  const [selectedOption, setSelectedOption] = useState('May');
  const [recording, setRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setRecordedAudio(blob);
      };

      recorder.start();
      setRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    setRecording(false);
  };

  const handleRecordToggle = () => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleUploadAudio = (e) => {
    const file = e.target.files[0];
    setRecordedAudio(file);
  };

  const handleSubmit = async () => {
    try {
      if (!recordedAudio) {
        console.error('No audio to submit');
        return;
      }

      const formData = new FormData();
      formData.append('audio', recordedAudio);
      formData.append('selectedOption', selectedOption);

      const response = await axios.post('http://localhost:3001/speech', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Audio submitted successfully:', response.data);
      // Handle successful submission (e.g., show success message)
    } catch (error) {
      console.error('Error submitting audio:', error);
      // Handle error (e.g., show error message)
    }
  };

  return (
    <div>
      <h2>Speech Component</h2>

      {/* Select dropdown menu */}
      <label htmlFor="selectOption">Select Option:</label>
      <select id="selectOption" value={selectedOption} onChange={(e) => setSelectedOption(e.target.value)}>
        <option value="May">May</option>
        <option value="Obama1">Obama1</option>
        <option value="Obama2">Obama2</option>
        <option value="McStay">McStay</option>
        <option value="Nadella">Nadella</option>
      </select>
      <br/><br/>

      {/* Record/Stop Recording button */}
      <button onClick={handleRecordToggle}>{recording ? 'Stop Recording' : 'Start Recording'}</button>

      {/* Upload audio file input */}
      <input type="file" accept="audio/*" onChange={handleUploadAudio} />

      
        <br/><br/>
      {/* Display recorded audio (optional) */}
      {recordedAudio && (
        <audio controls>
          <source src={URL.createObjectURL(recordedAudio)} type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
      )}
        <br/><br/>
      {/* Submit button */}
      <button onClick={handleSubmit} disabled={!recording && !recordedAudio}>
        Submit
      </button>
    </div>
  );
};

export default Speech;
