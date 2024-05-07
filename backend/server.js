const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');
const cors = require('cors');
const fs = require('fs');
const app = express();

const port = 3001;

let model_loading = false; // Initialize model_loading status
let audioInfo = {}; // Object to store selectedOption and filename

app.use(cors({
    origin: 'http://localhost:3000', // Allow requests from frontend URL (change as needed)
    methods: ['GET', 'POST'], // Allow specific HTTP methods
  }));

// Set up Multer storage to save uploaded audio files to the ./data/ directory
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dataDir = './data/Input/';
    fs.mkdirSync(dataDir, { recursive: true }); // Create ./data/Input/ directory if it doesn't exist
    cb(null, dataDir); // Save uploaded files to the './data/Input/' directory
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const filename = `audio_${Date.now()}${ext}`; // Set a unique filename for the uploaded file
    audioInfo = { selectedOption: req.body.selectedOption, filename }; // Store selectedOption and filename
    cb(null, filename);
  },
});

const upload = multer({ storage });

// Middleware to set model_loading status
const setLoadingStatus = (req, res, next) => {
  model_loading = true;
  next();
};

// POST route to handle audio file upload and store selectedOption and filename
app.post('/speech', setLoadingStatus, upload.single('audio'), (req, res) => {
  try {
    if (!req.file) {
      model_loading = false; // Set model_loading to false if no file was uploaded
      return res.status(400).json({ error: 'No audio file uploaded' });
    }

    const { selectedOption } = req.body;
    const { filename } = audioInfo;

    console.log(`Received audio file: ${filename}`);
    console.log(`Selected option: ${selectedOption}`);

    // Construct the command to run Python script with the selected option as argument
    const pythonScriptPath = 'C:/Users/suraj/Documents/Suraj/Spring-2024-sem/CCN/Project/LiveSpeechPortraits-main/demo.py';
    const condaPath = 'C:/Users/suraj/anaconda3/condabin/';

    const command = `${condaPath}conda init && ${condaPath}conda activate LSP && python "${pythonScriptPath}" --id ${selectedOption} --driving_audio "./data/Input/${filename}"`;

    exec(command, (error, stdout, stderr) => {
      model_loading = false; // Set model_loading to false after script execution
      if (error) {
        console.error('Error running Python script:', error);
        return res.status(500).json({ error: 'Failed to run Python script' });
      }
      
      console.log('Python script executed successfully:', stdout);
      res.status(200).json({ message: 'Audio file uploaded and processed successfully', filename });
    });
  } catch (error) {
    model_loading = false; // Set model_loading to false on error
    console.error('Error uploading audio:', error);
    res.status(500).json({ error: 'Failed to upload audio file' });
  }
});

// Route to get model_loading status
app.get('/modelstatus', (req, res) => {
  res.status(200).json({ model_loading });
});

// Route to retrieve video URL
app.get('/getvideo', (req, res) => {
  try {
    var { selectedOption, filename } = audioInfo;

    selectedOption = "Nadella"
    filename = "audio_1715047426969"

    console.log(selectedOption,filename)

    if (!selectedOption || !filename) {
      return res.status(400).json({ error: 'Missing audio information' });
    }


    const videoPath = path.join(__dirname, 'results', selectedOption, `${filename}`, `${filename}.avi`);

    console.log(videoPath)

    if (fs.existsSync(videoPath)) {
        res.status(200).send(videoPath);
    } else {
      res.status(404).json({ error: 'Video not found' });
    }
  } catch (error) {
    console.error('Error retrieving video:', error);
    res.status(500).json({ error: 'Failed to retrieve video' });
  }
});

// Start the Express server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
