# Audio Recorder and Player Application
<div align="center">
    <img src="/screenshots/phonautograph.png" width="800px"</img> 
</div>

## Overview

This is a simple Audio Recorder and Player application built using Python and PyQt5. The application allows users to record audio, play recorded audio files, visualize audio waveforms, and transcribe recorded audio into text.

## Features

- **Recording:**
  - Click the "Record" button to start recording audio.
  - Click the "Stop" button to stop the recording.

- **Playback:**
  - Select a recorded audio file from the list and click the "Play" button to play the audio.
  - Click the "Pause" button to pause audio playback.

- **Visualization:**
  - The application provides a real-time visualization of the audio waveform during recording.

- **Transcription:**
  - Click the "Transcribe" button to transcribe the selected audio file into text.
  - The transcribed text, detected language, and sentiment analysis results are displayed.

## Requirements

- Python 3.x
- PyQt5
- NumPy
- Matplotlib
- PyAudio
- NLTK

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/CyberRoute/phonautograph.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    python main.py
    ```

## Usage

1. **Recording:**
   - Click the "Record" button to start recording.
   - Click the "Stop" button to stop recording.

2. **Playback:**
   - Select a recorded audio file from the list.
   - Click the "Play" button to play the selected audio.
   - Click the "Pause" button to pause audio playback.

3. **Transcription:**
   - Click the "Transcribe" button to transcribe the selected audio file.
   - Transcribed text, detected language, and sentiment analysis results are displayed.

4. **Visualization:**
   - The application provides a real-time visualization of the audio waveform during recording.

## File Management

- Recorded audio files are stored in the current directory with filenames in the format: `recorded_audio_YYYY-MM-DD_HH-MM-SS.wav`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Icons by [FontAwesome](https://fontawesome.com/).
- Sentiment analysis using [NLTK](https://www.nltk.org/).

Feel free to customize this README to better suit your project and provide more detailed instructions or information.
