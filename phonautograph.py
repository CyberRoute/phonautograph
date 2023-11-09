import datetime
import sys
import os
import threading
import time
import wave
import pyaudio
import whisper
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QListWidget, QListWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore

class AudioRecorderPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Recorder and Player")
        self.setGeometry(100, 100, 400, 300)

        self.record_button = QPushButton(QIcon.fromTheme("media-record"), "", self)
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.record_button.setToolTip("Start recording audio")

        self.stop_button = QPushButton(QIcon.fromTheme("media-playback-stop"), "", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.stop_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.stop_button.setToolTip("Stop recording audio")

        self.play_button = QPushButton(QIcon.fromTheme("media-playback-start"), "", self)
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        self.play_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.play_button.setToolTip("Play selected audio file")

        self.status_label = QLabel("Status: Ready", self)

        # Create a QListWidget to display recorded files
        self.file_list_widget = QListWidget(self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.status_label)

        # Nested QVBoxLayout for other widgets
        control_layout = QVBoxLayout()
        control_layout.addLayout(button_layout)  # Add button layout horizontally
        control_layout.addWidget(self.file_list_widget)

        self.detected_language_box = QTextEdit(self)
        self.detected_language_box.setPlaceholderText("Detected Language")
        self.detected_language_box.setReadOnly(True)

        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Transcribed Text")
        self.transcription_box.setReadOnly(True)
        control_layout.addWidget(self.detected_language_box)
        control_layout.addWidget(self.transcription_box)

        self.plot_widget = plt.figure()
        self.plot_canvas = self.plot_widget.add_subplot(111)
        self.plot_canvas.set_xlabel('Time')
        self.plot_canvas.set_ylabel('Amplitude')
        self.plot_canvas.set_ylim(-32768, 32768)  # Assuming 16-bit audio
        control_layout.addWidget(self.plot_widget.canvas)

        self.setLayout(control_layout)
        self.update_file_list()
        self.stream = pyaudio.PyAudio()


        self.plot_interval_ms = 10  # Update the waveform every 100 milliseconds
        self.frames_per_update = 44100 // 1024 * (self.plot_interval_ms // 1000) or 1  # Minimum 1 frame per update

        self.sample_width = 2  # 2 bytes for 16-bit audio
        self.channels = 1  # Mono audio
        self.sample_rate = 16000  # Samples per second
        self.frames = []
        self.audio_stream = None
        self.recording_thread = None
        self.recording = False

        
    def transcribe_audio(self, file_path):
        model = whisper.load_model("base")
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        return detected_language, result.text
    

    def plot_waveform(self):
        if self.frames:
            samples = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            x_values = np.arange(0, len(samples), 1)
            self.plot_canvas.clear()
            self.plot_canvas.set_xlabel('Time')
            self.plot_canvas.set_ylabel('Amplitude')
            self.plot_canvas.set_ylim(-32768, 32768)
            self.plot_canvas.plot(x_values, samples)
            self.plot_widget.canvas.draw()
        
        
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.play_button.setEnabled(False)
        self.status_label.setText("Recording...")
        self.frames = []
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def record_audio(self):
        self.recording = True
        self.audio_stream = self.initialize_audio_stream()

        while self.recording:
            audio_chunk = self.audio_stream.read(1024)
            self.frames.append(audio_chunk)
            if len(self.frames) % self.frames_per_update == 0:
                self.plot_waveform()

    def update_file_list(self):
        # Path to the directory where .wav files are stored
        directory_path = "."  # Change this to the directory path where your .wav files are stored
        wav_files = [file for file in os.listdir(directory_path) if file.endswith(".wav")]

        # Update the file list widget with the .wav files from the directory
        self.file_list_widget.clear()
        for file_name in wav_files:
            item = QListWidgetItem(file_name)
            self.file_list_widget.addItem(item)
                

    def initialize_audio_stream(self):
        stream = self.stream.open(format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        frames_per_buffer=1024,
                        input=True
                        )
        return stream

    def stop_recording(self):
        self.recording = False
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.play_button.setEnabled(True)
        self.status_label.setText("Recording stopped")

        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None  # Set the audio stream to None after closing

        self.save_audio_to_file()
        self.update_file_list()

    def save_audio_to_file(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format the timestamp as desired
        file_name_with_timestamp = f"recorded_audio_{timestamp}.wav"
        item = QListWidgetItem(file_name_with_timestamp)
        self.file_list_widget.addItem(item)
        
        with wave.open(file_name_with_timestamp, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()

    def play_audio(self):
        selected_item = self.file_list_widget.currentItem()
        if selected_item:
            file_name = selected_item.text()
            file_path = os.path.join(".", file_name)  # Assuming files are in the current directory

            # Check if the file exists before attempting to play it
            if os.path.exists(file_path):
                wf = wave.open(file_path, 'rb')

                # Create a new PyAudio object for playback
                p = pyaudio.PyAudio()

                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)

                data = wf.readframes(1024)  # Read audio data in chunks
                while data:
                    stream.write(data)  # Play audio data
                    data = wf.readframes(1024)

                stream.stop_stream()
                stream.close()
                wf.close()

                detected_language, transcribed_text = self.transcribe_audio(file_path)
                print(f"Detected language: {detected_language}")
                print(f"Transcribed text: {transcribed_text}")
                # Update the QTextEdit widgets with detected language and transcribed text
                self.detected_language_box.setPlainText("Detected Language: " + detected_language)
                self.transcription_box.setPlainText("Transcribed Text: " + transcribed_text)

                # Terminate the new PyAudio object after playback
                p.terminate()
            else:
                print("File not found:", file_path)  # Print an error message if the file doesn't exist



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioRecorderPlayer()
    window.show()
    sys.exit(app.exec_())

