import datetime
import os
import threading
import time
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
)
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
from PyQt5 import QtCore
from transcriber import SoundTranscriber
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


class AudioRecorderPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Recorder and Player")
        self.setGeometry(100, 100, 400, 300)
        self.setMinimumSize(400, 300)

        self.record_button = QPushButton(QIcon.fromTheme("media-record"), "", self)
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.record_button.setToolTip("Start recording audio")
        self.record_button.setStyleSheet("background-color: #555555;")

        self.stop_button = QPushButton(QIcon.fromTheme("media-playback-stop"), "", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.stop_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.stop_button.setToolTip("Stop recording audio")
        self.stop_button.setStyleSheet("background-color: #555555;")

        self.play_button = QPushButton(
            QIcon.fromTheme("media-playback-start"), "", self
        )
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(True)
        self.play_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.play_button.setToolTip("Play selected audio file")
        self.play_button.setStyleSheet("background-color: #555555;")

        self.pause_button = QPushButton(
            QIcon.fromTheme("media-playback-pause"), "", self
        )
        self.pause_button.clicked.connect(self.pause_audio)
        self.pause_button.setEnabled(True)
        self.pause_button.setIconSize(QtCore.QSize(48, 48))
        self.pause_button.setToolTip("Pause audio playback")
        self.pause_button.setStyleSheet("background-color: #555555;")

        self.transcribe_button = QPushButton(
            QIcon.fromTheme("document-export"), "", self
        )
        self.transcribe_button.clicked.connect(self.transcribe_audio_button)
        self.transcribe_button.setEnabled(True)

        self.transcribe_button.setIconSize(QtCore.QSize(48, 48))  # Set icon size
        self.transcribe_button.setToolTip("Transcribe Audio")
        self.transcribe_button.setStyleSheet("background-color: #555555;")

        self.status_label = QLabel("Status: Ready", self)

        # Create a QListWidget to display recorded files
        self.file_list_widget = QListWidget(self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.transcribe_button)
        button_layout.addWidget(self.status_label)

        # Nested QVBoxLayout for other widgets
        control_layout = QVBoxLayout()
        control_layout.addLayout(button_layout)  # Add button layout horizontally
        control_layout.addWidget(self.file_list_widget)

        self.transcription_box = QTextEdit(self)
        self.transcription_box.setPlaceholderText("Transcribed Text")
        self.transcription_box.setReadOnly(True)
        control_layout.addWidget(self.transcription_box)

        self.plot_widget = plt.figure()
        self.plot_canvas = self.plot_widget.add_subplot(111)
        self.plot_canvas.set_xlabel("Time")
        self.plot_canvas.set_ylabel("Amplitude")
        control_layout.addWidget(self.plot_widget.canvas)

        self.progress_bar = QProgressBar(self)  # Defining progress bar on the layout
        control_layout.addWidget(self.progress_bar)

        self.setLayout(control_layout)
        self.update_file_list()
        self.stream = pyaudio.PyAudio()
        self.transcriber = SoundTranscriber()

        self.plot_interval_ms = 10  # Update the waveform every 100 milliseconds
        self.frames_per_update = (
            44100 // 1024 * (self.plot_interval_ms // 1000) or 1
        )  # Minimum 1 frame per update

        self.sample_width = 2  # 2 bytes for 16-bit audio
        self.channels = 1  # Mono audio
        self.sample_rate = 16000  # Samples per second
        self.frames = []
        self.audio_stream = None
        self.recording_thread = None
        self.recording = False
        self.playback_thread = None
        self.playback_event = threading.Event()

    def plot_waveform(self):
        if self.frames:
            samples = np.frombuffer(b"".join(self.frames), dtype=np.int16)
            x_values = np.arange(0, len(samples), 1)
            self.plot_canvas.clear()
            self.plot_canvas.set_xlabel("Time")
            self.plot_canvas.set_ylabel("Amplitude")
            self.plot_canvas.set_ylim(-32768, 32768)
            self.plot_canvas.set_facecolor("black")
            self.plot_canvas.plot(x_values, samples, color="green")

            # Add horizontal lines for amplitudes
            self.plot_canvas.axhline(y=0, color="white", linewidth=1, linestyle="--")
            self.plot_canvas.axhline(
                y=16384, color="white", linewidth=1, linestyle="--"
            )
            self.plot_canvas.axhline(
                y=-16384, color="white", linewidth=1, linestyle="--"
            )

            # Set x-ticks and labels
            num_ticks = 10
            x_tick_positions = np.linspace(0, len(samples) - 1, num_ticks)
            x_tick_labels = (
                np.linspace(0, len(samples) - 1, num_ticks) / self.sample_rate
            )
            self.plot_canvas.set_xticks(x_tick_positions)
            self.plot_canvas.set_xticklabels(
                ["{:.2f}s".format(t) for t in x_tick_labels]
            )
            self.plot_widget.canvas.draw()

    def transcribe_audio_button(self):
        selected_item = self.file_list_widget.currentItem()
        if selected_item:
            file_name = selected_item.text()
            file_path = os.path.join(
                ".", file_name
            )  # Assuming files are in the current directory

            # Check if the file exists before attempting to transcribe it
            if os.path.exists(file_path):
                detected_language, transcribed_text = self.transcriber.transcribe_audio(
                    file_path
                )

                # Ensure vader_lexicon resource is available
                nltk.download("vader_lexicon")

                # Perform sentiment analysis on the transcribed text
                sentiment_analyzer = SentimentIntensityAnalyzer()
                sentiment_scores = sentiment_analyzer.polarity_scores(transcribed_text)
                sentiment = (
                    "Positive" if sentiment_scores["compound"] >= 0 else "Negative"
                )

                # Format the text with detected language, transcribed text, and sentiment
                formatted_text = (
                    f"<b>Detected Language:</b> {detected_language}<br><br>"
                    f"<b>Transcribed Text:</b><br>{transcribed_text}<br><br>"
                    f"<b>Sentiment:</b> {sentiment}<br>"
                )

                print(f"Detected language: {detected_language}")
                print(f"Transcribed text: {transcribed_text}")
                print(f"Sentiment: {sentiment}")

                # Update the QTextEdit widget with formatted text
                self.transcription_box.setHtml(formatted_text)
            else:
                print(
                    "File not found:", file_path
                )  # Print an error message if the file doesn't exist

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
        directory_path = (
            "."  # Change this to the directory path where your .wav files are stored
        )
        wav_files = [
            file for file in os.listdir(directory_path) if file.endswith(".wav")
        ]

        # Update the file list widget with the .wav files from the directory
        self.file_list_widget.clear()
        for file_name in wav_files:
            item = QListWidgetItem(file_name)
            self.file_list_widget.addItem(item)

    def initialize_audio_stream(self):
        stream = self.stream.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=1024,
            input=True,
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
        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # Format the timestamp as desired
        file_name_with_timestamp = f"recorded_audio_{timestamp}.wav"
        item = QListWidgetItem(file_name_with_timestamp)
        self.file_list_widget.addItem(item)

        with wave.open(file_name_with_timestamp, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
            wf.close()

    def play_audio(self):
        selected_item = self.file_list_widget.currentItem()
        if selected_item:
            file_name = selected_item.text()
            file_path = os.path.join(
                ".", file_name
            )  # Assuming files are in the current directory

            if os.path.exists(file_path):
                self.playback_thread = threading.Thread(
                    target=self.play_audio_thread, args=(file_path,)
                )
                self.playback_thread.start()
            else:
                print("File not found:", file_path)

    def play_audio_thread(self, file_path):
        wf = wave.open(file_path, "rb")
        p = pyaudio.PyAudio()

        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        total_frames = wf.getnframes()
        total_time = total_frames / wf.getframerate()
        data = wf.readframes(1024)
        start_time = time.time()
        while data and not self.playback_event.is_set():
            stream.write(data)
            data = wf.readframes(1024)
            elapsed_time = time.time() - start_time
            progress = int((elapsed_time / total_time) * 100)
            # Ensure progress reaches 100% when playback is complete
            if progress == 99:
                progress = 100
            self.progress_bar.setValue(progress)

        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()

        self.playback_event.clear()
        self.status_label.setText("Audio playback paused")

    def pause_audio(self):
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_event.set()
            self.status_label.setText("Audio paused")
