import whisper

class SoundTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_audio(self, file_path):
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)
        return detected_language, result.text