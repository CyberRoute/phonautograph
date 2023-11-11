from setuptools import setup

setup(
    name="AudioRecorderPlayer",
    version="1.0",
    packages=[""],
    url="",
    license="",
    author="CyberRoute",
    author_email="alessandro.bresciani2016@gmail.com",
    description="Description of your package",
    install_requires=["PyQt5", "numpy", "matplotlib", "whisper", "pyaudio"],
    entry_points={
        "console_scripts": [
            "audio_recorder_player = main:main",
        ],
    },
)
