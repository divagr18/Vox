import speech_recognition as sr
import pyaudio
import pyautogui
import time
import webbrowser
import win32com.client
from google.cloud import speech_v1p1beta1 as speech
import os
import base64
import threading

audio = pyaudio.PyAudio()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agile-infinity-419609-0d76c5aa6ca2.json"
client = speech.SpeechClient()


def enable_bluetooth():
    """Simulates pressing the F15 key to enable Bluetooth on Windows via a shell command.

    This function utilizes the Windows Script Host Shell COM interface to send a virtual
    key press corresponding to F15, which is assumed to toggle Bluetooth on supported
    systems or software configurations.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F15}")


def disable_bluetooth():
    """Simulate pressing the F16 key to disable Bluetooth on supported Windows systems.

    This function uses the Windows Scripting Host shell interface to send the F16 key press,
    which is mapped on certain devices to toggle Bluetooth off.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F16}")


def open_gmail():
    """Open the Gmail web interface in the default web browser.

    This function launches the user's default web browser and navigates directly to the Gmail inbox page.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://mail.google.com")


def open_weather():
    """Open the default web browser to display the weather.com homepage.

    This function launches the user's default web browser and navigates to https://www.weather.com, providing quick access to weather information.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://www.weather.com")


def quit_process():
    """Stops the running voice recognition process by setting the global `running` flag to False.

    This function signals the termination of the ongoing process, often used to gracefully exit loops or background tasks in the voice recognition system.

    Args:
        None

    Returns:
        None"""
    global running
    print("Quitting process...")
    running = False


command_actions = {
    "open mail": open_gmail,
    "check mail": open_gmail,
    "view mail": open_gmail,
    "open weather": open_weather,
    "check weather": open_weather,
    "view weather": open_weather,
    "enable bluetooth": enable_bluetooth,
    "turn on bluetooth": enable_bluetooth,
    "disable bluetooth": disable_bluetooth,
    "turn off bluetooth": disable_bluetooth,
    "quit": quit_process,
    "stop": quit_process,
}
running = True
processing = True


def process_audio(audio_data):
    """Processes raw audio data for speech recognition using Google Speech-to-Text API and executes corresponding commands.

    Args:
        audio_data (AudioData): An instance containing the raw audio input to be processed.

    Returns:
        None

    This function operates only if the global `processing` flag is True. It converts the audio data to the required format, encodes it, and sends it to the Google Speech-to-Text API. Upon receiving a response, it extracts and processes the recognized transcript by matching it against predefined commands. If the transcript is "quit", it triggers a shutdown process via `quit_process()`. Other recognized commands invoke their corresponding actions from the `command_actions` mapping. Errors in recognition or API requests are caught and logged to the console."""
    global processing
    try:
        if processing:
            audio_content = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            audio_content_base64 = base64.b64encode(audio_content).decode("utf-8")
            audio = speech.RecognitionAudio(content=audio_content_base64)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            response = client.recognize(request={"config": config, "audio": audio})
            print("Response:", response)
            if response.results:
                command = response.results[0].alternatives[0].transcript.lower().strip()
                print("Command:", command)
                if command == "quit":
                    quit_process()
                elif command in command_actions:
                    command_actions[command]()
                else:
                    print("Command not recognized.")
            else:
                print("No speech recognized.")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )


def listen_for_speech():
    """Continuously listens for speech input from the microphone and initiates asynchronous processing upon detection.

    This function captures audio from the microphone using the SpeechRecognition library. While the global `running` flag is True, it listens for speech input. When speech is detected, it prints a notification, starts a new thread to process the captured audio data via the `process_audio` function, and sets the global `processing` flag to False to indicate processing has started. The listening loop can be interrupted gracefully with a KeyboardInterrupt, triggering the `quit_process` cleanup function.

    Uses global variables:
        running (bool): Controls whether the listening loop continues.
        processing (bool): Indicates whether audio processing is active.

    Returns:
        None"""
    global processing
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for speech...")
        while running:
            try:
                audio_data = recognizer.listen(source)
                if audio_data:
                    print("Speech detected!")
                    threading.Thread(target=process_audio, args=(audio_data,)).start()
                    processing = False
            except KeyboardInterrupt:
                quit_process()
                break


def start_process():
    """Starts the voice transcription process by initiating a background thread to listen for speech input.

    This function sets the global `running` flag (expected to be used elsewhere for controlling the process),
    prints an informational message to the console, and then starts a new thread running the
    `speech_recognition.listen_for_speech` function to capture and process voice input asynchronously.

    No arguments.

    Returns:
        None"""
    global running
    print("Transcription started... Press 'Ctrl + Q' to stop.")
    threading.Thread(target=listen_for_speech).start()


def stop_process():
    """Stops the ongoing transcription process by setting the global running flag to False.

    This function prints a notification indicating that transcription has been stopped and updates the global `running` variable to signal termination of the process.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription stopped.")
    running = False


start_process()
