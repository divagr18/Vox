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
    """Simulates pressing the F15 key to enable Bluetooth using the Windows Scripting Host.

    This function leverages the Windows COM interface to send a virtual keypress (F15), which is assumed to toggle Bluetooth on supported systems.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F15}")


def disable_bluetooth():
    """Simulates pressing the F16 key to disable Bluetooth on a Windows system.

    This function uses the Windows Script Host COM interface to send the F16 keypress,
    which is assumed to be mapped to disable Bluetooth hardware. It requires the
    `pywin32` package and appropriate system permissions.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F16}")


def open_gmail():
    """Open Gmail in the default web browser.

    This function launches the user's default web browser and navigates to the Gmail inbox.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://mail.google.com")


def open_weather():
    """Open the default web browser to the Weather Channel website.

    This function launches the system's default web browser and navigates to https://www.weather.com
    to provide the user with current weather information.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://www.weather.com")


def quit_process():
    """Stops the running process by setting the global running flag to False.

    This function modifies the global 'running' variable to signal the termination of the ongoing process.
    It also prints a message indicating that the process is quitting.

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
    """Processes raw audio data through Google Speech-to-Text API and executes associated commands.

    Args:
        audio_data (AudioData): An AudioData instance containing the audio to be processed.

    Returns:
        None

    This function checks if processing is enabled globally. If so, it converts the audio to the required format,
    encodes it in base64, and sends it to the Google Speech-to-Text API for transcription. Based on the recognized
    transcript, it triggers predefined command actions or quits the process if the "quit" command is detected.
    Errors during recognition or API requests are caught and logged. The function is designed to be used within a
    voice recognition context where global flags and command mappings are defined externally."""
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
    """Continuously listens for speech input from the microphone and processes the first detected speech asynchronously.

    This function initializes a speech recognizer and opens the microphone for input. It enters a loop that runs while the global `running` flag is True, listening for audio. Upon detecting speech, it starts a new thread to process the captured audio using the `process_audio` function and sets the global `processing` flag to False to stop further processing after the first detection. The loop can be interrupted gracefully with a KeyboardInterrupt, which triggers the `quit_process` cleanup function.

    Args:
        None

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
    """Starts the voice transcription process in a separate thread.

    This function sets the global running state and initiates the speech recognition
    by launching the listen_for_speech function in a new thread. The user is informed
    that transcription has started and how to stop it.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription started... Press 'Ctrl + Q' to stop.")
    threading.Thread(target=listen_for_speech).start()


def stop_process():
    """Stops the voice transcription process by setting the running flag to False.

    This function prints a message indicating that transcription has been stopped and updates the global `running` variable to signal the termination of the ongoing voice recognition process.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription stopped.")
    running = False


start_process()
