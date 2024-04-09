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

# Initialize PyAudio for audio input
audio = pyaudio.PyAudio()

# Initialize the Speech client with service account credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agile-infinity-419609-0d76c5aa6ca2.json"
client = speech.SpeechClient()

# Define the command actions
def enable_bluetooth():
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys('{F15}')

def disable_bluetooth():
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys('{F16}')

def open_gmail():
    webbrowser.open("https://mail.google.com")

def open_weather():
    webbrowser.open("https://www.weather.com")

def quit_process():
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
    "stop": quit_process
}

# Variable to control the loop
running = True

# Flag to indicate if processing should continue
processing = True

# Function to process audio
def process_audio(audio_data):
    global processing
    try:
        if processing:
            audio_content = audio_data.get_raw_data(convert_rate=16000, convert_width=2)  # Get raw audio data with correct format
            audio_content_base64 = base64.b64encode(audio_content).decode("utf-8")  # Encode audio data to base64
            audio = speech.RecognitionAudio(content=audio_content_base64)  # Create RecognitionAudio object
            config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code="en-US")  # Create RecognitionConfig object
            response = client.recognize(request={"config": config, "audio": audio})  # Send request to Google Speech-to-Text API
            
            print("Response:", response)  # Print response for debugging
            
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
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Main loop
def listen_for_speech():
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
                    processing = False  # Stop processing after the first speech is detected
            except KeyboardInterrupt:
                quit_process()
                break

# Function to start audio processing
def start_process():
    global running
    print("Transcription started... Press 'Ctrl + Q' to stop.")
    threading.Thread(target=listen_for_speech).start()

# Function to stop audio processing
def stop_process():
    global running
    print("Transcription stopped.")
    running = False

# Start the process
start_process()