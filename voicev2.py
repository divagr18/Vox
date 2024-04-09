import speech_recognition as sr
import pyautogui
import time
import numpy as np
import webbrowser
import win32com.client
from google.cloud import speech
import os
import threading

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agile-infinity-419609-0d76c5aa6ca2.json"
client = speech.SpeechClient()
ENERGY_THRESHOLD = 0.1
def simple_frequency_check(audio_array):    
    frequencies = np.fft.rfft(audio_array)
    human_speech_freq_range = (100, 5000) 
    speech_freq_threshold = 1000 
    relevant_frequencies = frequencies[human_speech_freq_range[0]:human_speech_freq_range[1]] # Correct slicing
    return np.any(relevant_frequencies > speech_freq_threshold)


def has_sufficient_energy(audio_data):
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)  # Convert to NumPy array
    rms = np.sqrt(np.mean(audio_array**2))  # Calculate RMS
    return rms > ENERGY_THRESHOLD
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
    processing = False

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
processing = False  # No need for a separate flag

# Consolidated audio processing with Google Speech-to-Text v2
def process_audio(recognizer, audio_data):
    global running, processing   # Access global variables

    if not running:  # Check the 'running' flag before proceeding
        return
    try:
        audio = speech.RecognitionAudio(content=audio_data.get_raw_data(convert_rate=16000, convert_width=2))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        print("Speech Detected!")  # Always indicate if speech was detected

        if response.results:
            command = response.results[0].alternatives[0].transcript.lower().strip()
            print("Recognized Speech:", command) 

            if command in command_actions:
                command_actions[command]()
            else:
                print("Command not recognized.") 

            print("Response:", response)  # Print the raw response from Google

        else:
            print("No speech recognized.")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:  # Catch more general exceptions
        print("Error processing audio:", e)
    finally:
        # Allow processing of future commands
        global processing
        processing = False
# Main listening loop
def listen_for_speech():
    global processing, running  # Make 'running' accessible
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Adjust for 0.5 seconds
        while running:
            try:
                audio_data = recognizer.listen(source)
                audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)  # Define audio_array
                if has_sufficient_energy(audio_data) and simple_frequency_check(audio_array):
                    if not processing:
                        processing = True
                        threading.Thread(target=process_audio, args=(recognizer, audio_data)).start()
                else:
                    print("not strong enough")

            except KeyboardInterrupt:
                print("User initiated shutdown...")  # More informative message
                running = False  # Stop the listening loop directly
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