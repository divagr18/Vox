import speech_recognition as sr
import pyautogui
import time
import numpy as np
import webbrowser
import win32gui
import win32com.client
import win32con
from google.cloud import speech
import os
import threading
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
shell = win32com.client.Dispatch("WScript.Shell")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0")
labels = ["check mail", "show time",
          "check weather", "enable bluetooth",
          "disable bluetooth", "quit", "refresh", "save", "copy",
          "paste", "delete", "search", "reload", "shut down", "verify this fact", "search google","maximize", "minimize", "close window", "next tab",
          "previous tab","next window","previous window", "reload", "back", "forward", "scroll up", "scroll down", "zoom in", "zoom out", "reset zoom", "print", "screenshot"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agile-infinity-419609-0d76c5aa6ca2.json"
client = speech.SpeechClient()
ENERGY_THRESHOLD = 0.1
def simple_frequency_check(audio_array):    
    frequencies = np.fft.rfft(audio_array)
    human_speech_freq_range = (100, 5000) 
    speech_freq_threshold = 1000 
    relevant_frequencies = frequencies[human_speech_freq_range[0]:human_speech_freq_range[1]] # Correct slicing
    return np.any(relevant_frequencies > speech_freq_threshold)

def analyze_intent(transcribed_text):
    results = []
    for label in labels:
        result = pipe(transcribed_text, label)
        results.append({
            'label': label,
            'score': result['scores'][0]
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    top_label = results[0]['label']
    return top_label
def has_sufficient_energy(audio_data):
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)  # Convert to NumPy array
    rms = np.sqrt(np.mean(audio_array**2))  # Calculate RMS
    return rms > ENERGY_THRESHOLD
# Define the command actions
def maximize_window():
    hwnd = win32gui.GetForegroundWindow()  # Get the active window
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

def minimize_window():
    hwnd = win32gui.GetForegroundWindow()  
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

def close_window(): 
    shell.SendKeys('%{F4}')  # Alt + F4

def switch_to_next_tab():
    shell.SendKeys('^{TAB}')  # Ctrl + Tab

def switch_to_previous_tab():
    shell.SendKeys('^+{TAB}')  # Ctrl + Shift + Tab
def switch_to_next_window():
    shell.SendKeys('^{TAB}')  # Ctrl + Tab

def switch_to_previous_window():
    shell.SendKeys('^+{TAB}')  # Ctrl + Shift + Tab

def reload_page():
    shell.SendKeys('^R')  # Ctrl + R

def go_back():
    shell.SendKeys('%{LEFT}')  # Alt + Left Arrow

def go_forward():
    shell.SendKeys('%{RIGHT}')  # Alt + Right Arrow

def scroll_up():
    pyautogui.scroll(100) 

def scroll_down():
    pyautogui.scroll(-100)

def zoom_in():
    shell.SendKeys('^+')  # Ctrl + '+'

def zoom_out():
    shell.SendKeys('^-')  # Ctrl + '-'

def reset_zoom():
    shell.SendKeys('^0')  # Ctrl + 0

def print_document():
    shell.SendKeys('^P')  # Ctrl + P

def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
def enable_bluetooth():
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys('{F15}')
def search_google(query):
    driver = webdriver.Chrome()
    driver.get("https://www.google.com")

    try:
        search_box = driver.find_element(By.NAME, "q")

        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

    except Exception as e:
        print(f"An error occurred during the search: {e}")

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
    "stop": quit_process,
    "search google": search_google,
    "maximize": maximize_window,
    "minimize": minimize_window,
    "close window": close_window, 
    "next tab": switch_to_next_tab,
    "previous tab": switch_to_previous_tab,
    "next window": switch_to_next_window,
    "previous window": switch_to_previous_window,
    "reload": reload_page,
    "back": go_back,
    "forward": go_forward,
    "scroll up": scroll_up,
    "scroll down": scroll_down,
    "zoom in": zoom_in,
    "zoom out": zoom_out,
    "reset zoom": reset_zoom,
    "print": print_document,
    "screenshot": take_screenshot
}

# Variable to control the loop
running = True
processing = False  # No need for a separate flag

# Consolidated audio processing with Google Speech-to-Text v2
def process_audio(recognizer, audio_data):
    global running, processing

    if not running:
        return

    try:
        audio = speech.RecognitionAudio(content=audio_data.get_raw_data(convert_rate=16000, convert_width=2))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        print("Speech Detected!")

        if response.results:
            transcribed_text = response.results[0].alternatives[0].transcript.lower().strip()
            print("Recognized Speech:", transcribed_text)

            # Perform intent analysis
            intent = analyze_intent(transcribed_text)
            print(f"Detected Label: {intent}")  # Print the detected label

            if intent in command_actions:
                if intent == "search google":
                    search_query = transcribed_text.replace("search google", "").strip()
                    command_actions[intent](search_query)
                else:
                    command_actions[intent]()
            else:
                print("Command not recognized.")

            print("Response:", response)

        else:
            print("No speech recognized.")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        print("Error processing audio:", e)
    finally:
        processing = False

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