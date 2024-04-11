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
import re
import winreg
import subprocess
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pyttsx3
engine = pyttsx3.init()
language = 'en' 
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # For a different voice
engine.setProperty('rate', 125)
shell = win32com.client.Dispatch("WScript.Shell")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0")
class Command:
    def __init__(self, text, action, target=None):
        self.text = text
        self.action = action
        self.target = target

    def execute(self):
        if self.target:
            self.action(self.target)
        else:
            self.action()
labels = ["check mail", "show time",
          "check weather", "enable bluetooth",
          "disable bluetooth", "quit", "refresh", "save", "copy",
          "paste", "delete", "search", "reload", "shut down", "verify this fact", "ask a question","search google","maximize", "minimize", "close window", "next tab",
          "previous tab","next window","previous window", "reload", "back", "forward", "scroll up", "scroll down", "zoom in", "zoom out", "reset zoom", "print","open app","screenshot"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agile-infinity-419609-0d76c5aa6ca2.json"
command_queue = []
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
        if 'scores' in result and result['scores']:
            score = result['scores'][0]
            results.append({
                'label': label,
                'score': score
            })

    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        top_label = results[0]['label']

        if top_label in command_actions:
            command = Command(transcribed_text, command_actions[top_label])
            command_queue.append(command)
        else:
            print("Command not recognized.")

        return top_label
    else:
        print("Unable to analyze intent.")
        return None
def has_sufficient_energy(audio_data):
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)  # Convert to NumPy array
    rms = np.sqrt(np.mean(audio_array**2))  # Calculate RMS
    return rms > ENERGY_THRESHOLD


def split_commands(text):
    pattern = r'(and|&|\+|,)'
    return re.split(pattern, text, flags=re.IGNORECASE)
# Define the command actions
def maximize_window():
    hwnd = win32gui.GetForegroundWindow()  # Get the active window
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    engine.say("maximising window")
    engine.runAndWait()

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
def ask(question):
    webbrowser.open(f"https://www.google.com/search?q={question}")
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
def open_target(target):
    try:
        print("1")
        # Check if the target is an installed application
        if is_program_installed(target):
            print("2")
            command = Command(target, subprocess.Popen, target)
            command.execute()
            engine.say("Opening " + target)
            print("done")
        # Check if the target is a built-in Windows application or setting
        elif is_builtin_windows_app(target):
            print("3")
            subprocess.Popen(f"explorer.exe shell:::{target}", shell=True)
            engine.say("Opening " + target)
            print("done")
        else:
            # Check if the target is a website by searching for its official website
            print("4")
            # Open Google search for the target
            print("5")
            command = Command(target, search_google, target)
            command.execute()
    except Exception as e:
        print(f"Error opening {target}: {e}")

def is_builtin_windows_app(app_name):
    try:
        key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"\\CLSID")
        with key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                clsid = winreg.EnumKey(key, i)
                try:
                    key_path = rf"\\CLSID\{clsid}\\System.ApplicationName"
                    with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path):
                        app_name_reg = winreg.QueryValueEx(key, "")[0]
                        if app_name_reg.lower() == app_name.lower():
                            return True
                except WindowsError:
                    pass
    except WindowsError:
        pass
    return False
def is_program_installed(program_name):
    uninstall_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall")

    for i in range(0, winreg.QueryInfoKey(uninstall_key)[0]):
        subkey_name = winreg.EnumKey(uninstall_key, i)
        subkey = winreg.OpenKey(uninstall_key, subkey_name)
        
        try:
            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
            if program_name.lower() in display_name.lower():
                return True
        except OSError:
            pass
        
    return False
def handle_compound_command(command):
    targets = command.split()[1:]  # Extract the targets after the "open" keyword
    actions = []
    for target in targets:
        # Check if the target is a built-in Windows application or setting
        if is_builtin_windows_app(target):
            action = lambda: subprocess.Popen(f"explorer.exe shell:::{target}", shell=True)
            actions.append(action)
        # Check if the target is an installed application
        elif is_program_installed(target):
            action = lambda: subprocess.Popen(target, shell=True)
            actions.append(action)
        else:
            print(f"Skipping unknown target: {target}")

    return actions
command_actions = {
    "check mail": open_gmail,
    "view mail": open_gmail,
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
    "screenshot": take_screenshot,
    "open app" : open_target,
    "ask a question" : ask
}
windows_apps = [
"Control Panel",
"Disk Management",
"Task Manager",
"Command Prompt",
"PowerShell",
"Registry Editor",
"Event Viewer",
"Device Manager",
"Disk Defragmenter",
"Disk Cleanup",
"System Configuration (msconfig)",
"System Information",
"System Restore",
"Update",
"Defender",
"Firewall",
"settings",
"Remote Desktop Connection",
"Hyper-V Manager",
"Sandbox",
"Terminal",
"Notepad",
"WordPad",
"Paint",
"Snipping Tool",
"Character Map",
"Sticky Notes",
"Calculator",
"Media Player",
"Photo Viewer",
"Internet Explorer",
"Microsoft Edge",
"Remote Assistance",
"Mobility Center",
"Ease of Access Center",
"Security",
"Backup and Restore",
"Management Instrumentation (WMI)",
"Performance Monitor",
"Resource Monitor",
"Subsystem for Linux (WSL)",
"PowerToys",
"Administrative Tools",
"Troubleshooting Tools",
"Remote Management (WinRM)",
"System Image Manager (SIM)"
]
def execute_command_queue():
    while command_queue:
        command = command_queue.pop(0)
        command.execute()
def get_service_website(service_name):
    query = f"{service_name} official website"
    try:
        # Perform a Google search and return the URL of the most relevant non-advertisement link
        for url in search(query, num=5, stop=5):
            # Ignore Wikipedia results
            if 'wikipedia.org' in url:
                continue

            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            ads = soup.find_all('div', class_='uEierd')
            if not ads:
                return url
    except Exception as e:
        print("An error occurred:", e)
    return None
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

            # Split the transcribed text into separate commands
            commands = split_commands(transcribed_text)

            for command in commands:
                # Check if the command starts with "open"
                words = command.split()
                if words[0] == "open":
                    # Handle compound commands like "open settings command prompt firewall"
                    actions = handle_compound_command(command)
                    for action in actions:
                        action()
                else:
                    # Perform intent analysis and add commands to the queue
                    analyze_intent(command)

            print("Response:", response)

            # Execute commands from the queue
            execute_command_queue()

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