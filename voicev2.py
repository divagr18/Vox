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
language = "en"
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 125)
shell = win32com.client.Dispatch("WScript.Shell")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0")


class Command:
    def __init__(self, text, action, target=None):
        """Initializes a new instance with the specified text, action, and optional target.

        Args:
            text (str): The text associated with this instance.
            action (callable): The action to be performed.
            target (optional): An optional target related to the action. Defaults to None."""
        self.text = text
        self.action = action
        self.target = target

    def execute(self):
        """Executes the stored action, optionally using the specified target.

        Calls the `action` callable with `target` as an argument if `target` is set; otherwise, calls `action` with no arguments.

        Args:
            None

        Returns:
            None"""
        if self.target:
            self.action(self.target)
        else:
            self.action()


labels = [
    "check mail",
    "show time",
    "check weather",
    "enable bluetooth",
    "disable bluetooth",
    "quit",
    "refresh",
    "save",
    "copy",
    "paste",
    "delete",
    "search",
    "reload",
    "shut down",
    "verify this fact",
    "ask a question",
    "search google",
    "maximize",
    "minimize",
    "close window",
    "next tab",
    "previous tab",
    "next window",
    "previous window",
    "reload",
    "back",
    "forward",
    "scroll up",
    "scroll down",
    "zoom in",
    "zoom out",
    "reset zoom",
    "print",
    "open app",
    "screenshot",
]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "redacted"
command_queue = []
client = speech.SpeechClient()
ENERGY_THRESHOLD = 0.1


def simple_frequency_check(audio_array):
    """Checks if the given audio signal contains frequency components typical of human speech.

    Args:
        audio_array (numpy.ndarray): 1D array representing the audio signal waveform.

    Returns:
        bool: True if any frequency component within the human speech range (100-5000 Hz)
              exceeds the speech frequency threshold (1000), indicating possible speech presence;
              False otherwise."""
    frequencies = np.fft.rfft(audio_array)
    human_speech_freq_range = 100, 5000
    speech_freq_threshold = 1000
    relevant_frequencies = frequencies[
        human_speech_freq_range[0] : human_speech_freq_range[1]
    ]
    return np.any(relevant_frequencies > speech_freq_threshold)


def analyze_intent(transcribed_text):
    """Analyzes the intent of a given transcribed text by scoring it against predefined labels.

    Args:
        transcribed_text (str): The text obtained from speech transcription to be analyzed.

    Returns:
        str or None: The label with the highest confidence score if recognized; otherwise, None.

    This function processes the input text using a pipeline with multiple labels, collects their scores,
    and identifies the most probable intent. If the top intent corresponds to a known command, it creates
    a Command object and appends it to the command queue. If no matching intent is found or scoring fails,
    it logs an appropriate message and returns None."""
    results = []
    for label in labels:
        result = pipe(transcribed_text, label)
        if "scores" in result and result["scores"]:
            score = result["scores"][0]
            results.append({"label": label, "score": score})
    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        top_label = results[0]["label"]
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
    """Determines if the given audio data has sufficient energy based on a predefined threshold.

    Args:
        audio_data: An object representing audio input, expected to have a get_raw_data() method that returns raw audio bytes.

    Returns:
        bool: True if the root mean square (RMS) energy of the audio exceeds the ENERGY_THRESHOLD constant, False otherwise."""
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)
    rms = np.sqrt(np.mean(audio_array**2))
    return rms > ENERGY_THRESHOLD


def split_commands(text):
    """Splits the input text into segments divided by command delimiters.

    Arguments:
        text (str): The input string potentially containing multiple commands separated by delimiters.

    Returns:
        list of str: A list of substrings split by delimiters 'and', '&', '+', or ',' (case-insensitive).

    This function is useful for parsing multiple commands entered as a single string, breaking them into separate components for further processing."""
    pattern = "(and|&|\\+|,)"
    return re.split(pattern, text, flags=re.IGNORECASE)


def maximize_window():
    """Maximizes the currently active window and announces the action via text-to-speech.

    This function retrieves the handle of the foreground window, maximizes it using Windows API calls,
    and then uses a speech engine to verbally confirm the window has been maximized.

    Args:
        None

    Returns:
        None"""
    hwnd = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    engine.say("maximising window")
    engine.runAndWait()


def minimize_window():
    """Minimizes the currently active foreground window.

    This function retrieves the handle of the foreground window and sends a command to minimize it.

    Args:
        None

    Returns:
        None"""
    hwnd = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)


def close_window():
    """Closes the currently active window by sending the Alt+F4 keyboard shortcut.

    Uses the shell.SendKeys method to simulate pressing Alt+F4, which is the standard shortcut for closing windows in Windows environments.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{F4}")


def switch_to_next_tab():
    """Switches to the next tab in the current application by sending the Ctrl+Tab keyboard shortcut.

    This function uses the shell interface to simulate the keypress combination for navigating to the next tab, commonly supported by tabbed user interfaces.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^{TAB}")


def switch_to_previous_tab():
    """Switches the active window focus to the previous browser or application tab.

    Emulates the keyboard shortcut Ctrl+Shift+Tab by sending the corresponding key combination,
    which is commonly used in many applications and browsers to move to the tab immediately
    to the left of the current tab.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^+{TAB}")


def switch_to_next_window():
    """Switches the active window to the next window in the current application.

    Sends the keyboard shortcut Ctrl + Tab to cycle forward through open windows or tabs.
    This function assumes `shell` is an initialized object capable of sending keyboard input.

    Returns:
        None"""
    shell.SendKeys("^{TAB}")


def switch_to_previous_window():
    """Switches focus to the previously active window using keyboard shortcut.

    Sends the 'Ctrl+Shift+Tab' key combination to simulate switching to the previous window
    in the current application or window manager context. Assumes `shell` is an initialized
    COM shell object capable of sending keyboard inputs.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^+{TAB}")


def reload_page():
    """Reloads the current page by sending the Ctrl+R keyboard shortcut.

    This function simulates the keypress for Ctrl+R via the shell interface,
    commonly used to refresh or reload the active page in many applications.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^R")


def go_back():
    """Simulates pressing the Alt + Left Arrow key combination to navigate back.

    This function sends the '%{LEFT}' key sequence using the shell interface,
    which typically triggers a "go back" action in many applications, such as
    web browsers or file explorers.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{LEFT}")


def go_forward():
    """Simulates pressing the Alt + Right Arrow key combination.

    This function sends the key sequence to the active window to trigger the system or application behavior associated with Alt+Right Arrow (commonly used for navigating forward in browsers or file explorers).

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{RIGHT}")


def scroll_up():
    """Scrolls the screen up by a predefined amount.

    This function uses the pyautogui library to scroll the mouse wheel upwards by 100 units.

    Args:
        None

    Returns:
        None"""
    pyautogui.scroll(100)


def scroll_down():
    """Scrolls the screen down by 100 units using the pyautogui library.

    This function performs a downward scroll action equivalent to moving the mouse wheel
    down by 100 units. It does not take any arguments and does not return a value.

    Args:
        None

    Returns:
        None"""
    pyautogui.scroll(-100)


def zoom_in():
    """Simulates the keyboard shortcut for zooming in by sending Ctrl and '+' key presses.

    This function uses the shell interface to send the Ctrl+'+' key combination, which is commonly used to increase zoom level in many applications.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^+")


def zoom_out():
    """Simulates the keyboard shortcut to zoom out in the active application.

    Sends the Ctrl and '-' key combination via the shell interface to decrease the zoom level.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^-")


def reset_zoom():
    """Resets the zoom level to the default setting by sending the keyboard shortcut Ctrl+0.

    This function simulates the key press combination Ctrl+0 to reset zoom, typically in applications where this shortcut resets the view scale. It uses the `shell.SendKeys` method to send the keys.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^0")


def print_document():
    """Simulates pressing Ctrl+P to open the print dialog.

    This function sends the Ctrl+P keyboard shortcut using the shell interface to trigger the print command in the active application.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^P")


def take_screenshot():
    """Takes a screenshot of the current screen and saves it as 'screenshot.png' in the working directory.

    This function captures the entire screen using the pyautogui library and stores the image as a PNG file named 'screenshot.png'.

    Args:
        None

    Returns:
        None"""
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")


def enable_bluetooth():
    """Simulates pressing the F15 key to enable Bluetooth via a Windows shell command.

    This function uses the Windows Script Host shell interface to send the F15 keypress,
    which is assumed to trigger Bluetooth enabling on the system.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F15}")


def search_google(query):
    """Performs a Google search for the given query using a Chrome WebDriver instance.

    Args:
        query (str): The search query string to be entered into Google's search box.

    Returns:
        None

    This function initializes a new Chrome WebDriver, navigates to Google's homepage,
    and attempts to perform a search by sending the query text followed by the RETURN key.
    If any error occurs during the interaction with the web elements, it prints an error message."""
    driver = webdriver.Chrome()
    driver.get("https://www.google.com")
    try:
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
    except Exception as e:
        print(f"An error occurred during the search: {e}")


def ask(question):
    """Open a web browser to perform a Google search for the given question.

    Args:
        question (str): The query string to search on Google.

    Returns:
        None

    This function launches the default web browser and directs it to a Google search results page
    for the specified question. It does not return any value."""
    webbrowser.open(f"https://www.google.com/search?q={question}")


def disable_bluetooth():
    """Simulates pressing the F16 key to disable Bluetooth on Windows.

    This function uses the Windows Script Host shell interface to send a
    key press event for the F16 key, which is assumed to toggle or disable
    Bluetooth on the target system.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F16}")


def open_gmail():
    """Open the Gmail inbox in the default web browser.

    This function launches the user's default web browser and navigates to the Gmail web interface.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://mail.google.com")


def open_weather():
    """Open the Weather Channel website in the default web browser.

    This function launches the user's default web browser and navigates to the Weather Channel's homepage.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://www.weather.com")


def quit_process():
    """Stops the running process by setting control flags to False.

    This function sets the global flags `running` and `processing` to False to signal that the process should terminate. It also prints a status message indicating the quit operation.

    Args:
        None

    Returns:
        None"""
    global running
    print("Quitting process...")
    running = False
    processing = False


def open_target(target):
    """Attempts to open the specified target using an appropriate method.

    Args:
        target (str): The name or identifier of the target to open. This can be a program name,
            a built-in Windows app identifier, or a general search term.

    Returns:
        None

    The function checks if the target corresponds to an installed program, a built-in Windows app,
    or neither. It then executes the appropriate command to open or search for the target,
    providing voice feedback via the engine. Exceptions are caught and logged to the console."""
    try:
        print("1")
        if is_program_installed(target):
            print("2")
            command = Command(target, subprocess.Popen, target)
            command.execute()
            engine.say("Opening " + target)
            print("done")
        elif is_builtin_windows_app(target):
            print("3")
            subprocess.Popen(f"explorer.exe shell:::{target}", shell=True)
            engine.say("Opening " + target)
            print("done")
        else:
            print("4")
            print("5")
            command = Command(target, search_google, target)
            command.execute()
    except Exception as e:
        print(f"Error opening {target}: {e}")


def is_builtin_windows_app(app_name):
    """Determines whether a given application name corresponds to a built-in Windows app by checking the CLSID registry entries.

    Args:
        app_name (str): The name of the application to check.

    Returns:
        bool: True if the application is a built-in Windows app; False otherwise.

    This function queries the Windows registry under HKEY_CLASSES_ROOT\\CLSID for entries that have a
    System.ApplicationName value matching the provided app_name (case-insensitive). It handles registry
    access errors gracefully and returns False if the application is not found or access is denied."""
    try:
        key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "\\\\CLSID")
        with key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                clsid = winreg.EnumKey(key, i)
                try:
                    key_path = f"\\\\CLSID\\{clsid}\\\\System.ApplicationName"
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
    """Checks if a program with the given name is installed on the Windows system by scanning the uninstall registry keys.

    Args:
        program_name (str): The name (or partial name) of the program to check for.

    Returns:
        bool: True if a matching installed program is found, False otherwise.

    This function queries the `HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall` registry key to identify installed programs
    based on their "DisplayName". It performs a case-insensitive substring match to determine if the specified program is installed."""
    uninstall_key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
    )
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
    """Generate a list of actions to execute each component of a compound voice command.

    Args:
        command (str): A voice command string where the first word is the compound command
            identifier and subsequent words specify target applications or shell items.

    Returns:
        List[Callable[[], None]]: A list of zero-argument functions. When called, each
        function executes one of the recognized targets by launching a built-in Windows
        app via its shell identifier or an installed program via subprocess. Targets
        not recognized are skipped with a printed warning."""
    targets = command.split()[1:]
    actions = []
    for target in targets:
        if is_builtin_windows_app(target):
            action = lambda: subprocess.Popen(
                f"explorer.exe shell:::{target}", shell=True
            )
            actions.append(action)
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
    "open app": open_target,
    "ask a question": ask,
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
    "System Image Manager (SIM)",
]


def execute_command_queue():
    """Execute all commands in the global command queue sequentially.

    This function processes each command in the `command_queue` list by removing it
    from the front of the queue and invoking its `execute` method. This continues until
    the queue is empty. It assumes that each item in `command_queue` has an `execute`
    method and that `command_queue` is defined in the global scope of the `voicev2.py` module.

    Args:
        None

    Returns:
        None"""
    while command_queue:
        command = command_queue.pop(0)
        command.execute()


def get_service_website(service_name):
    """Retrieve the most likely official website URL for a given service by performing a web search.

    Args:
        service_name (str): The name of the service to find the official website for.

    Returns:
        str or None: The URL of the service's official website if found and verified (excluding Wikipedia and pages with certain ads), otherwise None."""
    query = f"{service_name} official website"
    try:
        for url in search(query, num=5, stop=5):
            if "wikipedia.org" in url:
                continue
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            ads = soup.find_all("div", class_="uEierd")
            if not ads:
                return url
    except Exception as e:
        print("An error occurred:", e)
    return None


running = True
processing = False


def process_audio(recognizer, audio_data):
    """Processes raw audio data for speech recognition, analyzes recognized commands, and executes corresponding actions.

    Args:
        recognizer: An instance of the speech recognizer used for processing audio.
        audio_data: Audio data to be processed, typically obtained from a microphone or audio source.

    Returns:
        None. The function performs side effects including printing messages, executing commands, and updating global state."""
    global running, processing
    if not running:
        return
    try:
        audio = speech.RecognitionAudio(
            content=audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        )
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = client.recognize(config=config, audio=audio)
        print("Speech Detected!")
        if response.results:
            transcribed_text = (
                response.results[0].alternatives[0].transcript.lower().strip()
            )
            print("Recognized Speech:", transcribed_text)
            commands = split_commands(transcribed_text)
            for command in commands:
                words = command.split()
                if words[0] == "open":
                    actions = handle_compound_command(command)
                    for action in actions:
                        action()
                else:
                    analyze_intent(command)
            print("Response:", response)
            execute_command_queue()
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
    except Exception as e:
        print("Error processing audio:", e)
    finally:
        processing = False


def listen_for_speech():
    """Continuously listens for speech from the microphone and initiates processing of detected audio.

    Monitors microphone input in real-time, adjusting for ambient noise and capturing audio data. When audio with sufficient energy and passing a frequency check is detected, it spawns a separate thread to process the audio, ensuring asynchronous handling. The function runs until the global `running` flag is set to False. It also gracefully handles user interruption (KeyboardInterrupt) by stopping the listening loop.

    Args:
        None

    Returns:
        None"""
    global processing, running
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while running:
            try:
                audio_data = recognizer.listen(source)
                audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)
                if has_sufficient_energy(audio_data) and simple_frequency_check(
                    audio_array
                ):
                    if not processing:
                        processing = True
                        threading.Thread(
                            target=process_audio, args=(recognizer, audio_data)
                        ).start()
                else:
                    print("not strong enough")
            except KeyboardInterrupt:
                print("User initiated shutdown...")
                running = False
                break


def start_process():
    """Starts the transcription process in a new thread and informs the user how to stop it.

    This function sets the global running state, prints an informational message to the console,
    and launches the speech listening functionality asynchronously.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription started... Press 'Ctrl + Q' to stop.")
    threading.Thread(target=listen_for_speech).start()


def stop_process():
    """Stops the ongoing transcription process by setting the global running flag to False.

    This function outputs a notification message indicating that transcription has been stopped
    and updates the global `running` variable to signal the termination of the transcription loop.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription stopped.")
    running = False


start_process()
