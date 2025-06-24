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
        """Initializes a new instance with given text, action, and optional target.

        Args:
            text (str): The text associated with this instance.
            action (Any): The action to be performed or represented.
            target (Optional[Any], optional): An optional target related to the action. Defaults to None."""
        self.text = text
        self.action = action
        self.target = target

    def execute(self):
        """Executes the stored action, optionally using the specified target.

        If a target is set, calls the action with the target as an argument; otherwise,
        calls the action without any arguments.

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
    """Checks for the presence of significant frequency components within the typical human speech range in an audio signal.

    Args:
        audio_array (np.ndarray): 1D numpy array containing the audio signal samples.

    Returns:
        bool: True if any frequency component in the range 100 Hz to 5000 Hz exceeds the threshold magnitude of 1000, indicating likely speech presence; False otherwise."""
    frequencies = np.fft.rfft(audio_array)
    human_speech_freq_range = 100, 5000
    speech_freq_threshold = 1000
    relevant_frequencies = frequencies[
        human_speech_freq_range[0] : human_speech_freq_range[1]
    ]
    return np.any(relevant_frequencies > speech_freq_threshold)


def analyze_intent(transcribed_text):
    """Analyzes the intent of a transcribed text by scoring it against predefined labels.

    Args:
        transcribed_text (str): The text input obtained from speech transcription to analyze.

    Returns:
        str or None: The label with the highest confidence score if recognized; otherwise, None.

    This function uses an external pipeline (`pipe`) to score the input text against each label in `labels`.
    If the top scoring label matches an entry in `command_actions`, a corresponding `Command` is
    created and appended to the global `command_queue`. If no label is confidently matched,
    or no scores are returned, the function returns None and prints a message indicating failure."""
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
    """Determines if the provided audio data has sufficient energy based on a predefined threshold.

    Args:
        audio_data: An object with a `get_raw_data()` method that returns raw audio bytes.

    Returns:
        bool: True if the root mean square (RMS) energy of the audio exceeds the ENERGY_THRESHOLD; otherwise, False."""
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)
    rms = np.sqrt(np.mean(audio_array**2))
    return rms > ENERGY_THRESHOLD


def split_commands(text):
    """Splits the input text into a list of substrings divided by common command separators.

    Args:
        text (str): The input string containing one or more commands separated by delimiters.

    Returns:
        list of str: A list of substrings split at occurrences of 'and', '&', '+', or ',' (case-insensitive),
        including the delimiters themselves as separate elements."""
    pattern = "(and|&|\\+|,)"
    return re.split(pattern, text, flags=re.IGNORECASE)


def maximize_window():
    """Maximizes the currently active window and announces the action using speech synthesis.

    This function retrieves the handle of the foreground window, maximizes it using the Windows API,
    and uses a speech engine to audibly confirm the window has been maximized.

    Args:
        None

    Returns:
        None"""
    hwnd = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    engine.say("maximising window")
    engine.runAndWait()


def minimize_window():
    """Minimize the currently active (foreground) window.

    This function retrieves the handle of the foreground window and minimizes it using the Windows API.

    Args:
        None

    Returns:
        None"""
    hwnd = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)


def close_window():
    """Closes the currently active window by sending the Alt+F4 keystroke sequence.

    This function uses the Windows shell to simulate the keyboard shortcut for closing the active window.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{F4}")


def switch_to_next_tab():
    """Sends the keyboard shortcut to switch to the next tab in the active application.

    Uses the shell object to send the Ctrl+Tab key combination, which typically switches to the next tab in tabbed interfaces such as web browsers or editors.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^{TAB}")


def switch_to_previous_tab():
    """Simulates the keyboard shortcut to switch to the previous tab in most tabbed interfaces.

    Sends the Ctrl+Shift+Tab key combination to the active window using the shell interface.

    Returns:
        None"""
    shell.SendKeys("^+{TAB}")


def switch_to_next_window():
    """Switches focus to the next window or tab by simulating the Ctrl+Tab key press.

    This function sends the Ctrl+Tab keyboard shortcut to the active shell, which typically cycles
    through open tabs or windows in most applications and browsers.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^{TAB}")


def switch_to_previous_window():
    """Simulates the keyboard shortcut to switch to the previously active window.

    Sends the Ctrl+Shift+Tab key combination using the shell interface to navigate
    backwards through open windows or tabs in the current application or environment.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^+{TAB}")


def reload_page():
    """Reloads the current page by sending the Ctrl+R keyboard shortcut.

    This function uses the shell object to simulate pressing Ctrl+R, which triggers a page reload in most web browsers.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^R")


def go_back():
    """Simulates pressing the 'Alt + Left Arrow' key combination to navigate back.

    This function sends the keystroke sequence corresponding to 'Alt + Left Arrow' using the shell interface, typically used to navigate back in file explorers or web browsers.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{LEFT}")


def go_forward():
    """Simulates the Alt + Right Arrow key press to trigger the "go forward" action.

    Sends the keyboard shortcut '%{RIGHT}' via the shell interface, which typically navigates forward in applications such as web browsers or file explorers.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("%{RIGHT}")


def scroll_up():
    """Scrolls the screen content upward by a fixed amount.

    This function simulates a mouse wheel scroll up action by 100 units using pyautogui.

    Args:
        None

    Returns:
        None"""
    pyautogui.scroll(100)


def scroll_down():
    """Scrolls the screen down by 100 units using the mouse wheel.

    This function uses pyautogui to simulate a downward scroll action.

    Returns:
        None"""
    pyautogui.scroll(-100)


def zoom_in():
    """Simulates a zoom-in keyboard shortcut by sending Ctrl and '+' keys.

    This function uses the shell interface to send the Ctrl+'+' key combination,
    commonly used to zoom in on content in many applications.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^+")


def zoom_out():
    """Simulates the keyboard shortcut for zooming out by sending Ctrl + '-' keypresses.

    This function uses the shell interface to send the key combination that typically decreases the zoom level in many applications supporting standard shortcuts.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^-")


def reset_zoom():
    """Resets the zoom level to the default setting by sending the Ctrl+0 keyboard shortcut.

    This function simulates pressing the Control key and the '0' key simultaneously, a common shortcut for resetting zoom in many applications.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^0")


def print_document():
    """Simulates pressing Ctrl+P to open the print dialog.

    This function sends the Ctrl+P keyboard shortcut to the active window using the shell's SendKeys method, typically triggering the print command in most applications.

    Args:
        None

    Returns:
        None"""
    shell.SendKeys("^P")


def take_screenshot():
    """Captures the current screen and saves the image as 'screenshot.png' in the working directory.

    This function uses the pyautogui library to take a screenshot of the entire screen.

    Args:
        None

    Returns:
        None"""
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")


def enable_bluetooth():
    """Simulates pressing the F15 key to enable Bluetooth via a Windows shell command.

    This function uses the Windows Scripting Host shell to send a simulated key press event for the F15 key, which is assumed to trigger enabling Bluetooth on the system.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F15}")


def search_google(query):
    """Performs a Google search for the specified query using a Chrome WebDriver instance.

    Args:
        query (str): The search query string to be entered into Google Search.

    Returns:
        None

    This function initializes a Chrome WebDriver, navigates to the Google homepage,
    and attempts to locate the search input box by name attribute 'q'. It then enters
    the provided query and submits the search. Any exceptions during the search process
    are caught and printed. The WebDriver is not closed automatically within this function."""
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
        question (str): The query string to search for on Google.

    Returns:
        None

    This function constructs a Google search URL using the provided question and opens it in the default web browser."""
    webbrowser.open(f"https://www.google.com/search?q={question}")


def disable_bluetooth():
    """Simulates pressing the F16 key to disable Bluetooth via a system-level shortcut.

    This function uses the Windows Script Host shell interface to send a keystroke event corresponding to the F16 key, which is assumed to toggle the Bluetooth state on supported systems or applications.

    Args:
        None

    Returns:
        None"""
    bt_shell = win32com.client.Dispatch("WScript.Shell")
    bt_shell.SendKeys("{F16}")


def open_gmail():
    """Opens the Gmail inbox in the default web browser.

    This function uses the webbrowser module to launch the Gmail website,
    allowing the user to access their email directly.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://mail.google.com")


def open_weather():
    """Opens the default web browser to the Weather Channel website.

    This function launches the user's default web browser and navigates to https://www.weather.com,
    providing quick access to weather information.

    Args:
        None

    Returns:
        None"""
    webbrowser.open("https://www.weather.com")


def quit_process():
    """Stops the ongoing process by setting control flags to False and prints a quitting message.

    This function modifies the global variables `running` and `processing` to signal the termination of the current process.

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

    If the target corresponds to an installed program, it executes the program using a Command instance with subprocess.Popen.
    If the target is recognized as a built-in Windows app identifier, it launches the app via explorer.exe shell command.
    Otherwise, it performs a Google search for the target using a Command instance with the search_google function.
    Speech feedback is provided via the engine.say method when opening known programs or apps.
    Any exceptions during the process are caught and logged to the console.

    Args:
        target (str): The name or identifier of the program, built-in app, or search query to open.

    Returns:
        None"""
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
    """Checks if the specified application name corresponds to a built-in Windows app by querying the Windows registry.

    Args:
        app_name (str): The name of the application to verify.

    Returns:
        bool: True if the application is a built-in Windows app, False otherwise.

    Note:
        This function accesses the Windows registry under HKEY_CLASSES_ROOT\\CLSID to find CLSIDs with a
        System.ApplicationName value matching the given app_name. It requires appropriate permissions
        to read the registry and only runs correctly on Windows systems."""
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
    """Checks if a specified program is installed on the local Windows machine by searching the system registry's uninstall entries.

    Args:
        program_name (str): The name (or partial name) of the program to search for in the installed applications list.

    Returns:
        bool: True if the program is found in the uninstall registry keys, False otherwise.

    Note:
        This function accesses the Windows registry under HKEY_LOCAL_MACHINE and assumes it is run on a Windows platform with sufficient permissions."""
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
    """Parses a compound command string and returns a list of actions to execute for each recognized target.

    Args:
        command (str): A compound command string where the first word is the command keyword and subsequent words are target identifiers.

    Returns:
        list of callable: A list of zero-argument functions, each launching a Windows built-in app or an installed program corresponding to a target. Targets that are unrecognized are skipped with a printed warning."""
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
    """Executes all commands in the global command_queue in FIFO order.

    Processes each command by removing it from the front of the command_queue and invoking its execute() method until the queue is empty.

    Args:
        None

    Returns:
        None"""
    while command_queue:
        command = command_queue.pop(0)
        command.execute()


def get_service_website(service_name):
    """Retrieve the official website URL for a given service by performing a web search.

    Args:
        service_name (str): The name of the service to look up.

    Returns:
        str or None: The URL of the service's official website if found and verified by absence of specific ad elements, otherwise None.

    This function performs a web search for the service's official website, filters out Wikipedia links,
    and tries to identify the official site by checking the page content for certain ad-related HTML elements.
    If an error occurs during the process, it prints the error and returns None."""
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
    """Processes audio data from the recognizer, performs speech recognition, and executes commands based on the transcribed text.

    Args:
        recognizer: The speech recognizer instance used for processing the audio.
        audio_data: An audio data object containing the raw audio to be transcribed.

    Returns:
        None. This function processes the audio input asynchronously, printing recognition results and executing commands as detected.

    Raises:
        Handles and logs exceptions related to speech recognition failures and other processing errors internally.

    Note:
        This function relies on global flags `running` and `processing` to control execution flow and modifies the `processing` flag to indicate completion.
        It uses an external speech recognition client (`client`), and helper functions like `split_commands`, `handle_compound_command`, `analyze_intent`, and `execute_command_queue` to process recognized speech into actionable commands."""
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
    """Continuously listens for speech input from the microphone and triggers audio processing on detected speech.

    This function uses the SpeechRecognition library to capture audio through the microphone, adjusts for ambient noise,
    and enters a loop where it listens for speech as long as the global `running` flag is True. If the captured audio
    meets predefined energy and frequency criteria (via `has_sufficient_energy` and `simple_frequency_check`),
    it starts a new thread to process the audio asynchronously, provided that no other processing is currently underway
    (as indicated by the global `processing` flag).

    It handles KeyboardInterrupt to gracefully stop the listening loop and cleanup.

    Globals:
        processing (bool): Indicates if audio processing is currently happening.
        running (bool): Controls the main listening loop; set to False to stop listening.

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
    """Starts the transcription process by initiating a background thread to listen for speech input.

    Prints a message indicating that transcription has started and instructions for stopping it.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription started... Press 'Ctrl + Q' to stop.")
    threading.Thread(target=listen_for_speech).start()


def stop_process():
    """Stops the ongoing transcription process.

    Sets the global `running` flag to False to indicate that transcription should cease, and prints a confirmation message to the console.

    Args:
        None

    Returns:
        None"""
    global running
    print("Transcription stopped.")
    running = False


start_process()
