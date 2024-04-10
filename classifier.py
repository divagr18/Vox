import winreg

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

program_name = "whatsapp"
if is_program_installed(program_name):
    print(f"{program_name} is installed.")
else:
    print(f"{program_name} is not installed.")
