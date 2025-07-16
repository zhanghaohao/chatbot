import logging
from colorama import Fore, Style, init


init(autoreset=True)

def alert_exception(e: Exception, message: str = None):
    """
    Print an error message to the console and log the exception
    """
    if message:
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")
    print(f"{Fore.RED}{e}{Style.RESET_ALL}")
    logging.exception(e)

def alert_info(message: str):
    """
    Print an info message to the console and log the message
    """
    print(f"{Fore.MAGENTA}{message}{Style.RESET_ALL}")
    logging.info(message)