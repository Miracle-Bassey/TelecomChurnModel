import logging  # to keep track of messages in our program
import os  #to interact with the operating system
from datetime import datetime  #to work with dates and times

# Create a log file name based on the current date and time ("2025_01_22_15_30_45.log")
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
# Create a path to the "logs" folder in the current directory, and include the log file name
logs_paths = os.path.join(os.getcwd(), "logs", LOG_FILE)
# Make sure the "logs" folder exists. If not, it creates it.
os.makedirs(logs_paths, exist_ok=True)

# This is the final file path where the log file will be saved inside the "logs" folder
LOG_FILE_PATH = os.path.join(logs_paths, LOG_FILE)

# Set up the logging configuration:
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file to the one created
    format="[ %(asctime)s ] %(filename)s %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Log messages that are  INFO, WARNING, ERROR, and CRITICAL messages.Not DEBUG
)
