This project consists of two Python programs and a requirements.txt file specifying the requirements, that work together to detect emergency sounds, log them into a database, and visualize the results on an interactive dashboard.

Prerequisites
Install the latest version of Python 3.10+
Clone this github folder through the command prompt, or you can also download this folder

Installation
Open Command Prompt (Windows) or Terminal (Linux/Mac) and install the required Python libraries which are given in the requirements.txt file:
pip install requirements.txt



Running the Programs
1. Start the MQTT Logger
This script connects to the HiveMQ Cloud broker and logs detected sound events into the SQLite database.
python mqtt_loger.py
If successful, you should see:
Connected to HiveMQ Cloud
Keep this window open and running.

2. Launch the Dashboard
Open a new terminal window (do not close the logger) and run:
streamlit run emergency_dashboard_final.py
This will open a dashboard in your browser at:
http://localhost:8501
