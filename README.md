  **Real-Time Human Distress Sound Event Detection using Edge AI**

This project provides tools for detecting **emergency sounds** using Edge AI, logging the events into a database, and visualizing the results in an interactive dashboard. The repository also includes a pre-trained audio classifier, sample datasets, and web-based audio analysis utilities.

---

##  Project Structure

```
Real-Time-Human-Distress-Sound-Event-Detection-using-Edge-AI-main
├── MAX-Audio-Classifier         # Pre-trained audio classifier based on VGGish
│   ├── api                      # API definition for prediction
│   ├── core                     # Core model implementation and features
│   ├── docs                     # Documentation and demo images
│   ├── samples                  # Example audio clips and labels
│   ├── tests                    # Test script
│   ├── Dockerfile               # Container setup
│   ├── requirements.txt         # Dependencies
│   └── app.py                   # Entry point for classifier
│
├── yamnet                       # TensorFlow Lite model and labels
│   ├── 1.tflite
│   ├── yamnet.py
│   └── yamnet_label_list.txt
│
├── audio-analyser               # Node.js audio analyser (frontend utility)
│   ├── main.js
│   ├── package.json
│   └── README.md
│
├── mqtt_loger.py                # Logs detected sound events to SQLite DB
├── emergency_dashboard_final.py # Streamlit dashboard for visualization
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Getting Started

### Prerequisites

* Python **3.10+**
* Node.js (for the optional `audio-analyser` tool)

### Installation

Clone this repository or download the project folder.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

(For `MAX-Audio-Classifier`, use its `requirements.txt` separately if needed.)

---

## Running the Programs

### 1. Start the MQTT Logger

This script connects to the HiveMQ Cloud broker and logs detected sound events into the SQLite database:

```bash
python mqtt_loger.py
```

Expected output:

```
Connected to HiveMQ Cloud
```

Keep this terminal running.

### 2. Launch the Dashboard

In a new terminal:

```bash
streamlit run emergency_dashboard_final.py
```

This opens the dashboard at: [http://localhost:8501](http://localhost:8501)

---

## Audio Classifier (MAX-Audio-Classifier)

* Implements a **VGGish-based classifier** for environmental and emergency sounds.
* Includes pretrained weights, Docker setup, and test samples (e.g., *sirens, gunshots, crowd noises, rain, thunder*).
* Run `app.py` or use the `predict.py` API for classification.

---

## YamNet Module

* Contains a lightweight **TensorFlow Lite YAMNet model** (`1.tflite`) and label mappings.
* Can be integrated into the main detection pipeline for on-device inference.

---

## Audio Analyser

The `audio-analyser` folder provides a simple Node.js frontend tool for visualizing audio streams.

Setup:

```bash
cd audio-analyser
npm install
node main.js
```

---

## Samples

The `samples/` folder inside `MAX-Audio-Classifier` includes various audio clips (e.g., sirens, gunshots, piano, thunder) and a `class_labels_indices.csv` file mapping sound labels.

---

## Next Steps

* Train or fine-tune models on custom emergency datasets.
* Extend the dashboard with real-time alerts.
* Deploy to IoT edge devices for live monitoring.

---
