import paho.mqtt.client as mqtt
import sqlite3
import ssl
import json
import time
from datetime import datetime

# ============ MQTT Configuration ============
USERNAME = "shreya"
PASSWORD = "Shreya123"
BROKER = "2d467655eed049ffbb040e09aaac42ef.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "#"  # or your specific topic

# ============ Emergency Labels ============
emergency_labels = ["Gunshot", "Siren", "Explosion", "Glass_breaking", "Screaming", "Fire_alarm"]

# ============ SQLite Setup ============
conn = sqlite3.connect("mqtt_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    is_emergency BOOLEAN,
    timestamp REAL,
    topic TEXT,
    payload TEXT
)
""")
conn.commit()

# ============ MQTT Callbacks ============
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to HiveMQ Cloud")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        label = data.get("label", "")
        is_emergency = label in emergency_labels
        timestamp = time.time()
        cursor.execute("""
            INSERT INTO logs (label, is_emergency, timestamp, topic, payload)
            VALUES (?, ?, ?, ?, ?)
        """, (label, is_emergency, timestamp, msg.topic, payload))
        conn.commit()
        print(f"✅ Logged: {label} | Emergency: {is_emergency} | Topic: {msg.topic}")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============ Start MQTT Client ============
client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)
client.connect(BROKER, PORT)
client.loop_forever()
