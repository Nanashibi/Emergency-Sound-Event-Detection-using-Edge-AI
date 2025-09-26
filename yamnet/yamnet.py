# file: yamnet_tflite_mqtt.py
# Raspberry Pi continuous audio classification using YAMNet TFLite (no MediaPipe).
# Uses local model '1.tflite' and label list 'yamnet_label_list.txt' (one label per line).
# Publishes only top-1 result above threshold to HiveMQ with same creds/topic/payload.

import argparse
import json
import os
import queue
import ssl
import sys
import time

import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

# --- MQTT config (same as attached file) ---
MQTT_HOST = "2d467655eed049ffbb040e09aaac42ef.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "shreya"
MQTT_PASS = "Shreya123"
MQTT_TOPIC = "audioAnalyser"

# --- Audio config to mirror original behavior ---
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SECONDS = 4.0
THRESHOLD = 0.3
DEVICE_INDEX = None  # set a specific input if needed

# Local file names as provided
DEFAULT_TFLITE_MODEL = "1.tflite"
DEFAULT_LABELS_TXT = "yamnet_label_list.txt"

def load_labels_txt(labels_txt_path):
    labels = []
    with open(labels_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels

def make_mqtt_client():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client

def init_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def infer_top1(interpreter, input_details, output_details, audio_blocks, labels):
    def run_one(x1d):
        x = x1d.astype(np.float32)
        if len(input_details[0]['shape']) == 2:
            x = np.expand_dims(x, axis=0)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        probs = out[0] if out.ndim == 2 else out
        if probs.ndim == 2:
            probs = probs.mean(axis=0)
        return probs

    if audio_blocks.ndim == 1:
        probs = run_one(audio_blocks)
    else:
        acc = None
        for i in range(audio_blocks.shape[0]):
            p = run_one(audio_blocks[i])
            acc = p if acc is None else acc + p
        probs = acc / audio_blocks.shape[0]

    idx = int(np.argmax(probs))
    prob = float(probs[idx])
    label = labels[idx] if 0 <= idx < len(labels) else "Unknown"
    return {"label": label, "probability": prob}

def stream_and_publish(model_path, labels_path):
    client = make_mqtt_client()
    print("connected to MQTT")

    labels = load_labels_txt(labels_path)
    interpreter, in_det, out_det = init_interpreter(model_path)

    expected = in_det[0]['shape']
    if len(expected) == 1:
        CHUNK_SAMPLES = int(expected[0])
    elif len(expected) == 2:
        CHUNK_SAMPLES = int(expected[1])
    else:
        sys.exit(1)

    block_size = 1024
    blocks_per_window = int(SAMPLE_RATE * FRAME_SECONDS / block_size)
    q_in = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        q_in.put(indata.copy())

    with sd.InputStream(channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        blocksize=block_size,
                        dtype='float32',
                        callback=audio_callback,
                        device=DEVICE_INDEX):
        ring = []
        while True:
            start = time.time()
            ring.clear()
            for _ in range(blocks_per_window):
                block = q_in.get()
                if block.ndim > 1:
                    block = block[:, 0]
                ring.append(block)
            audio_f32 = np.concatenate(ring, axis=0)

            total = audio_f32.shape[0]
            num_chunks = max(1, total // CHUNK_SAMPLES)
            trim = num_chunks * CHUNK_SAMPLES
            audio_f32 = audio_f32[:trim]
            if num_chunks > 1:
                chunks = audio_f32.reshape(num_chunks, CHUNK_SAMPLES)
            else:
                chunks = audio_f32

            pred = infer_top1(interpreter, in_det, out_det, chunks, labels)
            label, prob = pred["label"], pred["probability"]
            if label != "Unknown" and prob > THRESHOLD:
                client.publish(MQTT_TOPIC, json.dumps(pred))
                print(label, prob)

            time.sleep(max(0.0, 0.01 - (time.time() - start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_TFLITE_MODEL)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS_TXT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"Labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)

    #global THRESHOLD
    #THRESHOLD = float(args.threshold)

    stream_and_publish(args.model, args.labels)

if __name__ == "__main__":
    main()
