from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
from gtts import gTTS
import pygame
import requests

# ============================================================
# WEB APP VERSION
# Your original detection process is kept: Optical Flow,
# Face Mesh, EAR, MAR, drowsiness, yawning and Telegram SOS.
# This file adds a browser dashboard around that process.
# ============================================================

app = Flask(__name__)
current_lat = "12.304558"
current_lng = "79.739840"

@app.route('/update_location', methods=['POST'])
def update_location():
    global current_lat, current_lng
    data = request.json
    if data:
        current_lat = str(data.get("latitude", current_lat))
        current_lng = str(data.get("longitude", current_lng))
    return jsonify({"status": "success"})    

EMERGENCY_CONTACT = "8870514906"

TELEGRAM_BOT_TOKEN = "8156004303:AAG3WArXduKXQU2qJSYXp-YG-osD3-lvUg0"
TELEGRAM_CHAT_IDS = ["8625724135"]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60
CLOSED_FRAMES_LIMIT = 10

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init()

camera = None
running = False
worker_thread = None
latest_jpeg = None
frame_lock = threading.Lock()
state_lock = threading.Lock()

status = {
    "system": "STOPPED",
    "drowsiness": "SAFE",
    "yawning": "SAFE",
    "accident": "SAFE",
    "sos": "READY",
    "ear": 0.0,
    "mar": 0.0,
    "motion": 0.0,
    "location": "Not available",
    "message": "Press START SYSTEM"
}

sos_sent = False


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def get_ear(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
           for i in eye_indices]
    v1 = calculate_distance(pts[1], pts[5])
    v2 = calculate_distance(pts[2], pts[4])
    h = calculate_distance(pts[0], pts[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def get_mar(landmarks, img_w, img_h):
    p13 = (int(landmarks[13].x * img_w), int(landmarks[13].y * img_h))
    p14 = (int(landmarks[14].x * img_w), int(landmarks[14].y * img_h))
    p81 = (int(landmarks[81].x * img_w), int(landmarks[81].y * img_h))
    p178 = (int(landmarks[178].x * img_w), int(landmarks[178].y * img_h))
    p311 = (int(landmarks[311].x * img_w), int(landmarks[311].y * img_h))
    p402 = (int(landmarks[402].x * img_w), int(landmarks[402].y * img_h))
    p61 = (int(landmarks[61].x * img_w), int(landmarks[61].y * img_h))
    p291 = (int(landmarks[291].x * img_w), int(landmarks[291].y * img_h))

    v1 = calculate_distance(p13, p14)
    v2 = calculate_distance(p81, p178)
    v3 = calculate_distance(p311, p402)
    h = calculate_distance(p61, p291)

    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def get_live_location():
    global current_lat, current_lng
    return current_lat, current_lng


def send_telegram_alert_worker(reason):
    global sos_sent
    lat, lng = get_live_location()
    maps_url = f"https://www.google.com/maps?q={lat},{lng}"

    message = (
        f"*EMERGENCY ALERT: ACCIDENT DETECTED!*\n\n"
        f"*Status:* {reason}\n"
        f"*Live Location:* [Click Here for Google Maps]({maps_url})\n\n"
        f"Latitude: {lat}\nLongitude: {lng}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    ok_count = 0
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            response = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=5
            )
            data = response.json()
            print("Telegram response:", data)
            if data.get("ok"):
                ok_count += 1
        except Exception as e:
            print("Telegram error:", e)

    with state_lock:
        status["sos"] = f"SENT ({ok_count})"
        status["location"] = f"{lat}, {lng}"
        status["message"] = reason


def trigger_sos(reason="ACCIDENT DETECTED"):
    global sos_sent
    if sos_sent:
        return
    sos_sent = True
    with state_lock:
        status["sos"] = "SENDING..."
        status["accident"] = "DETECTED"
        status["message"] = reason
    threading.Thread(
        target=send_telegram_alert_worker,
        args=(reason,),
        daemon=True
    ).start()


def play_sound_thread(file_path):
    if not pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            print("Sound play error:", e)


def play_sound(file_path):
    threading.Thread(
        target=play_sound_thread,
        args=(file_path,),
        daemon=True
    ).start()


def create_audio_files():
    if not os.path.exists("drowsy.mp3"):
        print("Creating drowsy.mp3...")
        gTTS(
            text="you are closing your eyes, stay alert!",
            lang="en"
        ).save("drowsy.mp3")

    if not os.path.exists("yawn.mp3"):
        print("Creating yawn.mp3...")
        gTTS(
            text="you are yawning, take a short break!",
            lang="en"
        ).save("yawn.mp3")


def camera_worker():
    global camera, running, latest_jpeg, sos_sent

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        with state_lock:
            status["system"] = "CAMERA ERROR"
            status["message"] = "Camera could not be opened"
        running = False
        return

    for _ in range(5):
        camera.read()

    ret, prev_frame = camera.read()
    if not ret or prev_frame is None:
        with state_lock:
            status["system"] = "CAMERA ERROR"
            status["message"] = "Camera frame not received"
        camera.release()
        running = False
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0

    with state_lock:
        status["system"] = "RUNNING"
        status["message"] = "Driver safety monitoring active"

    while running:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        img_h, img_w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_speed = float(np.mean(magnitude))

        with state_lock:
            status["motion"] = round(motion_speed, 1)
            status["accident"] = "SAFE"

        if motion_speed > 12.0 and not sos_sent:
            cv2.putText(
                frame, "IMPACT / ROLLOVER DETECTED!",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3
            )
            trigger_sos("Vehicle Rollover / Crash Detected")

        if sos_sent:
            cv2.putText(
                frame, "TELEGRAM ALERT SENT!",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2
            )

        cv2.putText(
            frame, f"Motion Intensity: {motion_speed:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 0), 2
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        with state_lock:
            status["drowsiness"] = "SAFE"
            status["yawning"] = "SAFE"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = get_ear(
                    landmarks, LEFT_EYE, img_w, img_h
                )
                right_ear = get_ear(
                    landmarks, RIGHT_EYE, img_w, img_h
                )
                avg_ear = (left_ear + right_ear) / 2.0
                mar = get_mar(landmarks, img_w, img_h)

                with state_lock:
                    status["ear"] = round(avg_ear, 2)
                    status["mar"] = round(mar, 2)

                if avg_ear < EAR_THRESHOLD:
                    frame_count += 1
                    if frame_count >= CLOSED_FRAMES_LIMIT:
                        with state_lock:
                            status["drowsiness"] = "DETECTED"
                        cv2.putText(
                            frame, "ALERT: DROWSINESS DETECTED!",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 3
                        )
                        play_sound("drowsy.mp3")
                else:
                    frame_count = 0

                if mar > MAR_THRESHOLD:
                    with state_lock:
                        status["yawning"] = "DETECTED"
                    cv2.putText(
                        frame, "ALERT: YAWNING DETECTED!",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 3
                    )
                    play_sound("yawn.mp3")

                cv2.putText(
                    frame, f"EAR: {avg_ear:.2f}",
                    (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"MAR: {mar:.2f}",
                    (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

        prev_gray = gray.copy()

        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            with frame_lock:
                latest_jpeg = encoded.tobytes()

    if camera is not None:
        camera.release()
    camera = None


def start_system():
    global running, worker_thread, sos_sent

    if running:
        return

    sos_sent = False
    with state_lock:
        status["system"] = "STARTING"
        status["sos"] = "READY"
        status["accident"] = "SAFE"
        status["message"] = "Starting camera..."

    running = True
    worker_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )
    worker_thread.start()


def stop_system():
    global running
    running = False
    with state_lock:
        status["system"] = "STOPPED"
        status["message"] = "System stopped"


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def get_status():
    with state_lock:
        return jsonify(dict(status))


@app.post("/start")
def start():
    start_system()
    return jsonify({"ok": True})


@app.post("/stop")
def stop():
    stop_system()
    return jsonify({"ok": True})


@app.post("/sos")
def manual_sos():
    trigger_sos("Manual SOS Triggered by Driver")
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Opening Driver Safety Web Application...")
    create_audio_files()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
