# AI Driver Safety Web Application

This package keeps the original `driver_safety.py` that you supplied and adds a separate
`web_app.py` browser dashboard.

## 1. Install packages

Open the VS Code Terminal in this folder and run:

```text
pip install -r requirements.txt
```

## 2. Add Telegram values

Open `web_app.py` and replace:

```python
TELEGRAM_BOT_TOKEN = "Your bot token here"
TELEGRAM_CHAT_IDS = ["your bot id here"]
```

Do not share your real bot token publicly.

## 3. Start the website

Run:

```text
python web_app.py
```

Then open a browser and go to:

```text
http://127.0.0.1:5000
```

Click **START SYSTEM**.

## Important

The original `driver_safety.py` is included unchanged from the code supplied in the chat.
The browser version is in `web_app.py`.

The web version keeps the same main detection ideas:
- OpenCV camera
- MediaPipe Face Mesh
- EAR drowsiness detection
- MAR yawning detection
- Farneback optical-flow crash/rollover detection
- audio warning
- Telegram SOS with location
- manual SOS

The web version adds a dashboard, live camera stream, status values, Start/Stop controls
and Manual SOS.

Internet is needed for first-time gTTS audio generation and for Telegram/location requests.
The local dashboard itself runs on the laptop.
