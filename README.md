# RoadGuard — Moroccan traffic sign detection

RoadGuard is a Streamlit dashboard for a custom YOLO model. It can detect signs in an uploaded image, an included illustrative image, a browser camera feed, or frames streamed from a Raspberry Pi 4. The Pi captures video and can announce high-confidence detections through its configured audio output. Inference runs on the machine hosting Streamlit; `stream.py` does not run the model.

## Try the dashboard

1. Open the dashboard. **Image Demo** is the default mode.
2. Select **Included sample** or **Upload image**. The included STOP sign illustration is a pipeline test, not a photographed road sign or evidence of model accuracy.
3. Press **Run detection**. Inspect the annotated image, class labels, confidence values, and inference time. A valid image can produce zero detections.
4. Use **Download annotated image** to keep the result. The **Camera** source requests browser permission only if selected.

Each press runs detection once. Changing settings or choosing another image does not silently rerun it. Analytics and history are updated for each run; **Clear History** resets them. The browser camera's continuous WebRTC mode does not populate history.

A hosted Streamlit app performs inference on its cloud server. It cannot normally connect to a private Pi address such as `192.168.x.x` or `10.x.x.x`. **Raspberry Pi Stream** is intended for a local Streamlit process running on a computer that can reach the Pi over the same LAN. Saved snapshots reside on the Streamlit host and may be temporary in cloud deployments; download the image result in your browser to keep it.

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and model inference |
| `stream.py` | Raspberry Pi Flask camera stream and optional speech endpoint |
| `best.pt` | Trained model weights |
| `sample_images/illustrative_stop.png` | Synthetic input for testing the image flow |
| `requirements.txt` | Hosted Streamlit dependencies |
| `requirements_pc.txt` | Local dashboard dependencies |
| `requirements_rpi.txt` | Pi streamer dependencies |
| `packages.txt` | Hosted deployment system packages |

## Run locally on Windows

Install Python and Git for Windows, then open PowerShell in the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_pc.txt
python -m streamlit run app.py
```

Put `best.pt` next to `app.py`. Install the PyTorch build appropriate for your machine if the default installed by `pip` is unsuitable. To try the Pi stream, run the dashboard on the same network as the Pi, choose **Raspberry Pi Stream**, and enter the Pi's current LAN address.

## Run the Pi streamer

The current camera capture uses OpenCV `VideoCapture`. A USB webcam that exposes `/dev/video0` is the simplest option. The official Raspberry Pi camera on recent Raspberry Pi OS releases may require a Picamera2/libcamera capture adapter; this repository does not include one.

```bash
python3 -m pip install -r requirements_rpi.txt
python3 stream.py --device 0 --width 640 --height 480 --fps 15
```

Check `http://<pi-address>:5000/health` and `http://<pi-address>:5000/` from the local computer. Pair and route the Pi's audio output to the Bluetooth speaker at the operating-system level before enabling **Send detections to Pi speaker**. Speech will use the Pi's selected default output device. The Pi stream and `/announce` endpoint do not implement authentication: use them only on a trusted local network, and do not port-forward them to the public internet.

## Testing and presentation

The included illustration confirms that the interface can accept an image and run the inference path, but it may yield no detections. For a portfolio screenshot, run the model on a real photo you have permission to use, verify the label and bounding box, then capture the actual output. Add that screenshot under `docs/` and link it here. No accuracy, latency, or FPS measurements are asserted without a recorded evaluation or hardware test. The Raspberry Pi camera and Bluetooth output require testing on the physical device.

The model file is about 88 MB. GitHub may warn on files larger than 50 MiB; its normal hard limit is 100 MiB. Keep the weights in the repo only if the model's dataset and training terms permit redistribution.
