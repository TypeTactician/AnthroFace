"""Webcam capture with MediaPipe FaceLandmarker integration and auto-capture logic."""

import os
import cv2
import numpy as np
import urllib.request
from PyQt6.QtCore import QThread, pyqtSignal

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

from landmarks import (
    extract_landmarks, draw_landmarks, draw_face_guide,
    draw_profile_guide, compute_head_tilt, check_lighting,
    check_face_centered, is_profile_view,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    return MODEL_PATH


class CameraWorker(QThread):
    """Background thread for webcam capture and landmark detection."""

    frame_ready = pyqtSignal(object, object, float, str)
    captured = pyqtSignal(object, object, object)
    auto_capture_ready = pyqtSignal(object, object, object)
    status_message = pyqtSignal(str)

    def __init__(self, camera_id: int = 0, capture_mode: str = "front"):
        super().__init__()
        self.camera_id = camera_id
        self.capture_mode = capture_mode
        self.running = False
        self.do_capture = False
        self.auto_capture = False
        self.frame_interval = 3
        self.frame_count = 0
        self.landmarker = None

        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.status_message.emit("Could not open camera. Check camera permissions.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Failed to read from camera.")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_interval != 0:
                self.frame_ready.emit(frame, None, 0.0, "")
                continue

            landmarks, confidence = extract_landmarks(frame, self.landmarker)

            status = ""
            if landmarks is not None:
                tilt = compute_head_tilt(landmarks)
                lighting = check_lighting(frame)
                centered, center_offset = check_face_centered(landmarks, frame.shape)

                if self.capture_mode == "front":
                    status = f"Tilt: {abs(tilt):.1f}deg | Light: {lighting:.0%} | Center: {'OK' if centered else 'Adjust'}"

                    if self.auto_capture and confidence > 0.8 and abs(tilt) < 5 and lighting > 0.4 and centered:
                        self._emit_capture(frame, landmarks)
                elif self.capture_mode == "profile":
                    profile = is_profile_view(landmarks)
                    status = f"Profile: {'Detected' if profile else 'Turn head 90 deg'} | Conf: {confidence:.0%}"

                    if self.auto_capture and profile and confidence > 0.7:
                        self._emit_capture(frame, landmarks)

            display_frame = frame.copy()
            if self.capture_mode == "front":
                if landmarks is None:
                    display_frame = draw_face_guide(display_frame)
                else:
                    display_frame = draw_landmarks(display_frame, landmarks)
            else:
                if landmarks is None:
                    display_frame = draw_profile_guide(display_frame)
                else:
                    display_frame = draw_landmarks(display_frame, landmarks)

            self.frame_ready.emit(frame, landmarks, confidence or 0.0, status)

            if self.do_capture and landmarks is not None:
                self._emit_capture(frame, landmarks)
                self.do_capture = False

        cap.release()

    def _emit_capture(self, frame, landmarks):
        landmark_frame = draw_landmarks(frame.copy(), landmarks)
        self.captured.emit(frame, landmarks, landmark_frame)
        if self.auto_capture:
            self.auto_capture.emit(frame, landmarks, landmark_frame)
            self.auto_capture = False

    def trigger_capture(self):
        self.do_capture = True

    def set_auto_capture(self, enabled: bool):
        self.auto_capture = enabled

    def stop(self):
        self.running = False
        self.wait()

    def cleanup(self):
        if self.landmarker:
            self.landmarker.close()
