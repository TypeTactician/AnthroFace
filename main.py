"""Main application entry point. Wires together all components into a PyQt6 GUI."""

import sys
import os
import uuid
import base64
import io
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDialog, QFormLayout,
    QLineEdit, QTabWidget, QScrollArea, QFileDialog, QMessageBox,
    QTextEdit, QHeaderView, QTableWidget, QTableWidgetItem, QSplitter,
)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from camera import CameraWorker
from landmarks import extract_landmarks, draw_landmarks, draw_face_guide, compute_head_tilt, validate_landmark_distances, draw_hairline_overlay
from metrics import compute_all_metrics
from scorer import run_scoring
from suggestions import get_suggestions, get_posture_warnings
from ui_results import ResultsPanel
from database import save_session, get_all_sessions, get_session_metrics
from report import generate_pdf


class CalibrationDialog(QDialog):
    """Dialog for user calibration before analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.sex = "male"
        self.ethnicity = ""
        self.camera_distance = 500
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Before starting, please provide the following information.\n"
            "This helps calibrate reference norms for your analysis."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #cccccc; padding: 10px;")
        layout.addWidget(info)

        form = QFormLayout()

        sex_label = QLabel("Biological Sex:")
        sex_label.setStyleSheet("color: #ffffff;")
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Male", "Female"])
        self.sex_combo.setStyleSheet(
            "QComboBox { background-color: #2a2a3a; color: #ffffff; padding: 5px; border: 1px solid #444444; }"
        )
        form.addRow(sex_label, self.sex_combo)

        eth_label = QLabel("Ethnicity (optional):")
        eth_label.setStyleSheet("color: #ffffff;")
        self.eth_combo = QComboBox()
        self.eth_combo.addItems(["Not specified", "Caucasian", "African", "Asian", "Hispanic", "Other"])
        self.eth_combo.setStyleSheet(
            "QComboBox { background-color: #2a2a3a; color: #ffffff; padding: 5px; border: 1px solid #444444; }"
        )
        form.addRow(eth_label, self.eth_combo)

        dist_label = QLabel("Distance from camera (mm):")
        dist_label.setStyleSheet("color: #ffffff;")
        self.dist_input = QLineEdit("500")
        self.dist_input.setStyleSheet(
            "QLineEdit { background-color: #2a2a3a; color: #ffffff; padding: 5px; border: 1px solid #444444; }"
        )
        form.addRow(dist_label, self.dist_input)

        layout.addLayout(form)

        disclaimer = QLabel(
            "This tool measures geometric facial proportions only.\n"
            "It does not evaluate attractiveness, worth, or beauty.\n"
            "All measurements are based on published anthropometric literature.\n"
            "Results are for educational and self-improvement purposes only."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "color: #888888; font-size: 10px; font-style: italic; "
            "padding: 10px; background-color: #1a1a2a; border-radius: 5px;"
        )
        layout.addWidget(disclaimer)

        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #444444; color: #ffffff; padding: 8px 16px; border-radius: 4px; }"
        )
        btn_layout.addWidget(self.cancel_btn)

        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.accept)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: #ffffff; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3498db; }"
        )
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

    def get_values(self) -> tuple[str, str, int]:
        return (
            self.sex_combo.currentText().lower(),
            self.eth_combo.currentText() if self.eth_combo.currentText() != "Not specified" else "",
            int(self.dist_input.text() or 500),
        )


class HistoryDialog(QDialog):
    """Dialog showing session history with progress chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session History")
        self.setMinimumSize(800, 600)
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(8, 3), dpi=100)
        self.figure.patch.set_facecolor("#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Date", "Sex", "Overall", "Symmetry", "Proportions", "Profile"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1e1e2e; color: #ffffff; gridline-color: #333333; }"
            "QHeaderView::section { background-color: #2a2a3a; color: #ffffff; }"
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._load_session_details)
        layout.addWidget(self.table)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        self.details.setStyleSheet(
            "QTextEdit { background-color: #1e1e2e; color: #ffffff; border: 1px solid #333333; padding: 8px; }"
        )
        layout.addWidget(self.details)

        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_selected)
        delete_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: #ffffff; padding: 6px 12px; border-radius: 4px; }"
        )
        btn_layout.addWidget(delete_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #444444; color: #ffffff; padding: 6px 12px; border-radius: 4px; }"
        )
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _load_history(self):
        sessions = get_all_sessions()
        self.table.setRowCount(len(sessions))
        dates = []
        scores = []

        for i, s in enumerate(sessions):
            self.table.setItem(i, 0, QTableWidgetItem(s["date"]))
            self.table.setItem(i, 1, QTableWidgetItem(s["sex"] or "N/A"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{s['overall_score']:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{s['symmetry_score']:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{s['proportions_score']:.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{s['profile_score']:.1f}"))
            dates.append(s["date"])
            scores.append(s["overall_score"])

        self.figure.clear()
        if scores:
            ax = self.figure.add_subplot(111)
            ax.plot(range(len(scores)), scores, marker="o", color="#3498db", linewidth=2, markersize=6)
            ax.fill_between(range(len(scores)), scores, alpha=0.1, color="#3498db")
            ax.set_xticks(range(len(scores)))
            ax.set_xticklabels([d.split(" ")[0] for d in dates], rotation=45, fontsize=7, color="#cccccc")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Overall Score", color="#cccccc", fontsize=10)
            ax.set_title("Session History - Overall Score Trend", color="#ffffff", fontsize=12)
            ax.spines["bottom"].set_color("#444444")
            ax.spines["top"].set_color("#444444")
            ax.spines["left"].set_color("#444444")
            ax.spines["right"].set_color("#444444")
            ax.tick_params(colors="#888888")
            ax.set_facecolor("#1e1e2e")
        self.figure.tight_layout()
        self.canvas.draw()

    def _load_session_details(self, row, col):
        session_id = self.table.item(row, 0).text()
        sessions = get_all_sessions()
        for s in sessions:
            if s["date"] == session_id:
                metrics = get_session_metrics(s["session_id"])
                html = f'<div style="color:#ffffff; font-family:Arial;"><h3>Session: {s["date"]}</h3>'
                html += f'<p>Sex: {s["sex"]} | Ethnicity: {s["ethnicity"] or "N/A"}</p>'
                html += f'<p>Overall Score: {s["overall_score"]:.1f}</p>'
                if metrics:
                    html += "<h4>Metrics:</h4><table border='1' style='border-color:#444;'>"
                    html += "<tr style='background:#2a2a3a;'><th>Metric</th><th>Value</th><th>Score</th></tr>"
                    for m in metrics[:10]:
                        html += f"<tr><td>{m['name']}</td><td>{m['measured_value']}</td><td>{m['score']}</td></tr>"
                    html += "</table>"
                html += "</div>"
                self.details.setHtml(html)
                break

    def _delete_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        date_str = self.table.item(row, 0).text()
        sessions = get_all_sessions()
        for s in sessions:
            if s["date"] == date_str:
                from database import delete_session
                delete_session(s["session_id"])
                self._load_history()
                self.details.clear()
                break


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Anthropometric Analysis Tool")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #121220;")

        self.session_id = str(uuid.uuid4())[:8]
        self.sex = "male"
        self.ethnicity = ""
        self.front_capture = None
        self.profile_capture = None
        self.front_landmarks = None
        self.profile_landmarks = None
        self.front_landmark_frame = None
        self.profile_landmark_frame = None
        self.head_tilt = 0.0
        self.current_metrics = []
        self.radar_b64 = None
        self.persist_landmarks = False
        self.last_landmarks = None

        self.camera_worker = None
        self._setup_ui()

        show_cal = CalibrationDialog(self)
        if show_cal.exec() == QDialog.DialogCode.Accepted:
            self.sex, self.ethnicity, _ = show_cal.get_values()
            self._start_camera("front")
        else:
            self.close()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([480, 720])
        main_layout.addWidget(splitter)

        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar { background-color: #1e1e2e; color: #ffffff; }"
            "QMenuBar::item { padding: 8px 12px; }"
            "QMenuBar::item:selected { background-color: #333333; }"
        )

        settings_menu = menubar.addMenu("Settings")
        settings_act = settings_menu.addAction("Calibration")
        settings_act.triggered.connect(self._show_calibration)

        history_menu = menubar.addMenu("History")
        history_act = history_menu.addAction("View History")
        history_act.triggered.connect(self._show_history)

        export_menu = menubar.addMenu("Export")
        export_act = export_menu.addAction("Export PDF Report")
        export_act.triggered.connect(self._export_pdf)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(480)
        panel.setMaximumWidth(640)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet("background-color: #000000; border: 2px solid #333333;")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText('<span style="color:#666666;">Initializing camera...</span>')
        layout.addWidget(self.camera_label)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888; padding: 4px;")
        layout.addWidget(self.status_label)

        self.mode_label = QLabel("Mode: Front Face")
        self.mode_label.setStyleSheet("color: #3498db; font-weight: bold; padding: 4px;")
        layout.addWidget(self.mode_label)

        btn_layout = QHBoxLayout()
        self.capture_front_btn = QPushButton("Capture Front")
        self.capture_front_btn.clicked.connect(self._capture_front)
        self.capture_front_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: #ffffff; padding: 10px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3498db; }"
        )
        btn_layout.addWidget(self.capture_front_btn)

        self.capture_profile_btn = QPushButton("Capture Profile")
        self.capture_profile_btn.clicked.connect(self._capture_profile)
        self.capture_profile_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: #ffffff; padding: 10px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #2ecc71; }"
        )
        btn_layout.addWidget(self.capture_profile_btn)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: #ffffff; padding: 10px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #9b59b6; }"
            "QPushButton:disabled { background-color: #555555; }"
        )
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.analyze_btn)

        self.toggle_landmarks_btn = QPushButton("Hide Landmarks")
        self.toggle_landmarks_btn.clicked.connect(self._toggle_landmark_overlay)
        self.toggle_landmarks_btn.setStyleSheet(
            "QPushButton { background-color: #555555; color: #ffffff; padding: 10px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #666666; }"
        )
        self.toggle_landmarks_btn.setVisible(False)
        btn_layout.addWidget(self.toggle_landmarks_btn)

        layout.addLayout(btn_layout)

        auto_layout = QHBoxLayout()
        self.auto_capture_checkbox = QLabel("Auto-capture when ready:")
        self.auto_capture_checkbox.setStyleSheet("color: #cccccc;")
        from PyQt6.QtWidgets import QCheckBox
        self.auto_capture_cb = QCheckBox()
        self.auto_capture_cb.setStyleSheet("QCheckBox { color: #ffffff; }")
        self.auto_capture_cb.stateChanged.connect(self._toggle_auto_capture)
        auto_layout.addWidget(self.auto_capture_checkbox)
        auto_layout.addWidget(self.auto_capture_cb)
        auto_layout.addStretch()
        layout.addLayout(auto_layout)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        self.results_panel = ResultsPanel()
        layout.addWidget(self.results_panel)
        return panel

    def _start_camera(self, mode: str):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.cleanup()

        self.persist_landmarks = False
        self.toggle_landmarks_btn.setVisible(False)

        self.camera_worker = CameraWorker(capture_mode=mode)
        self.camera_worker.frame_ready.connect(self._update_frame)
        self.camera_worker.captured.connect(self._on_captured)
        self.camera_worker.status_message.connect(self._update_status)
        self.camera_worker.start()

        if mode == "front":
            self.mode_label.setText("Mode: Front Face")
        else:
            self.mode_label.setText("Mode: Side Profile")

    def _update_frame(self, frame, landmarks, confidence, status):
        if status:
            self.status_label.setText(status)

        display_frame = frame.copy()
        if self.persist_landmarks and self.last_landmarks is not None:
            display_frame = draw_landmarks(display_frame, self.last_landmarks)
        elif landmarks is not None:
            display_frame = draw_landmarks(display_frame, landmarks)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(img))

        if landmarks is not None:
            self.head_tilt = compute_head_tilt(landmarks)
            if not self.persist_landmarks:
                self.last_landmarks = landmarks.copy()

    def _update_status(self, msg):
        self.status_label.setText(msg)

    def _capture_front(self):
        if self.camera_worker:
            self.camera_worker.trigger_capture()

    def _capture_profile(self):
        if self.camera_worker and self.camera_worker.capture_mode == "front":
            self._start_camera("profile")
        elif self.camera_worker:
            self.camera_worker.trigger_capture()

    def _on_captured(self, frame, landmarks, landmark_frame):
        if self.camera_worker.capture_mode == "front":
            self.front_capture = frame.copy()
            self.front_landmarks = landmarks.copy() if landmarks is not None else None
            overlay = draw_hairline_overlay(frame.copy(), landmarks) if landmarks is not None else frame.copy()
            self.front_landmark_frame = draw_landmarks(overlay, landmarks) if landmarks is not None else overlay
            self._show_captured_preview(self.front_landmark_frame)
            self.status_label.setText("Front face captured. Now capture profile or click Analyze.")
            self._update_analyze_button()
        else:
            self.profile_capture = frame.copy()
            self.profile_landmarks = landmarks.copy() if landmarks is not None else None
            self.profile_landmark_frame = landmark_frame.copy()
            self._show_captured_preview(self.profile_landmark_frame)
            self.status_label.setText("Profile captured. Click Analyze to run measurements.")
            self._update_analyze_button()

    def _show_captured_preview(self, landmark_frame):
        if landmark_frame is None:
            return
        rgb = cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(img))

    def _update_analyze_button(self):
        self.analyze_btn.setEnabled(self.front_landmarks is not None)

    def _toggle_landmark_overlay(self):
        self.persist_landmarks = not self.persist_landmarks
        if self.persist_landmarks:
            self.toggle_landmarks_btn.setText("Hide Landmarks")
        else:
            self.toggle_landmarks_btn.setText("Show Landmarks")

    def _toggle_auto_capture(self, state):
        if self.camera_worker:
            self.camera_worker.set_auto_capture(state == 2)

    def _run_analysis(self):
        if self.front_landmarks is None:
            QMessageBox.warning(self, "No Data", "Please capture a front face image first.")
            return

        self.status_label.setText("Running analysis...")
        QApplication.processEvents()

        data_valid, validation_warnings = validate_landmark_distances(self.front_landmarks)
        self.current_metrics = compute_all_metrics(self.front_landmarks, self.front_capture, self.sex)

        if self.profile_landmarks is not None:
            from metrics import measure_profile
            profile_data = measure_profile(self.profile_landmarks)
            profile_metric = {
                "name": "Profile Convexity",
                "measured_value": round(profile_data.get("profile_convexity", 0), 1),
                "ideal_value": 180.0,
                "deviation": round(abs(profile_data.get("profile_convexity", 180) - 180) / 180 * 100, 1),
                "reference": "Ricketts 1981",
                "category": "Profile",
            }
            profile_metric["score"] = max(0, min(100, 100 - profile_metric["deviation"]))
            self.current_metrics.append(profile_metric)

        scored = run_scoring(self.current_metrics)

        front_b64 = self._frame_to_b64(self.front_landmark_frame) if self.front_landmark_frame is not None else None
        profile_b64 = self._frame_to_b64(self.profile_landmark_frame) if self.profile_landmark_frame is not None else None

        from ui_results import RadarChartWidget
        self._save_radar_chart(scored["category_scores"], scored["overall"])

        self.results_panel.display_results(scored["metrics"], self.head_tilt, validation_warnings)

        save_session(
            session_id=self.session_id,
            sex=self.sex,
            ethnicity=self.ethnicity,
            overall_score=scored["overall"],
            symmetry_score=scored["category_scores"].get("Symmetry", 0),
            proportions_score=scored["category_scores"].get("Proportions", 0),
            profile_score=scored["category_scores"].get("Profile", 0),
            golden_ratio_score=scored["category_scores"].get("Golden Ratio", 0),
            metrics=scored["metrics"],
            front_image_b64=front_b64,
            profile_image_b64=profile_b64,
        )

        self.status_label.setText(f"Analysis complete. Overall score: {scored['overall']}/100")
        self.persist_landmarks = True
        self.toggle_landmarks_btn.setVisible(True)
        self.session_id = str(uuid.uuid4())[:8]

    def _save_radar_chart(self, category_scores: dict, overall: float):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#1e1e2e")

        categories = ["Symmetry", "Proportions", "Profile", "Golden Ratio", "Skin"]
        values = [category_scores.get(cat, 0) for cat in categories]

        num_vars = len(categories)
        angles = [n / num_vars * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]
        ideal_values = [100] * (num_vars + 1)

        ax.fill(angles, values, alpha=0.25, color="#3498db")
        ax.plot(angles, values, color="#3498db", linewidth=2, label="Measured")
        ax.plot(angles, ideal_values, color="#2ecc71", linewidth=1, linestyle="--", label="Ideal")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color="#cccccc", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#888888", fontsize=8)
        ax.tick_params(colors="#888888")
        ax.spines["polar"].set_color("#444444")
        ax.grid(color="#333333")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8, labelcolor="#cccccc")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1e1e2e")
        buf.seek(0)
        self.radar_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    def _frame_to_b64(self, frame) -> str | None:
        if frame is None:
            return None
        _, buf = cv2.imencode(".png", frame)
        return base64.b64encode(buf).decode("utf-8")

    def _export_pdf(self):
        if not self.current_metrics:
            QMessageBox.warning(self, "No Data", "Run an analysis first before exporting.")
            return

        scored = run_scoring(self.current_metrics)
        suggestions = get_suggestions(scored["metrics"], scored["category_scores"])

        front_b64 = self._frame_to_b64(self.front_landmark_frame) if self.front_landmark_frame is not None else None
        profile_b64 = self._frame_to_b64(self.profile_landmark_frame) if self.profile_landmark_frame is not None else None

        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )
        if not path:
            return

        try:
            output = generate_pdf(
                session_id=self.session_id,
                sex=self.sex,
                ethnicity=self.ethnicity,
                overall_score=scored["overall"],
                category_scores=scored["category_scores"],
                metrics=scored["metrics"],
                suggestions=suggestions,
                front_image_b64=front_b64,
                profile_image_b64=profile_b64,
                radar_image_b64=self.radar_b64,
                output_path=path,
            )
            QMessageBox.information(self, "Export Complete", f"Report saved to:\n{output}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to generate report:\n{e}")

    def _show_calibration(self):
        dialog = CalibrationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.sex, self.ethnicity, _ = dialog.get_values()

    def _show_history(self):
        dialog = HistoryDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QMainWindow { background-color: #121220; }"
        "QWidget { background-color: #121220; color: #ffffff; }"
        "QLabel { color: #ffffff; }"
        "QPushButton { background-color: #2a2a3a; color: #ffffff; border: 1px solid #444444; padding: 6px 12px; border-radius: 4px; }"
        "QPushButton:hover { background-color: #3a3a4a; }"
        "QDialog { background-color: #1e1e2e; }"
    )
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
