"""Results panel UI with radar chart, metrics table, circular gauge, and suggestions."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QGroupBox,
    QPushButton, QTextEdit, QProgressBar, QFrame, QSplitter,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QConicalGradient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scorer import compute_category_scores, compute_overall_score
from suggestions import get_suggestions


class CircularGauge(QWidget):
    """Circular progress indicator for overall score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.score = 0
        self.setMinimumSize(180, 180)
        self.setMaximumSize(180, 180)

    def set_score(self, score: float):
        self.score = score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2

        bg_pen = QPen(QColor(40, 40, 50), 12)
        painter.setPen(bg_pen)
        painter.drawEllipse(
            center.x() - radius, center.y() - radius,
            radius * 2, radius * 2,
        )

        if self.score > 0:
            if self.score >= 85:
                color = QColor(46, 204, 113)
            elif self.score >= 60:
                color = QColor(241, 196, 15)
            else:
                color = QColor(231, 76, 60)

            pen = QPen(color, 12)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)

            start_angle = 90 * 16
            span_angle = -int(self.score / 100 * 360 * 16)
            painter.drawArc(
                center.x() - radius, center.y() - radius,
                radius * 2, radius * 2,
                start_angle, span_angle,
            )

        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 28, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{int(self.score)}")

        painter.setPen(QColor(150, 150, 160))
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignCenter, "/ 100")


class RadarChartWidget(QWidget):
    """Matplotlib radar chart embedded in Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor("#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update_chart(self, category_scores: dict, overall: float):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="polar")

        categories = ["Symmetry", "Proportions", "Profile", "Golden Ratio", "Skin"]
        values = []
        for cat in categories:
            values.append(category_scores.get(cat, 0))

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

        self.figure.tight_layout()
        self.canvas.draw()


class MetricsTableWidget(QTableWidget):
    """Color-coded metrics table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Metric", "Measured", "Ideal", "Deviation", "Score", "Reference"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(
            "QTableWidget { background-color: #1e1e2e; color: #ffffff; gridline-color: #333333; }"
            "QHeaderView::section { background-color: #2a2a3a; color: #ffffff; padding: 4px; }"
        )

    def populate(self, metrics: list[dict]):
        self.setRowCount(len(metrics))
        for i, m in enumerate(metrics):
            score = m["score"]
            if score >= 85:
                bg_color = "rgba(46, 204, 113, 0.15)"
            elif score >= 60:
                bg_color = "rgba(241, 196, 15, 0.15)"
            else:
                bg_color = "rgba(231, 76, 60, 0.15)"

            items = [
                m["name"],
                str(m["measured_value"]),
                str(m["ideal_value"]),
                f"{m['deviation']}%",
                f"{score}/100",
                m["reference"],
            ]
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setBackground(QColor(bg_color))
                self.setItem(i, j, item)


class SuggestionsWidget(QTextEdit):
    """Read-only suggestions display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(
            "QTextEdit { background-color: #1e1e2e; color: #ffffff; border: 1px solid #333333; padding: 8px; }"
        )

    def populate(self, suggestions: list[dict], warnings: list[str]):
        html = '<div style="color:#ffffff; font-family: Arial, sans-serif;">'

        if warnings:
            html += '<h3 style="color:#e74c3c;">Posture Warnings</h3>'
            for w in warnings:
                html += f'<p style="color:#e74c3c; font-size:12px;">{w}</p>'

        html += '<h3 style="color:#3498db;">Improvement Suggestions</h3>'

        if not suggestions:
            html += '<p style="color:#2ecc71;">All metrics are within good range. Keep up the good work!</p>'
        else:
            for s in suggestions:
                score_color = "#e74c3c" if s["score"] < 60 else ("#f1c40f" if s["score"] < 80 else "#2ecc71")
                html += f'<div style="margin-bottom:12px; padding:8px; border-left:3px solid {score_color};">'
                html += f'<strong style="color:{score_color};">Priority {s["priority"]}: {s["metric_name"]}</strong> '
                html += f'<span style="color:#888888;">(Score: {s["score"]})</span><br>'
                html += f'<p style="margin:4px 0; font-size:12px;">{s["suggestion"]}</p>'
                html += f'<p style="margin:2px 0; font-size:10px; color:#666666;"><em>{s["evidence"]}</em></p>'
                html += '</div>'

        html += '</div>'
        self.setHtml(html)


class CategoryScoreDisplay(QWidget):
    """Horizontal display of category scores."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.labels = {}
        for cat in ["Symmetry", "Proportions", "Profile", "Golden Ratio", "Skin"]:
            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.setContentsMargins(0, 0, 0, 0)
            v_layout.setSpacing(2)

            name_label = QLabel(cat)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            v_layout.addWidget(name_label)

            score_label = QLabel("--/100")
            score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            score_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
            v_layout.addWidget(score_label)

            bar = QProgressBar()
            bar.setFixedHeight(6)
            bar.setStyleSheet(
                "QProgressBar { background-color: #333333; border: none; border-radius: 3px; }"
                "QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }"
            )
            bar.setValue(0)
            v_layout.addWidget(bar)

            layout.addWidget(container)
            self.labels[cat] = (score_label, bar)

    def update_scores(self, category_scores: dict):
        for cat, (score_label, bar) in self.labels.items():
            score = category_scores.get(cat, 0)
            score_label.setText(f"{int(score)}/100")
            bar.setValue(int(score))

            if score >= 85:
                bar.setStyleSheet(
                    "QProgressBar { background-color: #333333; border: none; border-radius: 3px; }"
                    "QProgressBar::chunk { background-color: #2ecc71; border-radius: 3px; }"
                )
            elif score >= 60:
                bar.setStyleSheet(
                    "QProgressBar { background-color: #333333; border: none; border-radius: 3px; }"
                    "QProgressBar::chunk { background-color: #f1c40f; border-radius: 3px; }"
                )
            else:
                bar.setStyleSheet(
                    "QProgressBar { background-color: #333333; border: none; border-radius: 3px; }"
                    "QProgressBar::chunk { background-color: #e74c3c; border-radius: 3px; }"
                )


class ResultsPanel(QWidget):
    """Main results panel combining all result displays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #121220;")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        scores_layout = QHBoxLayout()

        self.gauge = CircularGauge()
        scores_layout.addWidget(self.gauge)

        gauge_labels = QVBoxLayout()
        title = QLabel("OVERALL SCORE")
        title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        gauge_labels.addWidget(title)

        self.disclaimer = QLabel(
            "This tool measures geometric facial proportions only.\n"
            "It does not evaluate attractiveness, worth, or beauty.\n"
            "All measurements are based on published anthropometric literature.\n"
            "Results are for educational and self-improvement purposes only."
        )
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setStyleSheet("color: #888888; font-size: 10px; font-style: italic;")
        gauge_labels.addWidget(self.disclaimer)

        scores_layout.addLayout(gauge_labels)
        scores_layout.addStretch()
        main_layout.addLayout(scores_layout)

        self.category_scores = CategoryScoreDisplay()
        main_layout.addWidget(self.category_scores)

        self.radar_chart = RadarChartWidget()
        main_layout.addWidget(self.radar_chart)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #333333; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)

        metrics_group = QGroupBox("Detailed Metrics")
        metrics_group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #444444; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_table = MetricsTableWidget()
        metrics_layout.addWidget(self.metrics_table)
        scroll_layout.addWidget(metrics_group)

        suggestions_group = QGroupBox("Improvement Suggestions")
        suggestions_group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #444444; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        suggestions_layout = QVBoxLayout(suggestions_group)
        self.suggestions_text = SuggestionsWidget()
        suggestions_layout.addWidget(self.suggestions_text)
        scroll_layout.addWidget(suggestions_group)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def display_results(self, metrics: list[dict], head_tilt: float = 0.0, validation_warnings: list[str] | None = None):
        from scorer import compute_category_scores, compute_overall_score
        from suggestions import get_suggestions, get_posture_warnings

        category_scores = compute_category_scores(metrics)
        overall = compute_overall_score(metrics, category_scores)

        self.gauge.set_score(overall)
        self.category_scores.update_scores(category_scores)
        self.radar_chart.update_chart(category_scores, overall)
        self.metrics_table.populate(metrics)

        suggestions = get_suggestions(metrics, category_scores)
        warnings = get_posture_warnings(head_tilt)
        if validation_warnings:
            warnings = validation_warnings + warnings
        self.suggestions_text.populate(suggestions, warnings)

    def clear(self):
        self.gauge.set_score(0)
        self.metrics_table.setRowCount(0)
        self.suggestions_text.setHtml(
            '<div style="color:#888888; padding:20px; text-align:center;">'
            "Capture front and profile images to see results."
            "</div>"
        )
