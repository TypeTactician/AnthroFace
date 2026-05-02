# AnthroFace

> Desktop facial anthropometry tool using MediaPipe landmarks for 14 scientific proportion measurements, automated hairline detection, and evidence-based improvement suggestions. 100% local.

A local desktop application that uses your webcam to objectively measure facial proportions based on established scientific and anthropometric standards, then produces a structured report with actionable improvement suggestions.

> **Disclaimer:** This tool measures geometric facial proportions only. It does not evaluate attractiveness, worth, or beauty. All measurements are based on published anthropometric literature. Results are for educational and self-improvement purposes only.

## Features

- **Live webcam analysis** with 478-point facial landmark overlay (MediaPipe FaceMesh)
- **Automated hairline detection** using adaptive HSV skin/hair segmentation
- **Front face capture** with auto-detection of proper alignment, lighting, and head pose
- **Side profile capture** for nasal and chin projection measurements
- **14 anthropometric metrics** across 5 categories:
  - **Symmetry** — facial bilateral asymmetry index
  - **Proportions** — facial thirds, fifths, eye spacing, lip ratio, nose width
  - **Profile** — gonial angle (jawline), chin projection
  - **Golden Ratio** — face width/height, nose width/length, mouth/nose width
  - **Skin** — skin tone uniformity analysis
- **Radar chart visualization** showing measured vs ideal proportions
- **Evidence-based improvement suggestions** with scientific citations
- **PDF report export** with captured images, full metrics, and bibliography
- **Session history** with SQLite database and progress tracking
- **100% local** — no API calls, no data transmitted anywhere

## Scientific Basis

All measurements reference established anthropometric standards:

| Standard | Source |
|---|---|
| Neoclassical Canons (facial thirds/fifths) | Farkas, L.G. (1994). *Anthropometry of the Head and Face* |
| Facial Proportions | Powell, N. & Humphreys, B. (1984). *Proportions of the Aesthetic Face* |
| Nasal Projection (Goode's Ratio) | Goode, R.L. (1981). *Archives of Otolaryngology* |
| E-Line / Profile Assessment | Ricketts, R.M. (1981). *American Journal of Orthodontics* |
| Gonial Angle Norms | Argyropoulos & Sassouni (1989) |
| Facial Asymmetry | Peck et al. (2008). *AJODO* |
| Skin Tone Analysis | Kafi et al. (2007). *Archives of Dermatology* |

## Installation

### Prerequisites

- Python 3.11 or higher
- Webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/TypeTactician/AnthroFace.git
cd AnthroFace

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

The MediaPipe FaceLandmarker model (~8MB) downloads automatically on first run.

## Usage

1. **Calibration**: On startup, enter your biological sex and optional ethnicity. This calibrates reference norms.

2. **Front Face Capture**:
   - Align your face inside the oval guide
   - Look straight ahead with a neutral expression
   - Click "Capture Front" or enable auto-capture
   - The app checks head tilt (<5°), lighting, face centering, and landmark confidence

3. **Profile Capture**:
   - Turn your head 90° to the right
   - Align your profile with the center line
   - Click "Capture Profile"

4. **Analysis**: Click "Analyze" to run all measurements. Results appear instantly with a radar chart, detailed metrics table, and improvement suggestions.

5. **Export**: Use "Export PDF Report" from the menu to generate a comprehensive report.

6. **History**: View past sessions and track progress via "View History" in the menu.

## Project Structure

```
├── main.py              # App entry point + PyQt6 main window
├── camera.py            # Webcam capture + MediaPipe integration
├── landmarks.py         # Landmark extraction, hairline detection, pixel-to-mm
├── metrics.py           # All anthropometric measurement functions
├── scorer.py            # Scoring engine (exponential decay model)
├── suggestions.py       # Evidence-based improvement database
├── ui_results.py        # Results panel (gauge, radar chart, table)
├── report.py            # PDF report generation (ReportLab)
├── database.py          # SQLite session history
├── requirements.txt     # Python dependencies
└── reports/             # Generated PDF reports
```

## Metrics Reference

| Metric | Ideal | Category |
|---|---|---|
| Facial Thirds (upper/middle/lower) | 0.33 each | Proportions |
| Intercanthal Fifth | 1.0 × eye width | Proportions |
| Nose Width | 1.0 × eye width | Proportions |
| Eye Spacing Ratio | 0.46 | Proportions |
| Lip Ratio (upper/lower) | 0.625 | Proportions |
| Facial Symmetry Index | 1.0 | Symmetry |
| Face Width/Height | 0.618 (1/φ) | Golden Ratio |
| Nose Width/Length | 0.618 (1/φ) | Golden Ratio |
| Mouth/Nose Width | 1.618 (φ) | Golden Ratio |
| Gonial Angle | 120° (M) / 126° (F) | Profile |
| Cheekbone/Jaw Ratio | ~1.3 (M) | Proportions |
| Goode's Ratio (nasal projection) | 0.55–0.60 | Proportions |
| Skin Tone Uniformity | 1.0 | Skin |

## Scoring

Scores use gentle exponential decay: `score = 100 × e^(-deviation/100)`

| Deviation from Ideal | Score |
|---|---|
| 0% | 100 |
| 10% | 90 |
| 25% | 78 |
| 50% | 61 |
| 100% | 37 |

## Privacy

- **All processing is local** — camera feed is never saved unless you explicitly export a PDF
- **No API calls** — no data is sent to any server
- **Session data** is stored in a local SQLite database (`sessions.db`)
- You can delete session history at any time from the app

## Known Limitations

- **Hairline detection** uses HSV segmentation and may struggle with very dark hair on dark backgrounds, bright lighting, or bangs. Falls back to estimation if confidence is low.
- **MediaPipe does not detect the actual trichion** (hairline) — the hairline is detected via image segmentation above the brow. Accuracy depends on hair-to-skin color contrast.
- **Profile measurements** are less reliable than front-face measurements due to reduced landmark visibility.
- **2D image analysis** cannot capture true 3D depth — measurements are planar approximations.

## License

This project is for educational and research purposes only. Not intended for clinical or diagnostic use.

## Bibliography

- Farkas, L.G. (1994). *Anthropometry of the Head and Face*. Raven Press.
- Powell, N. & Humphreys, B. (1984). *Proportions of the Aesthetic Face*. Thieme-Stratton.
- Goode, R.L. (1981). Nasal projection measurement. *Archives of Otolaryngology*, 107(7), 431-433.
- Ricketts, R.M. (1981). The aesthetic environment. *American Journal of Orthodontics*, 79(4), 399-402.
- Argyropoulos, E. & Sassouni, V. (1989). Comparison of craniofacial morphology. *AJODO*, 96(1), 52-61.
- Peck, H. et al. (2008). Facial asymmetry: prevalence and causes. *AJODO*, 133(2), 221-228.
- Kafi, R. et al. (2007). Improvement of naturally aged skin with topical retinol. *Archives of Dermatology*, 143(5), 606-612.
