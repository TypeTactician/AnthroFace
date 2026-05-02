"""PDF report generation using ReportLab."""

import os
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, ListFlowable, ListItem,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.flowables import HRFlowable


DISCLAIMER_TEXT = (
    "DISCLAIMER: This tool measures geometric facial proportions only. "
    "It does not evaluate attractiveness, worth, or beauty. "
    "All measurements are based on published anthropometric literature. "
    "Results are for educational and self-improvement purposes only."
)


def generate_pdf(
    session_id: str,
    sex: str,
    ethnicity: str,
    overall_score: float,
    category_scores: dict,
    metrics: list[dict],
    suggestions: list[dict],
    front_image_b64: str | None = None,
    profile_image_b64: str | None = None,
    radar_image_b64: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate a comprehensive PDF report.

    Returns the path to the generated PDF file.
    """
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"facial_report_{timestamp}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6,
        textColor=colors.HexColor("#2c3e50"),
    ))
    styles.add(ParagraphStyle(
        name="CustomHeading",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2980b9"),
    ))
    styles.add(ParagraphStyle(
        name="CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        textColor=colors.HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        fontName="Helvetica-Oblique",
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="ScoreBig",
        parent=styles["Normal"],
        fontSize=36,
        textColor=colors.HexColor("#2c3e50"),
        alignment=1,
    ))

    story = []

    story.append(Paragraph("Facial Anthropometric Analysis Report", styles["CustomTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Session ID: {session_id}", styles["CustomBody"]))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["CustomBody"]))
    story.append(Paragraph(f"Sex: {sex.capitalize()} | Ethnicity: {ethnicity.capitalize() if ethnicity else 'Not specified'}", styles["CustomBody"]))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    story.append(Paragraph(DISCLAIMER_TEXT, styles["Disclaimer"]))

    story.append(Paragraph("Overall Score", styles["CustomHeading"]))
    overall_color = colors.HexColor("#2ecc71") if overall_score >= 85 else (colors.HexColor("#f1c40f") if overall_score >= 60 else colors.HexColor("#e74c3c"))
    story.append(Paragraph(f"{int(overall_score)} / 100", styles["ScoreBig"]))

    cat_data = [["Category", "Score"]]
    for cat, score in category_scores.items():
        cat_data.append([cat, f"{int(score)}/100"])

    cat_table = Table(cat_data, colWidths=[2.5 * inch, 2.5 * inch])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 12))

    if radar_image_b64:
        story.append(Paragraph("Radar Chart", styles["CustomHeading"]))
        radar_img = _b64_to_image(radar_image_b64, width=5 * inch, height=4 * inch)
        if radar_img:
            story.append(radar_img)
            story.append(Spacer(1, 12))

    if front_image_b64:
        story.append(Paragraph("Captured Front Image", styles["CustomHeading"]))
        front_img = _b64_to_image(front_image_b64, width=4 * inch)
        if front_img:
            story.append(front_img)
            story.append(Spacer(1, 12))

    if profile_image_b64:
        story.append(Paragraph("Captured Profile Image", styles["CustomHeading"]))
        profile_img = _b64_to_image(profile_image_b64, width=4 * inch)
        if profile_img:
            story.append(profile_img)
            story.append(Spacer(1, 12))

    story.append(PageBreak())

    story.append(Paragraph("Detailed Metrics", styles["CustomHeading"]))
    metric_table_data = [["Metric", "Measured", "Ideal", "Score", "Reference"]]
    for m in metrics:
        score_val = m.get("score", 0)
        metric_table_data.append([
            m["name"],
            str(m["measured_value"]),
            str(m["ideal_value"]),
            f"{score_val}/100",
            m["reference"],
        ])

    metric_table = Table(metric_table_data, colWidths=[1.8 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch, 1.7 * inch])
    row_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    for i in range(1, len(metric_table_data)):
        bg = colors.white if i % 2 == 0 else colors.HexColor("#f5f5f5")
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
        score = int(metric_table_data[i][3].split("/")[0])
        if score >= 85:
            row_styles.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#2ecc71")))
        elif score >= 60:
            row_styles.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#f39c12")))
        else:
            row_styles.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#e74c3c")))

    metric_table.setStyle(TableStyle(row_styles))
    story.append(metric_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Top Improvement Suggestions", styles["CustomHeading"]))
    for s in suggestions[:5]:
        suggestion_text = f"<b>Priority {s['priority']}: {s['metric_name']}</b> (Score: {s['score']})<br/>{s['suggestion']}<br/><i>{s['evidence']}</i>"
        story.append(Paragraph(suggestion_text, styles["CustomBody"]))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    story.append(Paragraph("Bibliography", styles["CustomHeading"]))
    bibliography = [
        "Farkas, L.G. (1994). Anthropometry of the Head and Face. Raven Press, New York.",
        "Powell, N. & Humphreys, B. (1984). Proportions of the Aesthetic Face. Thieme-Stratton, New York.",
        "Goode, R.L. (1981). Nasal projection measurement. Archives of Otolaryngology, 107(7), 431-433.",
        "Ricketts, R.M. (1981). The aesthetic environment. American Journal of Orthodontics, 79(4), 399-402.",
        "Argyropoulos, E. & Sassouni, V. (1989). Comparison of craniofacial morphology. American Journal of Orthodontics and Dentofacial Orthopedics, 96(1), 52-61.",
        "Peck, H., et al. (2008). Facial asymmetry: prevalence and causes. American Journal of Orthodontics and Dentofacial Orthopedics, 133(2), 221-228.",
        "Arnett, G.W. & Bergman, R.T. (1993). Facial keys to orthodontic diagnosis and treatment planning. American Journal of Orthodontics, 103(4), 299-312.",
        "Kafi, R., et al. (2007). Improvement of naturally aged skin with topical retinol. Archives of Dermatology, 143(5), 606-612.",
        "Jefferson, Y. (2010). Mouth breathing: adverse effects on facial growth, health, academics, and behavior. General Dentistry, 58(1), 18-25.",
        "Barton, F., et al. (2017). Liquid rhinoplasty techniques. Aesthetic Surgery Journal, 37(3), 271-279.",
    ]
    for ref in bibliography:
        story.append(Paragraph(f"- {ref}", styles["CustomBody"]))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "This report was generated by the Facial Anthropometric Analysis Tool. "
        "All data is stored locally and is never transmitted externally.",
        styles["Disclaimer"],
    ))

    doc.build(story)
    return output_path


def _b64_to_image(b64_str: str, width: float | None = None, height: float | None = None):
    """Convert base64 string to ReportLab Image."""
    try:
        img_data = base64.b64decode(b64_str)
        img_stream = io.BytesIO(img_data)
        img = Image(img_stream)
        if width:
            img.drawWidth = width
        if height:
            img.drawHeight = height
        return img
    except Exception:
        return None
