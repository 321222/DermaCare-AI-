"""Generate a downloadable, clinician-reviewable PDF summary."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Mapping
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DISCLAIMER = (
    "This AI-generated report is for informational purposes only and is not a diagnosis or a substitute "
    "for professional medical advice. A clinician must assess symptoms and may need laboratory "
    "confirmation (for example PCR). Do not delay urgent care."
)


def _image_flowable(image: Any, width: float = 78 * mm, height: float = 65 * mm) -> ReportImage:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88)
    buffer.seek(0)
    return ReportImage(buffer, width=width, height=height, kind="proportional")


def build_pdf_report(
    prediction: str,
    confidence: float,
    probabilities: Mapping[str, float],
    triage_level: str,
    triage_action: str,
    symptom_summary: str,
    advice: str,
    original_image: Any | None = None,
    gradcam_image: Any | None = None,
) -> bytes:
    """Render a concise report entirely in memory."""
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="DermaCare AI - Consultation Summary",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DermaSmall", parent=styles["BodyText"], fontSize=8, leading=11,
        textColor=colors.HexColor("#40505a"),
    ))
    styles.add(ParagraphStyle(
        name="DermaHeading", parent=styles["Heading2"], textColor=colors.HexColor("#087f8c"),
    ))
    story: list[Any] = [
        Paragraph("DermaCare AI", styles["Title"]),
        Paragraph("Dr. Derma · AI skin-health assistant", styles["Heading2"]),
        Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["DermaSmall"]),
        Spacer(1, 5 * mm),
        Paragraph("Model output (not a diagnosis)", styles["DermaHeading"]),
        Paragraph(f"<b>Model suggestion:</b> {escape(prediction)}", styles["BodyText"]),
        Paragraph(f"<b>Confidence:</b> {confidence:.1%} (confidence is not clinical certainty)", styles["BodyText"]),
        Paragraph(f"<b>Rule-based triage:</b> {triage_level}", styles["BodyText"]),
        Paragraph(f"<b>Next step:</b> {escape(triage_action)}", styles["BodyText"]),
        Spacer(1, 3 * mm),
    ]
    probability_rows = [["Class", "Model probability"]]
    probability_rows.extend([[name, f"{value:.1%}"] for name, value in probabilities.items()])
    table = Table(probability_rows, colWidths=[90 * mm, 55 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4f4f3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#174c55")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d8dc")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.extend([
        table, Spacer(1, 4 * mm), Paragraph("Symptoms provided", styles["DermaHeading"]),
        Paragraph(escape(symptom_summary) if symptom_summary else "No symptom information provided.", styles["BodyText"]),
        Spacer(1, 3 * mm), Paragraph("General guidance", styles["DermaHeading"]),
        Paragraph(
            escape(advice) if advice else "Discuss the result with a qualified healthcare professional.",
            styles["BodyText"],
        ),
    ])
    if original_image is not None or gradcam_image is not None:
        story.extend([Spacer(1, 4 * mm), Paragraph("Images", styles["DermaHeading"])])
        image_cells = []
        if original_image is not None:
            image_cells.append([Paragraph("Uploaded image", styles["DermaSmall"]), _image_flowable(original_image)])
        if gradcam_image is not None:
            image_cells.append([Paragraph("Grad-CAM visualization", styles["DermaSmall"]), _image_flowable(gradcam_image)])
        if image_cells:
            story.append(Table([[cell[0] for cell in image_cells], [cell[1] for cell in image_cells]],
                               colWidths=[82 * mm] * len(image_cells)))
            story.append(Paragraph(
                "Grad-CAM is a coarse visualization of model attention, not an explanation of disease or a clinical finding.",
                styles["DermaSmall"],
            ))
    story.extend([
        Spacer(1, 7 * mm), Paragraph("Important disclaimer", styles["DermaHeading"]),
        Paragraph(DISCLAIMER, styles["DermaSmall"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Model limitations: performance can vary with skin tone diversity, image lighting, camera quality, "
            "body site, image framing, and dataset representation. This tool is not validated for clinical use.",
            styles["DermaSmall"],
        ),
    ])

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64747c"))
        canvas.drawString(18 * mm, 10 * mm, "Private summary · Review with a qualified clinician")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
