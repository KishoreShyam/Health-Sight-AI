import os
import io
from datetime import datetime
from PIL import Image as PILImage
import qrcode

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_qr_code_buffer(data_url):
    """Generates a QR code image buffer for a given URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=4,
        border=1,
    )
    qr.add_data(data_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1A365D", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def pil_to_bytes(pil_img):
    """Converts a PIL Image to a BytesIO PNG buffer."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf

def generate_pdf_report(
    patient_name="Anonymous Patient",
    age=50,
    gender="Unspecified",
    duration=6.0,
    pain_score=3.0,
    lesion_size=10.0,
    family_history="No",
    prediction_label="Benign",
    confidence=92.8,
    risk_level="Low",
    probabilities=None,
    original_img=None,
    gradcam_img=None,
    blob_url=None,
    azure_tags=None,
    speech_transcript=None,
    speech_language="English"
):
    """
    Generates a high-quality PDF medical report and returns it as BytesIO buffer.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=10
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2D3748')
    )

    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    elements = []

    # 1. Header Banner
    header_table = Table(
        [[
            Paragraph("<b>HEALTH SIGHT AI</b>", title_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>Report ID:</b> HS-{datetime.now().strftime('%M%S')}", ParagraphStyle('HeadRight', parent=body_style, alignment=2))
        ]],
        colWidths=[4.0 * inch, 3.25 * inch]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Paragraph("AI-Powered Multimodal Cancer Screening & Diagnostic Assessment", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2B6CB0'), spaceAfter=10))

    # 2. Patient Info & QR Code Row
    qr_url = blob_url if blob_url else "https://healthsightai.cognitiveservices.azure.com"
    qr_buf = generate_qr_code_buffer(qr_url)
    qr_image = Image(qr_buf, width=0.95*inch, height=0.95*inch)

    patient_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(patient_name), body_style),
            Paragraph("<b>Age / Duration:</b>", body_style), Paragraph(f"{age} yrs / {duration} mos", body_style),
            qr_image
        ],
        [
            Paragraph("<b>Pain Score (0-10):</b>", body_style), Paragraph(f"{pain_score}/10", body_style),
            Paragraph("<b>Lesion Size:</b>", body_style), Paragraph(f"{lesion_size} mm", body_style),
            ""
        ],
        [
            Paragraph("<b>Family History:</b>", body_style), Paragraph(str(family_history), body_style),
            Paragraph("<b>Verification QR:</b>", body_style), Paragraph("Scan for Cloud Verification", ParagraphStyle('TinyNote', parent=body_style, fontSize=7, textColor=colors.HexColor('#718096'))),
            ""
        ]
    ]

    patient_table = Table(patient_data, colWidths=[1.1*inch, 1.4*inch, 1.2*inch, 1.8*inch, 1.25*inch])
    patient_table.setStyle(TableStyle([
        ('SPAN', (4,0), (4,2)), # Span QR Code vertically
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (3,-1), 0.25, colors.HexColor('#EDF2F7')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    
    elements.append(Paragraph("<b>📋 Patient & Clinical Summary</b>", heading_style))
    elements.append(patient_table)
    elements.append(Spacer(1, 8))

    # 3. AI Prediction Result Card
    is_malignant = "malignant" in str(prediction_label).lower()
    bg_color = colors.HexColor('#FFF5F5') if is_malignant else colors.HexColor('#F0FFF4')
    border_color = colors.HexColor('#E53E3E') if is_malignant else colors.HexColor('#38A169')
    text_color = colors.HexColor('#C53030') if is_malignant else colors.HexColor('#276749')

    pred_heading_style = ParagraphStyle(
        'PredHeading',
        parent=title_style,
        fontSize=14,
        leading=16,
        textColor=text_color
    )

    pred_card_data = [
        [
            Paragraph(f"<b>Diagnosis Prediction:</b> {prediction_label.upper()}", pred_heading_style),
            Paragraph(f"<b>Confidence:</b> {confidence:.1f}%", ParagraphStyle('Conf', parent=bold_body_style, fontSize=11, textColor=text_color, alignment=2))
        ],
        [
            Paragraph(f"<b>Risk Classification:</b> {risk_level} Risk", bold_body_style),
            Paragraph(f"<b>Benign Prob:</b> {probabilities[0]*100:.1f}% | <b>Malignant Prob:</b> {probabilities[1]*100:.1f}%" if probabilities is not None else "", ParagraphStyle('Probs', parent=body_style, alignment=2))
        ]
    ]

    pred_table = Table(pred_card_data, colWidths=[4.25*inch, 3.0*inch])
    pred_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))

    elements.append(Paragraph("<b>🔬 AI Assessment & Risk Classification</b>", heading_style))
    elements.append(pred_table)
    elements.append(Spacer(1, 8))

    # 4. Visual Analysis (Original Image + Grad-CAM Heatmap)
    elements.append(Paragraph("<b>🖼️ Visual Feature Analysis (MobileNetV3 + Grad-CAM)</b>", heading_style))
    
    img_cells = []
    if original_img is not None:
        if isinstance(original_img, PILImage.Image):
            orig_buf = pil_to_bytes(original_img)
            img_cells.append([Image(orig_buf, width=2.8*inch, height=2.1*inch), Paragraph("<b>Original Uploaded Lesion Image</b>", ParagraphStyle('ImgCaption', parent=body_style, alignment=1))])
        else:
            img_cells.append(["[Original Image]", ""])
    else:
        img_cells.append(["[Original Image N/A]", ""])

    if gradcam_img is not None:
        if isinstance(gradcam_img, PILImage.Image):
            grad_buf = pil_to_bytes(gradcam_img)
            img_cells.append([Image(grad_buf, width=2.8*inch, height=2.1*inch), Paragraph("<b>Grad-CAM Explainable AI Heatmap</b>", ParagraphStyle('ImgCaption', parent=body_style, alignment=1))])
        else:
            img_cells.append(["[Grad-CAM N/A]", ""])
    else:
        img_cells.append(["[Grad-CAM N/A]", ""])

    if len(img_cells) == 2:
        img_table = Table(
            [
                [img_cells[0][0], img_cells[1][0]],
                [img_cells[0][1], img_cells[1][1]]
            ],
            colWidths=[3.6*inch, 3.6*inch]
        )
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,1), (-1,1), 4),
        ]))
        elements.append(img_table)

    elements.append(Spacer(1, 8))

    # 5. Azure Cloud Services Integration Details
    elements.append(Paragraph("<b>☁️ Microsoft Azure Services Summary</b>", heading_style))
    
    tags_str = ", ".join(azure_tags) if azure_tags else "None detected / Bypassed"
    transcript_str = speech_transcript if speech_transcript else "N/A"
    blob_str = blob_url if blob_url else "Local file execution"

    azure_data = [
        [Paragraph("<b>Azure Storage Blob URL:</b>", body_style), Paragraph(f"<font color='#2B6CB0'><u>{blob_str}</u></font>", body_style)],
        [Paragraph("<b>Azure AI Vision Tags:</b>", body_style), Paragraph(tags_str, body_style)],
        [Paragraph(f"<b>Voice Audio ({speech_language}):</b>", body_style), Paragraph(f"<i>\"{transcript_str}\"</i>", body_style)]
    ]

    azure_table = Table(azure_data, colWidths=[2.0*inch, 5.25*inch])
    azure_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(azure_table)
    elements.append(Spacer(1, 10))

    # 6. Disclaimer & Next Steps
    disclaimer_text = (
        "<b>Important Notice:</b> This report is generated by Health Sight AI, an AI-powered screening assistance tool. "
        "It is designed to support rural healthcare workers and PHC staff in identifying suspicious cases. "
        "<b>This report is NOT a final diagnostic order.</b> All findings must be clinically evaluated and confirmed by a licensed medical professional."
    )
    elements.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=body_style, fontSize=7.5, leading=9, textColor=colors.HexColor('#718096'))))

    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer
