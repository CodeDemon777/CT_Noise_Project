"""
PDF Report Generator Module
Generates professional clinical reports for CT scan noise analysis
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


class CTReportGenerator:
    """
    Generates professional medical reports for CT noise analysis
    """
    
    @staticmethod
    def generate_pdf(
        output_pdf_path: str,
        original_img_path: str,
        annotated_img_path: str,
        report_data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Generate a professional clinical PDF report
        
        Args:
            output_pdf_path: Path where the PDF should be saved
            original_img_path: Path to the original CT scan image
            annotated_img_path: Path to the annotated result image
            report_data: The severity report dictionary from SeverityCalculator
            filename: The original filename of the CT scan
            
        Returns:
            Path to the generated PDF
        """
        # Ensure directories exist
        pdf_path = Path(output_pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up document with 30pt margins for a single page layout
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Colors (Matching Premium Slate Web Theme)
        primary_color = colors.HexColor('#0f172a')   # Deep Slate
        secondary_color = colors.HexColor('#38bdf8') # Sky Blue (Poisson)
        accent_color = colors.HexColor('#f43f5e')    # Rose Pink (Gaussian)
        success_color = colors.HexColor('#10b981')   # Emerald (Clean)
        bg_light = colors.HexColor('#f8fafc')        # Slate Light Background
        border_color = colors.HexColor('#cbd5e1')    # Slate Border
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=15,
            textColor=colors.white,
            alignment=0,
            spaceAfter=2
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=0
        )
        
        h2_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=primary_color,
            spaceBefore=6,
            spaceAfter=3,
            borderPadding=2
        )
        
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#334155'),
            spaceBefore=1,
            spaceAfter=1,
            leading=11
        )
        
        meta_label_style = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=primary_color
        )
        
        meta_value_style = ParagraphStyle(
            'MetaValue',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#334155')
        )
        
        # 1. Header Banner (Deep Slate Panel)
        header_data = [
            [
                Paragraph("LungCT AI &nbsp;|&nbsp; CLINICAL NOISE ASSESSMENT REPORT", title_style),
                Paragraph("<b>Date:</b> " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "<br/><b>System:</b> v1.0.0 (U-Net++ model)", subtitle_style)
            ]
        ]
        
        header_table = Table(header_data, colWidths=[5.17*inch, 2.5*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6))
        
        # 2. Metadata / Scan Information Block
        meta_data = [
            [
                Paragraph("Case / Scan ID:", meta_label_style), Paragraph(f"LCT-{hash(filename) % 100000:05d}", meta_value_style),
                Paragraph("Patient Name:", meta_label_style), Paragraph("ANONYMOUS (Clinical Study)", meta_value_style)
            ],
            [
                Paragraph("Original File Name:", meta_label_style), Paragraph(filename, meta_value_style),
                Paragraph("Analysis Protocol:", meta_label_style), Paragraph("U-Net++ Noise Segmentation", meta_value_style)
            ],
            [
                Paragraph("Scan Modality:", meta_label_style), Paragraph("Computed Tomography (CT)", meta_value_style),
                Paragraph("Report Status:", meta_label_style), Paragraph("<b>COMPLETED (AI-Generated)</b>", meta_value_style)
            ]
        ]
        
        meta_table = Table(meta_data, colWidths=[1.3*inch, 2.535*inch, 1.3*inch, 2.535*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, border_color),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6))
        
        # 3. Side-by-Side Images (Resized to 2.6 inches to fit on one page)
        story.append(Paragraph("CT Scan Imagery Analysis", h2_style))
        
        img_table_data = []
        col_widths = [3.835*inch, 3.835*inch]
        
        if os.path.exists(original_img_path) and os.path.exists(annotated_img_path):
            img_original = Image(original_img_path, width=2.6*inch, height=2.6*inch)
            img_annotated = Image(annotated_img_path, width=2.6*inch, height=2.6*inch)
            
            img_table_data = [
                [img_original, img_annotated],
                [
                    Paragraph("<font color='#0f172a'><b>Figure 1: Original CT Scan</b><br/>Input diagnostic image</font>", meta_value_style),
                    Paragraph("<font color='#0f172a'><b>Figure 2: Annotated CT Scan</b><br/>Red: Gaussian Noise | Blue: Poisson Noise</font>", meta_value_style)
                ]
            ]
        else:
            img_table_data = [
                [
                    Paragraph("<i>Image file missing: Original image could not be loaded.</i>", meta_value_style),
                    Paragraph("<i>Image file missing: Annotated image could not be loaded.</i>", meta_value_style)
                ]
            ]
            
        img_table = Table(img_table_data, colWidths=col_widths)
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 4))
        
        # 4. Tabulated Results
        story.append(Paragraph("Quantitative Noise Classification & Severity Metrics", h2_style))
        
        # Get data
        gaussian_perc = report_data.get("gaussian", {}).get("percentage", 0.0)
        gaussian_lvl = report_data.get("gaussian", {}).get("level", "None")
        gaussian_pixels = report_data.get("gaussian", {}).get("pixels", 0)
        
        poisson_perc = report_data.get("poisson", {}).get("percentage", 0.0)
        poisson_lvl = report_data.get("poisson", {}).get("level", "None")
        poisson_pixels = report_data.get("poisson", {}).get("pixels", 0)
        
        total_perc = report_data.get("summary", {}).get("total_noise_percentage", 0.0)
        total_lvl = report_data.get("summary", {}).get("total_noise_level", "None")
        total_pixels = report_data.get("summary", {}).get("total_pixels", 0)
        clean_pixels = report_data.get("summary", {}).get("clean_pixels", 0)
        
        # Metric Table headers
        metric_headers = [
            Paragraph("<b>Noise Classification</b>", meta_label_style),
            Paragraph("<b>Pixel Count</b>", meta_label_style),
            Paragraph("<b>Area Coverage (%)</b>", meta_label_style),
            Paragraph("<b>Severity Classification</b>", meta_label_style)
        ]
        
        metric_rows = [
            metric_headers,
            [
                Paragraph("<font color='#f43f5e'><b>Gaussian Noise (Electronic)</b></font>", meta_value_style),
                Paragraph(f"{gaussian_pixels:,}", meta_value_style),
                Paragraph(f"{gaussian_perc:.2f}%", meta_value_style),
                Paragraph(f"<font color='#f43f5e'><b>{gaussian_lvl}</b></font>" if gaussian_lvl != "None" else f"<b>{gaussian_lvl}</b>", meta_value_style)
            ],
            [
                Paragraph("<font color='#38bdf8'><b>Poisson Noise (Photon Shot)</b></font>", meta_value_style),
                Paragraph(f"{poisson_pixels:,}", meta_value_style),
                Paragraph(f"{poisson_perc:.2f}%", meta_value_style),
                Paragraph(f"<font color='#38bdf8'><b>{poisson_lvl}</b></font>" if poisson_lvl != "None" else f"<b>{poisson_lvl}</b>", meta_value_style)
            ],
            [
                Paragraph("<font color='#10b981'><b>Clean Regions (Signal)</b></font>", meta_value_style),
                Paragraph(f"{clean_pixels:,}", meta_value_style),
                Paragraph(f"{100.0 - total_perc:.2f}%", meta_value_style),
                Paragraph("<font color='#10b981'>Normal / Target</font>", meta_value_style)
            ],
            [
                Paragraph("<b>TOTAL NOISE (Artifacts)</b>", meta_label_style),
                Paragraph(f"{gaussian_pixels + poisson_pixels:,}", meta_label_style),
                Paragraph(f"{total_perc:.2f}%", meta_label_style),
                Paragraph(f"<b>{total_lvl}</b>", meta_label_style)
            ]
        ]
        
        metric_table = Table(metric_rows, colWidths=[3.0*inch, 1.3*inch, 1.5*inch, 1.87*inch])
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, border_color),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metric_table)
        story.append(Spacer(1, 6))
        
        # 5. Diagnostic Interpretation Summary & Recommendations
        story.append(Paragraph("Clinical Interpretation & Recommendations", h2_style))
        
        overall_severity = total_lvl
        rec_text = ""
        clinical_description = ""
        severity_color = success_color
        bg_highlight = colors.HexColor('#f0fdf4') # default light green
        
        if overall_severity == "None":
            clinical_description = (
                "The AI segmentation model detected no significant noise artifacts in the uploaded lung CT scan. "
                "The signal-to-noise ratio (SNR) is optimal, and the image exhibits excellent structural integrity."
            )
            rec_text = "The scan is suitable for full clinical diagnostic evaluation. No additional filtering or post-processing is required."
            severity_color = success_color
            bg_highlight = colors.HexColor('#f0fdf4')
        elif overall_severity == "Mild":
            clinical_description = (
                f"The AI model detected mild noise coverage ({total_perc:.2f}% of total pixels), "
                f"consisting of {gaussian_perc:.2f}% Gaussian electronic noise and {poisson_perc:.2f}% Poisson photon shot noise. "
                "The noise is primarily confined to non-critical peripheral tissues and does not severely compromise diagnostic quality."
            )
            rec_text = "Standard visual inspection is acceptable. Standard linear smoothing filters may be applied if micro-structures are evaluated."
            severity_color = success_color
            bg_highlight = colors.HexColor('#f0fdf4')
        elif overall_severity == "Moderate":
            clinical_description = (
                f"The analysis indicates moderate noise artifacts ({total_perc:.2f}% coverage). "
                f"Gaussian noise is recorded at {gaussian_perc:.2f}% ({gaussian_lvl}), and Poisson noise at {poisson_perc:.2f}% ({poisson_lvl}). "
                "Some low-contrast diagnostic features (such as subtle ground-glass opacities or micro-nodules) may be partially obscured."
            )
            rec_text = "Recommend applying edge-preserving denoising algorithms (e.g., bilateral filter or non-local means denoising) before final diagnostic signing. Exercise caution in interpreting fine textures."
            severity_color = colors.HexColor('#f59e0b') # Amber orange
            bg_highlight = colors.HexColor('#fffbeb')
        elif overall_severity == "Severe":
            clinical_description = (
                f"Significant noise degradation is detected, covering {total_perc:.2f}% of the scan area. "
                f"Gaussian noise is {gaussian_perc:.2f}% ({gaussian_lvl}) and Poisson noise is {poisson_perc:.2f}% ({poisson_lvl}). "
                "Image diagnostic quality is substantially degraded, and artifact interference may lead to high false-positive or false-negative nodules detection."
            )
            rec_text = "Denoising preprocessing is strongly advised. Advanced CNN-based image restoration or wavelet-based denoising should be executed. If diagnostic clarity remains compromised, consider scheduling a follow-up low-noise scan."
            severity_color = accent_color # Rose Red
            bg_highlight = colors.HexColor('#fff1f2')
        else: # Critical
            clinical_description = (
                f"CRITICAL noise levels are present in the CT scan, affecting {total_perc:.2f}% of the pixel space. "
                f"Gaussian noise is {gaussian_perc:.2f}% ({gaussian_lvl}) and Poisson noise is {poisson_perc:.2f}% ({poisson_lvl}). "
                "The signal is severely compromised, rendering the scan clinically unreliable for accurate diagnostic interpretation."
            )
            rec_text = "This scan should be flagged as sub-diagnostic. Recommend immediate patient rescan with adjusted dosage/calibration parameters, or review of the imaging hardware for systemic sensor noise."
            severity_color = colors.HexColor('#7c3aed') # Purple
            bg_highlight = colors.HexColor('#f5f3ff')

        summary_p = Paragraph(
            f"<b>Model Findings:</b> {clinical_description}<br/>"
            f"<b>Denoising Recommendations:</b> {rec_text}",
            body_style
        )
        
        summary_table = Table([[summary_p]], colWidths=[7.67*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_highlight),
            ('LINEBEFORE', (0, 0), (0, -1), 3.5, severity_color),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 6))
        
        # 6. Radiologist Signature Block (Review Panel)
        sig_data = [
            [
                Paragraph("<b>Radiologist Review Block</b>", meta_label_style),
                Paragraph("<b>AI System Validation Specs</b>", meta_label_style)
            ],
            [
                Paragraph("<br/><br/>________________________________________<br/>Reviewing Radiologist Signature", meta_value_style),
                Paragraph("<b>Model Architecture:</b> U-Net++ (Nested UNet)<br/><b>Dice Score:</b> 0.9886 &nbsp;|&nbsp; <b>IoU:</b> 0.9778<br/><b>Precision:</b> 0.9899 &nbsp;|&nbsp; <b>Recall:</b> 0.9875", meta_value_style)
            ]
        ]
        
        sig_table = Table(sig_data, colWidths=[3.835*inch, 3.835*inch])
        sig_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('BACKGROUND', (0, 0), (-1, 0), bg_light),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, border_color),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(KeepTogether([sig_table]))
        story.append(Spacer(1, 6))
        
        # 7. Disclaimer / Footer
        disclaimer_text = (
            "<i>Disclaimer: This report is automatically generated by an artificial intelligence model (U-Net++) "
            "for multi-noise classification and severity assessment. It is developed as a Software Engineering Capstone Project. "
            "The findings should be reviewed and verified by a licensed radiologist or healthcare professional before clinical decisions are made. "
            "All results are for research and educational validation purposes only.</i>"
        )
        disclaimer_p = Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=6.5, textColor=colors.HexColor('#64748b'), leading=9))
        story.append(disclaimer_p)
        
        # Build Document
        doc.build(story)
        print(f"✅ Generated clinical PDF report at: {pdf_path}")
        return str(pdf_path)


if __name__ == "__main__":
    # Quick visual check of code compilation
    print("Report Generator Module compiled successfully.")
