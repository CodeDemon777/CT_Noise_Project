"""
PDF Report Generator Module
Generates professional clinical reports for CT scan noise analysis across all 4 models:
  - Model 1: U-Net++ (Gaussian & Poisson Noise)
  - Model 2: Attention U-Net (Poisson & Speckle Noise)
  - Model 3: DeepLabV3+ (Salt & Pepper & RVIN Noise)
  - Model 4: NoiseCNN (Quantization & Periodic Noise)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


class CTReportGenerator:
    """
    Generates professional medical reports for CT noise analysis across all 4 models.
    """

    @staticmethod
    def _detect_model_type(report_data: Dict[str, Any], model_name: Optional[str] = None) -> str:
        if model_name:
            mn = model_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if "model2" in mn or "attention" in mn:
                return "model2"
            if "model3" in mn or "deeplab" in mn:
                return "model3"
            if "model4" in mn or "cnn" in mn or "noisecnn" in mn:
                return "model4"
            if "model1" in mn or "unet" in mn:
                return "model1"

        # Auto-detection from report data structure
        if "salt_pepper" in str(report_data) or "rvin" in str(report_data) or report_data.get("model") == "Model 3":
            return "model3"
        if "speckle" in str(report_data) or report_data.get("model") == "Model 2":
            return "model2"
        if "quantization" in str(report_data) or "periodic" in str(report_data) or report_data.get("model") == "Model 4":
            return "model4"
        return "model1"

    @classmethod
    def generate_pdf(
        cls,
        output_pdf_path: str,
        original_img_path: str,
        annotated_img_path: str,
        report_data: Dict[str, Any],
        filename: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Generate a professional clinical PDF report for any of the 4 models.
        """
        pdf_path = Path(output_pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = cls._detect_model_type(report_data, model_name)

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

        # Theme Colors
        primary_color = colors.HexColor('#0f172a')   # Deep Slate
        bg_light = colors.HexColor('#f8fafc')        # Slate Light Background
        border_color = colors.HexColor('#cbd5e1')    # Slate Border
        success_color = colors.HexColor('#10b981')   # Emerald

        # Typography
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=14,
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
            spaceBefore=5,
            spaceAfter=3,
        )
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.8,
            textColor=colors.HexColor('#334155'),
            spaceBefore=1,
            spaceAfter=1,
            leading=10.5
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

        # Model Metadata Configurations
        if model_type == "model2":
            arch_title = "Model 2 (Attention U-Net)"
            protocol_name = "Attention U-Net Segmentation (Poisson + Speckle)"
            fig2_legend = "Figure 2: Annotated CT Scan (Red: Poisson | Green: Speckle)"
            valid_specs = "<b>Architecture:</b> Attention U-Net (Joshna.pth)<br/><b>Gating:</b> Additive Attention Gates (AG)<br/><b>Classes:</b> Poisson, Speckle Noise"
        elif model_type == "model3":
            arch_title = "Model 3 (DeepLabV3+)"
            protocol_name = "DeepLabV3+ Multi-Scale ASPP (Salt-Pepper + RVIN)"
            fig2_legend = "Figure 2: Annotated CT Scan (Amber: Salt & Pepper | Purple: RVIN)"
            valid_specs = "<b>Architecture:</b> DeepLabV3+ ASPP (Jahnavi.pth)<br/><b>Dilation Rates:</b> [1, 6, 12, 18] + Decoder<br/><b>Classes:</b> Salt-Pepper, RVIN Noise"
        elif model_type == "model4":
            arch_title = "Model 4 (NoiseCNN)"
            protocol_name = "NoiseCNN Classification & FFT Spectrum Analysis"
            fig2_legend = "Figure 2: Diagnostic View & 2D Fourier Magnitude Spectrum"
            valid_specs = "<b>Architecture:</b> NoiseCNN Deep Classifier (Vasanth.pth)<br/><b>Spectrum:</b> 2D Fast Fourier Transform (FFT)<br/><b>Classes:</b> Quantization, Periodic Noise"
        else:
            arch_title = "Model 1 (U-Net++)"
            protocol_name = "U-Net++ Nested Dense Segmentation (Gaussian + Poisson)"
            fig2_legend = "Figure 2: Annotated CT Scan (Red: Gaussian | Blue: Poisson)"
            valid_specs = "<b>Architecture:</b> U-Net++ Nested UNet (best_model.pth)<br/><b>Dice Score:</b> 0.9886 &nbsp;|&nbsp; <b>IoU:</b> 0.9778<br/><b>Classes:</b> Gaussian, Poisson Noise"

        # 1. Header Banner
        header_data = [
            [
                Paragraph(f"LungCT AI &nbsp;|&nbsp; CLINICAL NOISE REPORT ({arch_title})", title_style),
                Paragraph("<b>Date:</b> " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"<br/><b>Protocol:</b> {arch_title}", subtitle_style)
            ]
        ]
        header_table = Table(header_data, colWidths=[5.17*inch, 2.5*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 5))

        # 2. Metadata Block
        meta_data = [
            [
                Paragraph("Case / Scan ID:", meta_label_style), Paragraph(f"LCT-{abs(hash(filename)) % 100000:05d}", meta_value_style),
                Paragraph("Patient Case:", meta_label_style), Paragraph("ANONYMOUS (Clinical Study)", meta_value_style)
            ],
            [
                Paragraph("File Name:", meta_label_style), Paragraph(filename, meta_value_style),
                Paragraph("Analysis Model:", meta_label_style), Paragraph(protocol_name, meta_value_style)
            ],
            [
                Paragraph("Modality:", meta_label_style), Paragraph("Computed Tomography (CT)", meta_value_style),
                Paragraph("Report Status:", meta_label_style), Paragraph("<b>COMPLETED (AI-Generated)</b>", meta_value_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[1.3*inch, 2.535*inch, 1.3*inch, 2.535*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, border_color),
            ('PADDING', (0, 0), (-1, -1), 3.5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 5))

        # 3. Side-by-Side Images
        story.append(Paragraph("CT Scan Imagery & Diagnostic Overlay", h2_style))
        img_table_data = []
        col_widths = [3.835*inch, 3.835*inch]

        if os.path.exists(original_img_path) and os.path.exists(annotated_img_path):
            img_original = Image(original_img_path, width=2.45*inch, height=2.45*inch)
            img_annotated = Image(annotated_img_path, width=2.45*inch, height=2.45*inch)
            img_table_data = [
                [img_original, img_annotated],
                [
                    Paragraph("<b>Figure 1: Original CT Scan</b><br/>Input diagnostic image", meta_value_style),
                    Paragraph(f"<b>{fig2_legend}</b>", meta_value_style)
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
            ('PADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 4))

        # 4. Quantitative Results Table (Tailored per model)
        story.append(Paragraph(f"Quantitative Noise Classification &amp; Metrics ({arch_title})", h2_style))

        metric_headers = [
            Paragraph("<b>Noise Classification</b>", meta_label_style),
            Paragraph("<b>Pixel Count / Prob.</b>", meta_label_style),
            Paragraph("<b>Severity / Score</b>", meta_label_style),
            Paragraph("<b>Clinical Classification</b>", meta_label_style)
        ]

        if model_type == "model2":
            p_noise = report_data.get("noise", {}).get("poisson", {})
            s_noise = report_data.get("noise", {}).get("speckle", {})
            summary = report_data.get("summary", {})
            
            p_pct = p_noise.get("severity_percentage", 0.0)
            p_lvl = p_noise.get("severity_level", "NONE")
            p_px = p_noise.get("pixels", 0)

            s_pct = s_noise.get("severity_percentage", 0.0)
            s_lvl = s_noise.get("severity_level", "NONE")
            s_px = s_noise.get("pixels", 0)

            total_pct = summary.get("total_noise_percentage", 0.0)
            total_lvl = summary.get("total_noise_level", "NONE")
            clean_px = summary.get("clean_pixels", 0)

            metric_rows = [
                metric_headers,
                [
                    Paragraph("<font color='#f43f5e'><b>Poisson Noise (Photon Shot)</b></font>", meta_value_style),
                    Paragraph(f"{p_px:,} px", meta_value_style),
                    Paragraph(f"{p_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#f43f5e'><b>{p_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<font color='#10b981'><b>Speckle Noise (Multiplicative)</b></font>", meta_value_style),
                    Paragraph(f"{s_px:,} px", meta_value_style),
                    Paragraph(f"{s_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#10b981'><b>{s_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<b>Clean Signal Region</b>", meta_value_style),
                    Paragraph(f"{clean_px:,} px", meta_value_style),
                    Paragraph(f"{100.0 - total_pct:.2f}%", meta_value_style),
                    Paragraph("<font color='#10b981'>Normal / Target</font>", meta_value_style)
                ],
                [
                    Paragraph("<b>TOTAL NOISE (Model 2)</b>", meta_label_style),
                    Paragraph(f"{p_px + s_px:,} px", meta_label_style),
                    Paragraph(f"{total_pct:.2f}%", meta_label_style),
                    Paragraph(f"<b>{total_lvl}</b>", meta_label_style)
                ]
            ]

        elif model_type == "model3":
            sp_noise = report_data.get("noise", {}).get("salt_pepper", {})
            rvin_noise = report_data.get("noise", {}).get("rvin", {})
            summary = report_data.get("summary", {})

            sp_pct = sp_noise.get("severity_percentage", 0.0)
            sp_lvl = sp_noise.get("severity_level", "NONE")
            sp_px = sp_noise.get("pixels", 0)

            rvin_pct = rvin_noise.get("severity_percentage", 0.0)
            rvin_lvl = rvin_noise.get("severity_level", "NONE")
            rvin_px = rvin_noise.get("pixels", 0)

            total_pct = summary.get("total_noise_percentage", 0.0)
            total_lvl = summary.get("total_noise_level", "NONE")
            clean_px = summary.get("clean_pixels", 0)

            metric_rows = [
                metric_headers,
                [
                    Paragraph("<font color='#f59e0b'><b>Salt &amp; Pepper Noise (Impulse)</b></font>", meta_value_style),
                    Paragraph(f"{sp_px:,} px", meta_value_style),
                    Paragraph(f"{sp_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#f59e0b'><b>{sp_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<font color='#8b5cf6'><b>RVIN (Random-Valued Impulse)</b></font>", meta_value_style),
                    Paragraph(f"{rvin_px:,} px", meta_value_style),
                    Paragraph(f"{rvin_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#8b5cf6'><b>{rvin_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<b>Clean Signal Region</b>", meta_value_style),
                    Paragraph(f"{clean_px:,} px", meta_value_style),
                    Paragraph(f"{100.0 - total_pct:.2f}%", meta_value_style),
                    Paragraph("<font color='#10b981'>Normal / Target</font>", meta_value_style)
                ],
                [
                    Paragraph("<b>TOTAL NOISE (Model 3)</b>", meta_label_style),
                    Paragraph(f"{sp_px + rvin_px:,} px", meta_label_style),
                    Paragraph(f"{total_pct:.2f}%", meta_label_style),
                    Paragraph(f"<b>{total_lvl}</b>", meta_label_style)
                ]
            ]

        elif model_type == "model4":
            pred_class = report_data.get("predicted_class", "Clean")
            confidence = report_data.get("confidence", 0.0)
            q_noise = report_data.get("noise", {}).get("quantization", {})
            p_noise = report_data.get("noise", {}).get("periodic", {})
            summary = report_data.get("summary", {})

            q_pct = q_noise.get("severity_percentage", 0.0)
            q_lvl = q_noise.get("severity_level", "NONE")
            p_pct = p_noise.get("severity_percentage", 0.0)
            p_lvl = p_noise.get("severity_level", "NONE")
            total_pct = summary.get("total_noise_percentage", 0.0)
            total_lvl = summary.get("total_noise_level", "NONE")

            metric_rows = [
                metric_headers,
                [
                    Paragraph("<font color='#eab308'><b>Quantization Noise (ADC Step)</b></font>", meta_value_style),
                    Paragraph(f"Probability: {q_pct:.1f}%", meta_value_style),
                    Paragraph(f"{q_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#eab308'><b>{q_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<font color='#8b5cf6'><b>Periodic Noise (FFT Spike)</b></font>", meta_value_style),
                    Paragraph(f"Probability: {p_pct:.1f}%", meta_value_style),
                    Paragraph(f"{p_pct:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#8b5cf6'><b>{p_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<b>Scan Status / Confidence</b>", meta_value_style),
                    Paragraph(f"Predicted: {pred_class}", meta_value_style),
                    Paragraph(f"{confidence:.1f}% Conf.", meta_value_style),
                    Paragraph(f"<b>{pred_class.upper()}</b>", meta_value_style)
                ],
                [
                    Paragraph("<b>TOTAL NOISE PROBABILITY (Model 4)</b>", meta_label_style),
                    Paragraph(f"Noise Likelihood", meta_label_style),
                    Paragraph(f"{total_pct:.2f}%", meta_label_style),
                    Paragraph(f"<b>{total_lvl}</b>", meta_label_style)
                ]
            ]

        else: # Model 1 Default
            gaussian_perc = report_data.get("gaussian", {}).get("percentage", 0.0)
            gaussian_lvl = report_data.get("gaussian", {}).get("level", "None")
            gaussian_pixels = report_data.get("gaussian", {}).get("pixels", 0)

            poisson_perc = report_data.get("poisson", {}).get("percentage", 0.0)
            poisson_lvl = report_data.get("poisson", {}).get("level", "None")
            poisson_pixels = report_data.get("poisson", {}).get("pixels", 0)

            total_perc = report_data.get("summary", {}).get("total_noise_percentage", 0.0)
            total_lvl = report_data.get("summary", {}).get("total_noise_level", "None")
            total_pct = total_perc
            clean_pixels = report_data.get("summary", {}).get("clean_pixels", 0)

            metric_rows = [
                metric_headers,
                [
                    Paragraph("<font color='#f43f5e'><b>Gaussian Noise (Electronic)</b></font>", meta_value_style),
                    Paragraph(f"{gaussian_pixels:,} px", meta_value_style),
                    Paragraph(f"{gaussian_perc:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#f43f5e'><b>{gaussian_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<font color='#38bdf8'><b>Poisson Noise (Photon Shot)</b></font>", meta_value_style),
                    Paragraph(f"{poisson_pixels:,} px", meta_value_style),
                    Paragraph(f"{poisson_perc:.2f}%", meta_value_style),
                    Paragraph(f"<font color='#38bdf8'><b>{poisson_lvl}</b></font>", meta_value_style)
                ],
                [
                    Paragraph("<b>Clean Signal Region</b>", meta_value_style),
                    Paragraph(f"{clean_pixels:,} px", meta_value_style),
                    Paragraph(f"{100.0 - total_perc:.2f}%", meta_value_style),
                    Paragraph("<font color='#10b981'>Normal / Target</font>", meta_value_style)
                ],
                [
                    Paragraph("<b>TOTAL NOISE (Model 1)</b>", meta_label_style),
                    Paragraph(f"{gaussian_pixels + poisson_pixels:,} px", meta_label_style),
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
            ('PADDING', (0, 0), (-1, -1), 3.5),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metric_table)
        story.append(Spacer(1, 5))

        # 5. Clinical Diagnostic Interpretation
        story.append(Paragraph("Clinical Interpretation & Denoising Recommendations", h2_style))

        overall_severity = total_lvl.capitalize() if isinstance(total_lvl, str) else "None"
        if total_pct < 5.0 or overall_severity in ["None", "Mild"]:
            clinical_findings = f"Scan demonstrates optimal structural integrity with minimal noise artifact coverage ({total_pct:.1f}%). High diagnostic clarity across lung parenchyma."
            rec_text = "Scan is approved for primary diagnostic review. Standard linear smoothing is optional if examining micro-vascular structures."
            sev_bar_color = success_color
            bg_sev = colors.HexColor('#f0fdf4')
        elif total_pct < 15.0 or overall_severity == "Moderate":
            clinical_findings = f"Moderate noise contamination identified ({total_pct:.1f}%). Low-contrast tissue borders and ground-glass opacities may exhibit minor graininess."
            rec_text = "Recommend applying edge-preserving bilateral or non-local means denoising filters prior to diagnostic sign-off."
            sev_bar_color = colors.HexColor('#f59e0b')
            bg_sev = colors.HexColor('#fffbeb')
        else:
            clinical_findings = f"Elevated noise artifacts detected ({total_pct:.1f}%). Structural detail is noticeably compromised by noise patterns."
            rec_text = "Strongly recommend executing deep neural restoration or wavelet-based denoising. If diagnostic ambiguities persist, evaluate sensor hardware."
            sev_bar_color = colors.HexColor('#f43f5e')
            bg_sev = colors.HexColor('#fff1f2')

        summary_p = Paragraph(
            f"<b>Model Assessment ({arch_title}):</b> {clinical_findings}<br/>"
            f"<b>Radiologist Action:</b> {rec_text}",
            body_style
        )
        summary_table = Table([[summary_p]], colWidths=[7.67*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_sev),
            ('LINEBEFORE', (0, 0), (0, -1), 3.5, sev_bar_color),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 5))

        # 6. Radiologist Signature Block & AI Validation
        sig_data = [
            [
                Paragraph("<b>Radiologist Review Block</b>", meta_label_style),
                Paragraph("<b>AI System Validation Specs</b>", meta_label_style)
            ],
            [
                Paragraph("<br/><br/>________________________________________<br/>Reviewing Radiologist Signature", meta_value_style),
                Paragraph(valid_specs, meta_value_style)
            ]
        ]
        sig_table = Table(sig_data, colWidths=[3.835*inch, 3.835*inch])
        sig_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('BACKGROUND', (0, 0), (-1, 0), bg_light),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, border_color),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(KeepTogether([sig_table]))
        story.append(Spacer(1, 4))

        # 7. Disclaimer
        disclaimer_text = (
            f"<i>Disclaimer: This report was automatically generated by {arch_title} "
            "for multi-noise classification and severity assessment in educational & research environments. "
            "Findings should be cross-referenced by a qualified medical professional prior to clinical intervention.</i>"
        )
        story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=6.5, textColor=colors.HexColor('#64748b'), leading=8.5)))

        doc.build(story)
        print(f"✅ Generated clinical PDF report for {arch_title} at: {pdf_path}")
        return str(pdf_path)


if __name__ == "__main__":
    print("Report Generator Module compiled successfully.")
