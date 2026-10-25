import os
import io
import json
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_session_pdf(session_data: Dict[str, Any], output_path: Optional[str] = None) -> bytes:
    """
    Generate a comprehensive, publication-quality rehabilitation assessment report in PDF format.
    """
    buffer = io.BytesIO() if output_path is None else None
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0A1128') # Dark Navy
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0077B6') # Deep Teal
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0A1128'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#222222')
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#C92A2A') # Dark Red
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("RehabVision AI — Clinical Movement Report", title_style))
    story.append(Paragraph("Automated Biomechanical Assessment & Deep Learning Analysis System", subtitle_style))
    story.append(Spacer(1, 8))

    # Educational & Clinical Disclaimer Box
    disclaimer_text = (
        "<b>MANDATORY NOTICE:</b> Educational research prototype for 7th-semester deep learning evaluation. "
        "This automated report is NOT validated for medical diagnosis, clinical decision-making, or determining exercise safety. "
        "All biomechanical recommendations and metrics require review by a licensed physiotherapist."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[530])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF5F5')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#FF8787')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 12))

    # 2. Session Summary Metadata Table
    angle_stats = session_data.get("angle_stats", {})
    if isinstance(angle_stats, str):
        try:
            angle_stats = json.loads(angle_stats)
        except:
            angle_stats = {}

    joint_deviations = session_data.get("joint_deviations", {})
    if isinstance(joint_deviations, str):
        try:
            joint_deviations = json.loads(joint_deviations)
        except:
            joint_deviations = {}

    ref_id = session_data.get("reference_id", "Default Baseline")
    ref_type = session_data.get("reference_type", "standard")
    ref_type_display = "Patient-Specific (Clinician Approved)" if ref_type == "patient_specific" else "Reusable Standard Clinical Baseline"
    ref_version = session_data.get("reference_version", "1.0")
    ref_provenance = session_data.get("reference_provenance") or ("Physiotherapy Clinical Protocol v1.0" if ref_type == "standard" else "Approved Clinician Protocol")
    patient_display = session_data.get("patient_id") or "General / Non-isolated"

    summary_data = [
        [
            Paragraph("<b>Session ID:</b>", body_style),
            Paragraph(str(session_data.get("id", "N/A")), body_style),
            Paragraph("<b>Timestamp:</b>", body_style),
            Paragraph(str(session_data.get("timestamp", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Target Exercise:</b>", body_style),
            Paragraph(str(session_data.get("exercise_name", "Standing Right-Arm Shoulder Abduction")), body_style),
            Paragraph("<b>Patient Identifier:</b>", body_style),
            Paragraph(str(patient_display), body_style),
        ],
        [
            Paragraph("<b>Repetitions Completed:</b>", body_style),
            Paragraph(str(session_data.get("repetitions_completed", 0)), body_style),
            Paragraph("<b>Duration:</b>", body_style),
            Paragraph(f"{float(session_data.get('duration_seconds', 0.0)):.1f} seconds", body_style),
        ],
        [
            Paragraph("<b>Reference ID:</b>", body_style),
            Paragraph(f"<b>{ref_id}</b> (v{ref_version})", body_style),
            Paragraph("<b>Reference Type:</b>", body_style),
            Paragraph(f"<b>{ref_type_display}</b>", body_style),
        ],
        [
            Paragraph("<b>Reference Provenance:</b>", body_style),
            Paragraph(f"<i>{ref_provenance}</i>", body_style),
            Paragraph("<b>DTW Similarity Index:</b>", body_style),
            Paragraph(f"<b>{float(session_data.get('comparison_score', 0.0)):.1f} / 100</b>", body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[125, 140, 125, 140])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E9ECEF')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("Session Metadata & Provenance Overview", section_heading))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 3. Biomechanical Measurements Table
    bio_data = [
        [
            Paragraph("<b>Biomechanical Metric</b>", body_style),
            Paragraph("<b>Recorded Value</b>", body_style),
            Paragraph("<b>Reference Target</b>", body_style),
            Paragraph("<b>Clinical Notes</b>", body_style),
        ],
        [
            Paragraph("Peak Shoulder Elevation", body_style),
            Paragraph(f"{float(angle_stats.get('peak_elevation', 0.0)):.1f}°", body_style),
            Paragraph("80.0° - 105.0°", body_style),
            Paragraph("Optimal functional abduction height", body_style),
        ],
        [
            Paragraph("Active Range of Motion (ROM)", body_style),
            Paragraph(f"{float(angle_stats.get('rom', 0.0)):.1f}°", body_style),
            Paragraph(">= 70.0°", body_style),
            Paragraph("Ascent from resting coronal line", body_style),
        ],
        [
            Paragraph("Mean Elbow Angle", body_style),
            Paragraph(f"{float(angle_stats.get('mean_elbow', 170.0)):.1f}°", body_style),
            Paragraph("165.0° - 180.0°", body_style),
            Paragraph("Straight arm posture (anti-flexion)", body_style),
        ],
        [
            Paragraph("Mean Elevation Deviation", body_style),
            Paragraph(f"{float(joint_deviations.get('mean_elevation_deviation', 0.0)):.1f}°", body_style),
            Paragraph("< 10.0°", body_style),
            Paragraph("Dynamic Time Warping trajectory difference", body_style),
        ],
        [
            Paragraph("Root Mean Square Error (RMSE)", body_style),
            Paragraph(f"{float(joint_deviations.get('rmse', 0.0)):.1f}°", body_style),
            Paragraph("< 12.0°", body_style),
            Paragraph("Overall curve variance vs reference", body_style),
        ],
    ]
    bio_table = Table(bio_data, colWidths=[160, 100, 110, 160])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7F5FF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#A5D8FF')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0EBFF')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("Biomechanical Kinematic Measurements", section_heading))
    story.append(bio_table)
    story.append(Spacer(1, 10))

    # 4. Optional Trajectory Plot Chart
    raw_traj = session_data.get("raw_trajectory")
    if isinstance(raw_traj, str):
        try:
            raw_traj = json.loads(raw_traj)
        except:
            raw_traj = None

    if raw_traj and len(raw_traj) >= 5:
        try:
            times = [p.get("timestamp", idx * 0.05) for idx, p in enumerate(raw_traj)]
            elevations = [p.get("elevation", 0.0) for p in raw_traj]
            elbows = [p.get("elbow", 170.0) for p in raw_traj]

            fig, ax1 = plt.subplots(figsize=(6.5, 2.3), dpi=150)
            ax1.set_title("Recorded Motion Kinematics (Shoulder Elevation & Elbow Angle)", fontsize=9, fontweight='bold', color='#0A1128')
            ax1.plot(times, elevations, color='#0077B6', linewidth=2, label="Right Arm Elevation (°)")
            ax1.axhline(90.0, color='#2B8A3E', linestyle='--', alpha=0.7, label="Target 90° Plane")
            ax1.set_xlabel("Time (seconds)", fontsize=8)
            ax1.set_ylabel("Arm Elevation (°)", fontsize=8, color='#0077B6')
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.tick_params(axis='both', labelsize=7)

            ax2 = ax1.twinx()
            ax2.plot(times, elbows, color='#E8590C', linewidth=1.5, linestyle='-.', label="Elbow Extension (°)")
            ax2.set_ylabel("Elbow (°)", fontsize=8, color='#E8590C')
            ax2.tick_params(axis='both', labelsize=7)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close(fig)
            img_buf.seek(0)

            chart_img = Image(img_buf, width=6.5 * inch, height=2.3 * inch)
            story.append(Paragraph("Kinematic Trajectory Visualization", section_heading))
            story.append(chart_img)
            story.append(Spacer(1, 10))
        except Exception as e:
            print("Chart generation skipped:", e)

    # 5. Hybrid Assessment & Ensemble Decision Section
    story.append(Paragraph("Hybrid Assessment & Ensemble Decision", section_heading))

    stat_assessment = session_data.get("statistical_assessment")
    if isinstance(stat_assessment, str):
        try:
            stat_assessment = json.loads(stat_assessment)
        except:
            stat_assessment = None

    model_assessment = session_data.get("model_assessment")
    if isinstance(model_assessment, str):
        try:
            model_assessment = json.loads(model_assessment)
        except:
            model_assessment = None

    ensemble_dec = session_data.get("ensemble_decision")
    if isinstance(ensemble_dec, str):
        try:
            ensemble_dec = json.loads(ensemble_dec)
        except:
            ensemble_dec = None

    lstm_info = session_data.get("lstm_prediction")
    if isinstance(lstm_info, str):
        try:
            lstm_info = json.loads(lstm_info)
        except:
            lstm_info = None

    # Construct 3-card table for Statistical, Deep Learning, and Ensemble
    # Card 1: Statistical Branch
    if stat_assessment:
        stat_sub = stat_assessment.get("sub_scores", {})
        stat_html = (
            f"<b>STATISTICAL BRANCH</b><br/>"
            f"<b>Score:</b> {stat_assessment.get('score', 0.0):.1f} / 100<br/>"
            f"<b>Classification:</b> {stat_assessment.get('classification_estimate', 'UNKNOWN')}<br/>"
            f"<b>Quality:</b> {stat_assessment.get('quality_status', 'VALID')}<br/>"
            f"<b>Sub-scores:</b> ROM: {stat_sub.get('rom', 0.0)*100:.0f}%, "
            f"DTW: {stat_sub.get('dtw_similarity', 0.0)*100:.0f}%, "
            f"Posture: {stat_sub.get('posture_compliance', 0.0)*100:.0f}%, "
            f"Cadence: {stat_sub.get('cadence', 0.0)*100:.0f}%"
        )
    else:
        stat_html = (
            f"<b>STATISTICAL BRANCH</b><br/>"
            f"<b>DTW Score:</b> {float(session_data.get('comparison_score', 0.0)):.1f} / 100<br/>"
            f"<b>RMSE:</b> {float(joint_deviations.get('rmse', 0.0)):.1f}°<br/>"
            f"<b>Status:</b> Biomechanical reference trajectory comparison"
        )

    # Card 2: Deep Learning Model Branch
    if model_assessment:
        m_status = model_assessment.get("input_quality_status", "VALID")
        if m_status == "VALID" and model_assessment.get("class_label"):
            probs = model_assessment.get("probabilities", {})
            p_corr = probs.get("correct", 0.0) if probs else 0.0
            p_inc = probs.get("incorrect", 0.0) if probs else 0.0
            model_html = (
                f"<b>DEEP LEARNING BRANCH</b><br/>"
                f"<b>Model:</b> {model_assessment.get('model_id', 'rehab-lstm')}<br/>"
                f"<b>Schema:</b> {model_assessment.get('input_schema', 'mocap_3d_64x80')}<br/>"
                f"<b>Prediction:</b> {model_assessment.get('class_label', '').upper()}<br/>"
                f"<b>Probabilities:</b> Correct {p_corr:.2f} | Incorrect {p_inc:.2f}<br/>"
                f"<b>Latency:</b> {model_assessment.get('inference_latency_ms', 0.0):.1f} ms"
            )
        else:
            model_html = (
                f"<b>DEEP LEARNING BRANCH</b><br/>"
                f"<b>Status:</b> {m_status}<br/>"
                f"<b>Model:</b> {model_assessment.get('model_id', 'rehab-lstm-v1')}<br/>"
                f"<b>Guard:</b> Monocular MediaPipe webcam inputs safely isolated from 3D mocap model to prevent spurious predictions."
            )
    elif lstm_info:
        model_html = (
            f"<b>DEEP LEARNING BRANCH</b><br/>"
            f"<b>Source:</b> {lstm_info.get('source', 'Compatible Sequence')}<br/>"
            f"<b>Prediction:</b> {lstm_info.get('class_label', 'Unknown').upper()} (Class {lstm_info.get('predicted_class')})<br/>"
            f"<b>Probabilities:</b> Correct: {lstm_info.get('probabilities', {}).get('correct', 0.0):.2f}, "
            f"Incorrect: {lstm_info.get('probabilities', {}).get('incorrect', 0.0):.2f}"
        )
    else:
        model_html = (
            f"<b>DEEP LEARNING BRANCH</b><br/>"
            f"<b>Status:</b> input_requires_validation<br/>"
            f"<b>Reason:</b> Live MediaPipe features use monocular camera frames; PyTorch LSTM requires verified 3D motion-capture tensor."
        )

    # Card 3: Ensemble Decision
    if ensemble_dec:
        dec = ensemble_dec.get("decision", "NEEDS_REVIEW")
        bg_col = "#EBFBEE" if dec == "CORRECT_ESTIMATE" else ("#FFF5F5" if dec == "INCORRECT_ESTIMATE" else "#FFF9DB")
        comb_score = ensemble_dec.get("combined_score")
        comb_str = f"{comb_score:.1f} / 100" if comb_score is not None else "Abstained"
        reasons = ensemble_dec.get("reasons", [])
        reason_str = "; ".join(reasons[:2]) if reasons else "Single branch or input requires validation."
        ensemble_html = (
            f"<b>ENSEMBLE DECISION</b><br/>"
            f"<b>Final Decision:</b> {dec}<br/>"
            f"<b>Combined Score:</b> {comb_str}<br/>"
            f"<b>Confidence:</b> {ensemble_dec.get('confidence', 0.0)*100:.0f}%<br/>"
            f"<b>Branch Agreement:</b> {ensemble_dec.get('branch_agreement')}<br/>"
            f"<b>Rationale:</b> {reason_str}"
        )
    else:
        ensemble_html = (
            f"<b>ENSEMBLE DECISION</b><br/>"
            f"<b>Status:</b> NEEDS_REVIEW<br/>"
            f"<b>Reason:</b> Requires both valid statistical and valid deep learning branches before computing ensemble decision."
        )

    cards_data = [
        [
            Paragraph(stat_html, body_style),
            Paragraph(model_html, body_style),
            Paragraph(ensemble_html, body_style),
        ]
    ]
    cards_table = Table(cards_data, colWidths=[175, 175, 180])
    cards_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#E7F5FF')),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#F1F3F5')),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#FFF9DB')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CED4DA')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(cards_table)
    story.append(Spacer(1, 10))

    # 6. Quality Warnings and Physiotherapist Notes
    warnings_list = session_data.get("warnings", [])
    if isinstance(warnings_list, str):
        try:
            warnings_list = json.loads(warnings_list)
        except:
            warnings_list = [warnings_list]

    story.append(Paragraph("Clinical Observations & Data Quality", section_heading))
    warn_text = ""
    if warnings_list and len(warnings_list) > 0:
        warn_text = "<b>Identified Caveats & Movement Deviations:</b><br/>" + "<br/>".join([f"• {w}" for w in warnings_list])
    else:
        warn_text = "• No significant data-quality anomalies detected. Movement was within stable capture parameters."
    
    notes_val = session_data.get("notes")
    if notes_val:
        warn_text += f"<br/><br/><b>User/Clinician Notes:</b> {notes_val}"

    obs_box = Table([[Paragraph(warn_text, body_style)]], colWidths=[530])
    obs_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(obs_box)
    story.append(Spacer(1, 14))

    # 7. Signature Block
    sig_data = [
        [
            Paragraph("<b>Automated Assessment Engine:</b> RehabVision AI v1.0", body_style),
            Paragraph("<b>Reviewing Clinician / Physiotherapist:</b>", body_style)
        ],
        [
            Paragraph("Validated: PyTorch 2.14 / MediaPipe Tasks Vision", body_style),
            Paragraph("Signature: ___________________________________", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[265, 265])
    sig_table.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)

    doc.build(story)

    if output_path is None:
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    return b""
