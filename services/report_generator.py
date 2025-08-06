# services/report_generator.py

from weasyprint import HTML
from datetime import datetime

from core.state import InterviewState
from core.scoring import WEIGHTS # Re-enable the correct import

def generate_html_report(state: InterviewState) -> str:
    """
    Generates a comprehensive HTML report from the final interview state.
    This HTML can be displayed directly or converted to a PDF.
    """
    
    # --- Report Header ---
    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Humility Interview Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
            body {{ 
                font-family: 'Inter', sans-serif; 
                margin: 2em; 
                color: #333;
                line-height: 1.6;
            }}
            h1, h2 {{ 
                color: #1a237e; /* Dark blue */
                border-bottom: 2px solid #c5cae9; /* Light blue */
                padding-bottom: 5px;
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 2em; 
                font-size: 0.9em;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #e8eaf6; /* Lighter blue */
                font-weight: 700;
            }}
            .score-positive {{ color: #2e7d32; font-weight: bold; }}
            .score-negative {{ color: #c62828; font-weight: bold; }}
            .evidence {{ font-style: italic; color: #555; }}
            .summary {{
                background-color: #f9f9f9;
                border-left: 5px solid #1a237e;
                padding: 15px;
                margin-bottom: 2em;
            }}
            .transcript-q {{ font-weight: bold; color: #303f9f; }}
            .transcript-a {{ margin-left: 20px; }}
        </style>
    </head>
    <body>
        <h1>Humility Interview Report</h1>
        <div class="summary">
            <p><strong>Candidate:</strong> {state.candidate_name or 'N/A'}</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <h2>Final Humility Score: <span class="score-positive">{state.normalized_humility_score}/100</span></h2>
        </div>
        
        <h2>Detailed Trait Analysis</h2>
        <table>
            <tr>
                <th>Humility Trait</th>
                <th>Score (Raw)</th>
                <th>Evidence / Justification</th>
            </tr>
    """
    
    # --- Table Rows for each agent's analysis ---
    for agent_name, raw_score in sorted(state.cumulative_scores.items()):
        weight = WEIGHTS.get(agent_name, 0)
        
        # Find the evidence from the conversation history
        evidence_text = "No specific evidence captured."
        for turn in state.conversation_history:
            for analysis in turn.analysis_results:
                if analysis.agent_name == agent_name and analysis.evidence:
                    evidence_text = analysis.evidence
                    break
            if evidence_text != "No specific evidence captured.":
                break
        
        score_class = "score-positive" if raw_score >= 0 else "score-negative"
        trait_name = agent_name.replace('Agent', '')
        
        html += f"""
        <tr>
            <td>{trait_name}</td>
            <td class="{score_class}">{raw_score}</td>
            <td class="evidence">"{evidence_text}"</td>
        </tr>
        """
        
    html += """
        </table>
        <h2>Full Interview Transcript</h2>
    """
    
    # --- Full Transcript Section ---
    if not state.conversation_history:
        html += "<p>No conversation was recorded.</p>"
    else:
        for i, turn in enumerate(state.conversation_history):
            html += f"<div>"
            html += f"<p class='transcript-q'><strong>Question {i+1}:</strong> {turn.question}</p>"
            html += f"<p class='transcript-a'><strong>Answer:</strong> {turn.transcript}</p>"
            html += "</div><hr>"

    html += "</body></html>"
    return html


def create_pdf_report(state: InterviewState) -> bytes:
    """
    Creates a PDF report from the final interview state by converting
    a generated HTML string.

    Args:
        state: The final InterviewState object.

    Returns:
        The generated PDF content as bytes.
    """
    print("INFO: Generating PDF report...")
    try:
        html_content = generate_html_report(state)
        # The core conversion step using WeasyPrint
        pdf_bytes = HTML(string=html_content).write_pdf()
        print("INFO: PDF report generated successfully.")
        return pdf_bytes
    except Exception as e:
        print(f"ERROR: Failed to create PDF report with WeasyPrint: {e}")
        # Return an empty bytes object or handle the error as needed
        return b""
