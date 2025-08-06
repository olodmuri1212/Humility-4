import os
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the analysis functions from agent_manager
from backend.agent_manager import (
    analyze_humility,
    analyze_learning_mindset,
    analyze_feedback_seeking,
    analyze_mistake_handling,
    AnalysisResult
)

@dataclass
class InterviewTurn:
    """Represents a single Q&A turn in the interview."""
    question: str
    answer: str
    analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterviewAnalysis:
    """Container for the complete interview analysis."""
    candidate_name: str
    date: str
    turns: List[InterviewTurn] = field(default_factory=list)
    overall_scores: Dict[str, float] = field(default_factory=dict)
    final_report: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis to a dictionary for JSON serialization."""
        return {
            "candidate_name": self.candidate_name,
            "date": self.date,
            "turns": [{
                "question": turn.question,
                "answer": turn.answer,
                "analysis": turn.analysis
            } for turn in self.turns],
            "overall_scores": self.overall_scores,
            "final_report": self.final_report
        }

class InterviewAnalyzer:
    """Handles the analysis of interview transcripts and report generation."""
    
    def __init__(self, candidate_name: str = ""):
        self.candidate_name = candidate_name
        self.analysis = InterviewAnalysis(
            candidate_name=candidate_name,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    async def analyze_response(self, question: str, answer: str) -> Dict[str, Any]:
        """Analyze a single Q&A pair."""
        if not answer.strip():
            return {"error": "Empty answer provided for analysis"}
        
        try:
            # Run all analysis functions concurrently
            humility_score, humility_evidence = await analyze_humility(answer)
            learning_score, learning_evidence = await analyze_learning_mindset(answer)
            feedback_score, feedback_evidence = await analyze_feedback_seeking(answer)
            mistakes_score, mistakes_evidence = await analyze_mistake_handling(answer)
            
            # Calculate overall score (simple average for now)
            scores = {
                "humility": humility_score,
                "learning": learning_score,
                "feedback": feedback_score,
                "mistakes": mistakes_score
            }
            overall_score = sum(scores.values()) / len(scores)
            
            # Create analysis result
            analysis = {
                "scores": scores,
                "overall_score": overall_score,
                "evidence": {
                    "humility": humility_evidence,
                    "learning": learning_evidence,
                    "feedback": feedback_evidence,
                    "mistakes": mistakes_evidence
                },
                "suggestions": self._generate_suggestions(scores)
            }
            
            # Add to interview turns
            turn = InterviewTurn(
                question=question,
                answer=answer,
                analysis=analysis
            )
            self.analysis.turns.append(turn)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error during analysis: {str(e)}"}
    
    def _generate_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []
        
        if scores["humility"] < 5:
            suggestions.append(
                "Consider showing more humility by acknowledging limitations and being open to learning from others."
            )
        
        if scores["learning"] < 5:
            suggestions.append(
                "Demonstrate more of a learning mindset by discussing how you've grown from experiences."
            )
            
        if scores["feedback"] < 5:
            suggestions.append(
                "Try to be more open to feedback and show how you've incorporated it in the past."
            )
            
        if scores["mistakes"] < 5:
            suggestions.append(
                "When discussing mistakes, focus more on what you learned and how you improved."
            )
            
        return suggestions if suggestions else ["Good overall responses! Keep up the good work."]
    
    def generate_report(self, format: str = "html") -> str:
        """Generate a report of the interview analysis."""
        if not self.analysis.turns:
            return "No interview data available for report generation."
        
        # Calculate overall scores
        total_scores = {"humility": 0, "learning": 0, "feedback": 0, "mistakes": 0}
        
        for turn in self.analysis.turns:
            for key in total_scores:
                total_scores[key] += turn.analysis["scores"][key]
        
        # Calculate averages
        for key in total_scores:
            total_scores[key] = round(total_scores[key] / len(self.analysis.turns), 1)
        
        self.analysis.overall_scores = total_scores
        
        # Generate report based on format
        if format.lower() == "html":
            return self._generate_html_report()
        else:
            return self._generate_text_report()
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score."""
        if score >= 8:
            return "#2f855a"  # Green
        elif score >= 5:
            return "#b7791f"  # Orange
        else:
            return "#c53030"  # Red

    def _get_performance_text(self, score: float) -> str:
        """Get performance text based on score."""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Average"
        else:
            return "Needs Improvement"

    def _generate_html_report(self) -> str:
        """Generate an HTML report of the interview analysis."""
        if not self.analysis.turns:
            return "<h2 style='color: white;'>No interview data available for report generation.</h2>"

        # Calculate overall score
        overall_score = sum(self.analysis.overall_scores.values()) / len(self.analysis.overall_scores) \
            if self.analysis.overall_scores else 0
        
        # Start HTML with dark theme
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Comprehensive Interview Analysis</title>
            <style>
                :root {{
                    --primary: #4a90e2;    /* Brighter blue for primary elements */
                    --secondary: #2ecc71;  /* Green for secondary elements */
                    --success: #2ecc71;    /* Green for success */
                    --warning: #f1c40f;    /* Yellow for warnings */
                    --danger: #e74c3c;     /* Red for danger */
                    --bg-dark: #121212;    /* Dark background */
                    --card-bg: #1e1e1e;    /* Slightly lighter than background */
                    --text: #ffffff;       /* White text */
                    --text-muted: #b0b0b0; /* Muted text */
                    --border: #333333;     /* Border color */
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.8;
                    margin: 0;
                    padding: 20px;
                    color: var(--text);
                    background-color: var(--bg-dark);
                    min-height: 100vh;
                }}
                
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: var(--bg-dark);
                    padding: 30px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid var(--border);
                }}
                
                .header h1 {{
                    color: var(--primary);
                    margin-bottom: 5px;
                }}
                
                .candidate-info {{
                    text-align: center;
                    margin-bottom: 30px;
                    color: var(--text);
                }}
                
                .overall-score {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 25px 20px;
                    background: var(--card-bg);
                    border-radius: 10px;
                    border: 1px solid var(--border);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }}
                
                .scores-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 30px 0;
                }}
                
                .score-card {{
                    background: var(--card-bg);
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    border: 1px solid var(--border);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .score-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                }}
                
                .question-section {{
                    margin: 30px 0;
                    padding: 25px;
                    background: var(--card-bg);
                    border-radius: 8px;
                    border: 1px solid var(--border);
                }}
                
                .question-text {{
                    font-size: 18px;
                    font-weight: 600;
                    color: var(--primary);
                    margin-bottom: 15px;
                }}
                
                .answer-text {{
                    background: rgba(255,255,255,0.05);
                    padding: 15px;
                    border-radius: 6px;
                    margin: 15px 0;
                    border-left: 3px solid var(--primary);
                    color: var(--text);
                }}
                
                .analysis-section {{
                    margin: 20px 0;
                }}
                
                .trait-analysis {{
                    margin: 15px 0;
                    padding: 15px;
                    background: rgba(255,255,255,0.03);
                    border-radius: 6px;
                    border-left: 3px solid var(--primary);
                    transition: all 0.3s ease;
                }}
                
                .trait-analysis:hover {{
                    background: rgba(255,255,255,0.07);
                    transform: translateX(5px);
                }}
                
                .trait-header {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }}
                
                .trait-name {{
                    font-weight: 600;
                    color: var(--primary);
                }}
                
                .trait-score {{
                    font-weight: 700;
                }}
                
                .trait-evidence {{
                    color: var(--text-muted);
                    font-size: 14px;
                    margin-top: 8px;
                    line-height: 1.6;
                }}
                
                .suggestions {{
                    background: rgba(255, 200, 0, 0.1);
                    padding: 20px;
                    border-radius: 8px;
                    margin: 30px 0;
                    border-left: 4px solid var(--warning);
                    color: var(--text);
                }}
                
                .suggestions h3 {{
                    color: var(--warning);
                    margin-top: 0;
                    margin-bottom: 15px;
                }}
                
                .suggestions ul {{
                    padding-left: 20px;
                }}
                
                .suggestions li {{
                    margin-bottom: 10px;
                    position: relative;
                    padding-left: 20px;
                }}
                
                .suggestions li:before {{
                    content: '→';
                    position: absolute;
                    left: 0;
                    color: var(--warning);
                }}
                
                @media (max-width: 768px) {{
                    .scores-grid {{
                        grid-template-columns: 1fr 1fr;
                    }}
                }}
                
                @media (max-width: 480px) {{
                    .scores-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .container {{
                        padding: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Comprehensive Interview Analysis</h1>
                    <p style="color: var(--text-muted);">Detailed assessment of candidate responses</p>
                </div>
                
                <div class="candidate-info">
                    <h2 style="margin: 0 0 5px 0; color: var(--primary);">{self.candidate_name}</h2>
                    <p style="color: var(--text-muted); margin: 0;">Report generated on: {self.analysis.date}</p>
                </div>
                
                <div class="overall-score">
                    <div style="font-size: 16px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">Overall Performance</div>
                    <div class="score" style="color: {self._get_score_color(overall_score)}; font-size: 3.5em; font-weight: 800; margin: 15px 0;">
                        {overall_score:.1f}<span style="font-size: 0.4em; color: var(--text-muted);">/10</span>
                    </div>
                    <div style="font-size: 20px; color: {self._get_score_color(overall_score)}; font-weight: 600;">
                        {self._get_performance_text(overall_score)}
                    </div>
                </div>
                
                <h3 style="color: var(--primary); border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-top: 40px;">
                    Key Assessment Areas
                </h3>
                
                <div class="scores-grid">
        """
        
        # Add score cards for each trait
        for trait, score in self.analysis.overall_scores.items():
            color = self._get_score_color(score)
            html += f"""
                    <div class="score-card">
                        <div class="label" style="color: var(--text-muted);">{trait.capitalize()}</div>
                        <div class="value" style="color: {color}; font-size: 1.8em; margin: 10px 0;">
                            {score:.1f}<span style="font-size: 0.6em; color: var(--text-muted);">/10</span>
                        </div>
                        <div style="font-size: 14px; color: {color};">
                            {self._get_performance_text(score)}
                        </div>
                    </div>
            """
            
        html += """
                </div>
                
                <div class="content">
                    <h3 style="color: var(--primary); border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-top: 40px;">
                        Detailed Response Analysis
                    </h3>
        """
        
        # Add each Q&A with analysis
        for i, turn in enumerate(self.analysis.turns, 1):
            html += f"""
            <div class="question-section">
                <div class="question-text">
                    <span style="color: var(--primary); font-weight: 700;">Q{i}.</span> {turn.question}
                </div>
                <div class="answer-text">
                    <strong>Response:</strong> {turn.answer}
                </div>
                
                <div class="analysis-section">
                    <h4 style="color: var(--primary); margin: 20px 0 10px 0;">Analysis</h4>
            """
            
            # Add analysis for each trait
            for trait, score in turn.analysis["scores"].items():
                evidence = turn.analysis["evidence"][trait]
                color = self._get_score_color(score)
                html += f"""
                <div class="trait-analysis">
                    <div class="trait-header">
                        <span class="trait-name">{trait.capitalize()}</span>
                        <span class="trait-score" style="color: {color};">{score:.1f}/10</span>
                    </div>
                    <div class="trait-evidence">
                        {evidence}
                    </div>
                </div>
                """
                
            # Add suggestions if available
            if i == len(self.analysis.turns) and turn.analysis.get("suggestions"):
                html += """
                <div class="suggestions">
                    <h3>Suggestions for Improvement</h3>
                    <ul>
                """
                for suggestion in turn.analysis["suggestions"]:
                    html += f"<li>{suggestion}</li>"
                html += """
                    </ul>
                </div>
                """
                
            html += """
                </div>
            </div>
            """
        
        # Close HTML
        html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        self.analysis.final_report = html
        return html
    
    def _generate_text_report(self) -> str:
        """Generate a plain text report of the interview analysis."""
        report = [
            "=" * 80,
            f"INTERVIEW ANALYSIS REPORT - {self.analysis.date}",
            "=" * 80,
            f"Candidate: {self.candidate_name}\n"
        ]
        
        # Add overall scores
        report.append("OVERALL SCORES:")
        report.append("-" * 40)
        for trait, score in self.analysis.overall_scores.items():
            report.append(f"{trait.capitalize()}: {score}/10")
        
        # Add detailed analysis for each question
        report.append("\nDETAILED ANALYSIS:")
        report.append("-" * 40)
        
        for i, turn in enumerate(self.analysis.turns, 1):
            report.append(f"\nQUESTION {i}: {turn.question}")
            report.append(f"ANSWER: {turn.answer}")
            
            report.append("\nANALYSIS:")
            for trait, score in turn.analysis["scores"].items():
                evidence = turn.analysis["evidence"][trait]
                report.append(f"- {trait.capitalize()}: {score}/10")
                report.append(f"  Evidence: {evidence}")
        
        # Add suggestions
        if self.analysis.turns:
            suggestions = self.analysis.turns[-1]["analysis"]["suggestions"]
            if suggestions:
                report.append("\nSUGGESTIONS FOR IMPROVEMENT:")
                report.append("-" * 40)
                for suggestion in suggestions:
                    report.append(f"- {suggestion}")
        
        report.append("\n" + "=" * 80)
        self.analysis.final_report = "\n".join(report)
        return self.analysis.final_report
    
    def save_report(self, directory: str = "reports", format: str = "html") -> str:
        """Save the report to a file."""
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename
        safe_name = "".join(c if c.isalnum() else "_" for c in self.candidate_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "html" if format.lower() == "html" else "txt"
        filename = f"interview_report_{safe_name}_{timestamp}.{ext}"
        filepath = os.path.join(directory, filename)
        
        # Generate and save report
        report = self.generate_report(format)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return filepath
