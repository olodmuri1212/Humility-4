# # report_generator.py
# from weasyprint import HTML
# from datetime import datetime
# import os
# from typing import Optional

# REPORTS_DIR = "reports"
# os.makedirs(REPORTS_DIR, exist_ok=True)

# def save_html_report(html: str, candidate_name: str) -> str:
#     """Save HTML string to a timestamped file and return path."""
#     safe_name = "".join(c if c.isalnum() else "_" for c in (candidate_name or "candidate"))
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"interview_report_{safe_name}_{ts}.html"
#     path = os.path.join(REPORTS_DIR, filename)
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(html)
#     return path

# def create_pdf_report_from_html(html: str, candidate_name: str) -> Optional[str]:
#     """
#     Convert HTML string to PDF with WeasyPrint and save it.
#     Returns path or None on failure.
#     """
#     try:
#         safe_name = "".join(c if c.isalnum() else "_" for c in (candidate_name or "candidate"))
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         pdf_filename = f"interview_report_{safe_name}_{ts}.pdf"
#         pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
#         HTML(string=html).write_pdf(pdf_path)
#         return pdf_path
#     except Exception as e:
#         print(f"[report_generator] Failed to create PDF: {e}")
#         return None








# ##trial1
# # report_double_generate.py
# # report_generator.py
# import os
# from typing import List, Dict, Any, Optional
# from datetime import datetime

# # Try WeasyPrint first; gracefully fall back if not installed
# _WEASY_AVAILABLE = True
# try:
#     from weasyprint import HTML, CSS  # type: ignore
# except Exception:
#     _WEASY_AVAILABLE = False

# def _safe_filename(name: str) -> str:
#     return "".join(c if c.isalnum() else "_" for c in name)

# def build_full_report_html(
#     analysis_payload: Dict[str, Any],
#     llm_agents: List[Dict[str, Any]] | None = None
# ) -> str:
#     """
#     analysis_payload is InterviewAnalyzer.to_summary_payload()
#     llm_agents: [{'agent_name': str, 'score': float, 'evidence': str}, ...]
#     """
#     candidate = analysis_payload["candidate_name"]
#     gen_at = analysis_payload["generated_at"]
#     overall = analysis_payload["overall_scores"]
#     turns = analysis_payload["turns"]
#     suggestions = analysis_payload["summary_suggestions"]

#     # basic styling inlined for PDF safety
#     style = """
#     <style>
#       :root{
#         --bg:#0f1115; --card:#171a21; --ink:#e6e6e6; --muted:#9aa0a6;
#         --primary:#4a90e2; --ok:#2ecc71; --warn:#f1c40f; --bad:#e74c3c; --border:#2a2f3a
#       }
#       body{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;}
#       .wrap{max-width:980px;margin:0 auto;}
#       .hdr{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:20px}
#       h1{margin:0 0 6px 0;color:var(--primary)}
#       .meta{color:var(--muted);font-size:14px}
#       .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:12px}
#       .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
#       .score{font-size:28px;font-weight:800;margin-top:8px}
#       .ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
#       .sec h2{color:var(--primary);margin:24px 0 8px 0}
#       .q{color:var(--primary);font-weight:600}
#       .ans{background:#1c212b;border-left:4px solid var(--primary);padding:12px;border-radius:8px;margin:8px 0}
#       .trait{background:#181c25;border-left:3px solid var(--primary);padding:10px;border-radius:6px;margin:8px 0;color:#cfd3da}
#       table{width:100%;border-collapse:collapse;border:1px solid var(--border)}
#       th,td{border-bottom:1px solid var(--border);padding:8px;text-align:left}
#       th{color:var(--muted);font-weight:600}
#       .tips{background:rgba(241,196,15,0.08);border-left:4px solid var(--warn)}
#       .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
#       .chip{background:#212634;border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:12px;color:var(--muted)}
#       .llm-evi{color:#aab0b7;font-size:12px}
#     </style>
#     """

#     def score_class(v: float) -> str:
#         if v >= 8: return "ok"
#         if v >= 5: return "warn"
#         return "bad"

#     # overall
#     overall_html = "".join(
#         f"""<div class="card">
#               <div>{k.capitalize()}</div>
#               <div class="score {score_class(v)}">{v:.1f}/10</div>
#             </div>"""
#         for k, v in overall.items()
#     )

#     # per‑turn details
#     turns_html = ""
#     for t in turns:
#         traits_html = "".join(
#             f"""<div class="trait">
#                     <b>{trait.capitalize()}</b>
#                     <span style="float:right">{t['scores'][trait]:.1f}/10</span>
#                     <div class="llm-evi">{t['evidence'][trait]}</div>
#                 </div>"""
#             for trait in ["humility","learning","feedback","mistakes"]
#         )
#         sug_html = ""
#         if t.get("suggestions"):
#             sug_html = "<div class='tips card'><b>Suggestions:</b><div class='chips'>" + \
#                        "".join(f"<div class='chip'>{s}</div>" for s in t["suggestions"]) + \
#                        "</div></div>"

#         turns_html += f"""
#         <div class="card">
#            <div class="q">Q{t['index']}. {t['question']}</div>
#            <div class="ans"><b>Response:</b> {t['answer']}</div>
#            {traits_html}
#            {sug_html}
#         </div>"""

#     # LLM Agent scores table (if provided)
#     llm_html = ""
#     if llm_agents:
#         rows = "".join(
#             f"<tr><td>{a.get('agent_name','')}</td><td>{float(a.get('score',0)):.2f}</td>"
#             f"<td><div class='llm-evi'>{a.get('evidence','') or '—'}</div></td></tr>"
#             for a in llm_agents
#         )
#         llm_html = f"""
#         <div class="sec">
#           <h2>LLM Agents — Detailed Scores</h2>
#           <div class="card">
#             <table>
#               <thead><tr><th>Agent</th><th>Score</th><th>Evidence</th></tr></thead>
#               <tbody>{rows}</tbody>
#             </table>
#           </div>
#         </div>
#         """

#     # summary suggestions
#     sug_section = ""
#     if suggestions:
#         sug_section = f"""
#         <div class="sec">
#           <h2>Summary Suggestions</h2>
#           <div class="tips card">
#             {"".join(f"<div>• {s}</div>" for s in suggestions)}
#           </div>
#         </div>
#         """

#     html = f"""
#     <!doctype html>
#     <html><head><meta charset="utf-8"><title>Interview Report</title>{style}</head>
#     <body><div class="wrap">
#         <div class="hdr">
#           <h1>Comprehensive Interview Analysis</h1>
#           <div class="meta">{candidate} &nbsp;•&nbsp; Generated: {gen_at}</div>
#         </div>

#         <div class="sec">
#           <h2>Core Behavioral Traits — Averages</h2>
#           <div class="grid">{overall_html}</div>
#         </div>

#         {llm_html}

#         <div class="sec">
#           <h2>Detailed Response Analysis</h2>
#           <div class="grid" style="grid-template-columns:1fr">{turns_html}</div>
#         </div>

#         {sug_section}
#     </div></body></html>
#     """
#     return html

# def save_html_report(html: str, candidate_name: str, out_dir: str = "reports") -> str:
#     os.makedirs(out_dir, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.html")
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(html)
#     return path

# def create_pdf_report_from_html(html: str, candidate_name: str, out_dir: str = "reports") -> Optional[str]:
#     os.makedirs(out_dir, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     pdf_path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.pdf")
#     if _WEASY_AVAILABLE:
#         HTML(string=html).write_pdf(pdf_path)
#         return pdf_path
#     else:
#         # Fallback: if WeasyPrint is unavailable, return None so caller can inform user.
#         return None



















# # report_generator.py# take 2 
# import os
# from typing import List, Dict, Any, Optional
# from datetime import datetime

# _WEASY_AVAILABLE = True
# try:
#     from weasyprint import HTML  # type: ignore
# except Exception:
#     _WEASY_AVAILABLE = False

# def _safe_filename(name: str) -> str:
#     return "".join(c if c.isalnum() else "_" for c in name)

# def _clean_agent_name(name: str) -> str:
#     if not name:
#         return ""
#     if name.endswith("Agent"):
#         name = name[:-5]
#     return name

# def _name_matches_any(name_l: str, needles: List[str]) -> bool:
#     return any(k in name_l for k in needles)

# def _humility_blend(
#     core_humility_avg: float,
#     llm_agents: List[Dict[str, Any]],
#     weight_core: float = 0.6,
#     weight_llm: float = 0.4
# ) -> float:
#     """
#     Build a humility score using:
#       - core parser humility average
#       - LLM humility signals (excluding PraiseHandling, Pronoun*, IDontKnow)
#     For agents that indicate *anti‑humility* (BragFlag, KnowItAll, BlameShift),
#     we invert via (10 - score).
#     """
#     if not llm_agents:
#         return round(core_humility_avg, 1)

#     exclude_needles = ["praisehandling", "pronoun", "idontknow"]
#     invert_needles   = ["bragflag", "knowitall", "blameshift"]
#     include_needles  = ["admitmistake", "mindchange", "sharecredit", "supportgrowth", "feedbackacceptance"] + invert_needles

#     vals = []
#     for a in llm_agents:
#         raw_name = str(a.get("agent_name", ""))
#         name_l = raw_name.lower()
#         if _name_matches_any(name_l, exclude_needles):
#             continue
#         if not _name_matches_any(name_l, include_needles):
#             # skip agents unrelated to humility
#             continue
#         try:
#             s = float(a.get("score", 0.0))
#         except Exception:
#             s = 0.0
#         if _name_matches_any(name_l, invert_needles):
#             s = 10.0 - s  # higher = more humble signal
#         vals.append(s)

#     if not vals:
#         return round(core_humility_avg, 1)

#     llm_avg = sum(vals) / len(vals)
#     final = weight_core * core_humility_avg + weight_llm * llm_avg
#     return round(final, 1)

# def _score_class(v: float) -> str:
#     if v >= 8: return "ok"
#     if v >= 5: return "warn"
#     return "bad"

# def build_full_report_html(
#     analysis_payload: Dict[str, Any],
#     llm_agents: List[Dict[str, Any]] | None = None
# ) -> str:
#     """
#     PDF layout per latest request:
#       - Core Behavioral Traits: show ONLY Humility (final humility blended score)
#       - Detailed Response Analysis: per question show
#           * Humility score
#           * Learning score
#           * Feedback (text from feedback_agent)
#       - LLM Agents table still shown with cleaned names
#     """
#     candidate = analysis_payload["candidate_name"]
#     gen_at = analysis_payload["generated_at"]
#     overall = analysis_payload["overall_scores"]     # contains humility (+ optional learning not displayed in Core)
#     turns = analysis_payload["turns"]

#     llm_agents = llm_agents or []
#     # Clean names for display in the LLM table
#     llm_agents_display = [
#         {
#             "agent_name": _clean_agent_name(a.get("agent_name", "")),
#             "score": float(a.get("score", 0.0)),
#             "evidence": a.get("evidence", "") or ""
#         } for a in llm_agents
#     ]

#     # Compute *final humility* for Core section (blend parser + LLM humility signals, excluding 3 groups)
#     core_hum = float(overall.get("humility", 0.0))
#     humility_final = _humility_blend(core_hum, llm_agents)

#     style = """
#     <style>
#       :root{
#         --bg:#0f1115; --card:#171a21; --ink:#e6e6e6; --muted:#9aa0a6;
#         --primary:#4a90e2; --ok:#2ecc71; --warn:#f1c40f; --bad:#e74c3c; --border:#2a2f3a
#       }
#       body{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;}
#       .wrap{max-width:980px;margin:0 auto;}
#       .hdr{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:20px}
#       h1{margin:0 0 6px 0;color:var(--primary)}
#       .meta{color:var(--muted);font-size:14px}
#       .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
#       .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
#       .score{font-size:28px;font-weight:800;margin-top:8px}
#       .ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
#       .sec h2{color:var(--primary);margin:24px 0 8px 0}
#       .q{color:var(--primary);font-weight:600}
#       .ans{background:#1c212b;border-left:4px solid var(--primary);padding:12px;border-radius:8px;margin:8px 0}
#       .trait{background:#181c25;border-left:3px solid var(--primary);padding:10px;border-radius:6px;margin:8px 0;color:#cfd3da}
#       table{width:100%;border-collapse:collapse;border:1px solid var(--border)}
#       th,td{border-bottom:1px solid var(--border);padding:8px;text-align:left}
#       th{color:var(--muted);font-weight:600}
#       .fbtext{background:#1b202a;border-left:3px solid #58a6ff;padding:10px;border-radius:6px;margin-top:8px;color:#cfd3da}
#       .note{color:var(--muted);font-size:12px;margin-top:6px}
#     </style>
#     """

#     # CORE — show only Humility (final blended)
#     core_html = f"""
#     <div class="card">
#       <div>Humility</div>
#       <div class="score {_score_class(humility_final)}">{humility_final:.1f}/10</div>
#       <div class="note">Humility blends parser score with relevant LLM signals, excluding Pronoun*, IDontKnow, and PraiseHandling agents.</div>
#     </div>"""

#     # LLM Agents table (names cleaned)
#     llm_rows = "".join(
#         f"<tr><td>{a['agent_name']}</td><td>{a['score']:.2f}</td>"
#         f"<td><div style='color:#aab0b7;font-size:12px'>{(a['evidence'] or '').strip()}</div></td></tr>"
#         for a in llm_agents_display
#     )
#     llm_html = f"""
#     <div class="sec">
#       <h2>LLM Agents — Detailed Scores</h2>
#       <div class="card">
#         <table>
#           <thead><tr><th>Agent</th><th>Score</th><th>Evidence</th></tr></thead>
#           <tbody>{llm_rows}</tbody>
#         </table>
#       </div>
#     </div>"""

#     # Detailed Response Analysis — per question: Humility score + Learning score + Feedback text
#     turns_html = ""
#     for t in turns:
#         hum = float(t["scores"]["humility"])
#         lrn = float(t["scores"]["learning"])
#         hum_ev = t["evidence"]["humility"]
#         lrn_ev = t["evidence"]["learning"]
#         fb_text = t.get("feedback_text", "")
#         turns_html += f"""
#         <div class="card">
#           <div class="q">Q{t['index']}. {t['question']}</div>
#           <div class="ans"><b>Response:</b> {t['answer']}</div>
#           <div class="trait"><b>Humility</b><span style="float:right">{hum:.1f}/10</span>
#               <div style="color:#aab0b7;font-size:12px;margin-top:4px">{hum_ev}</div>
#           </div>
#           <div class="trait"><b>Learning</b><span style="float:right">{lrn:.1f}/10</span>
#               <div style="color:#aab0b7;font-size:12px;margin-top:4px">{lrn_ev}</div>
#           </div>
#           <div class="fbtext"><b>Feedback</b><br>{fb_text}</div>
#         </div>"""

#     html = f"""
#     <!doctype html>
#     <html><head><meta charset="utf-8"><title>Interview Report</title>{style}</head>
#     <body><div class="wrap">
#         <div class="hdr">
#           <h1>Comprehensive Interview Analysis</h1>
#           <div class="meta">{candidate} &nbsp;•&nbsp; Generated: {gen_at}</div>
#         </div>

#         <div class="sec">
#           <h2>Core Behavioral Traits</h2>
#           <div class="grid">{core_html}</div>
#         </div>

#         {llm_html}

#         <div class="sec">
#           <h2>Detailed Response Analysis</h2>
#           <div class="grid" style="grid-template-columns:1fr">{turns_html}</div>
#         </div>
#     </div></body></html>
#     """
#     return html

# def save_html_report(html: str, candidate_name: str, out_dir: str = "reports") -> str:
#     os.makedirs(out_dir, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.html")
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(html)
#     return path

# def create_pdf_report_from_html(html: str, candidate_name: str, out_dir: str = "reports") -> Optional[str]:
#     os.makedirs(out_dir, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     pdf_path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.pdf")
#     if _WEASY_AVAILABLE:
#         HTML(string=html).write_pdf(pdf_path)
#         return pdf_path
#     return None























###take 3 
# report_generator.py
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

_WEASY_AVAILABLE = True
try:
    from weasyprint import HTML  # type: ignore
except Exception:
    _WEASY_AVAILABLE = False


# -------------------------- helpers --------------------------

def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)

def _clean_agent_name(name: str) -> str:
    if not name:
        return ""
    return name[:-5] if name.endswith("Agent") else name

def _score_class(v: float) -> str:
    if v >= 8: return "ok"
    if v >= 5: return "warn"
    return "bad"

def _name_has(name_lower: str, needles: List[str]) -> bool:
    return any(n in name_lower for n in needles)


# ------------------- humility from LLM agents only -------------------

def _humility_from_llm_only(
    llm_agents: List[Dict[str, Any]],
    fallback_parser_humility: float = 0.0
) -> float:
    """
    Compute final Humility from LLM agents ONLY, excluding:
      - IDontKnow
      - Pronoun*
      - ShareCredit
      - PraiseHandling (typo-safe: also excludes 'precisehandling')

    Anti-humility agents are inverted (10 - score):
      - BragFlag, KnowItAll, BlameShift

    If nothing remains after exclusion, fall back to parser humility average.
    """
    if not llm_agents:
        return round(float(fallback_parser_humility), 1)

    # Exclusions (case-insensitive, substring match)
    exclude_needles = ["idontknow", "pronoun", "sharecredit", "praisehandling", "precisehandling"]

    # Agents where a higher raw score means lower humility signal (invert)
    invert_needles = ["bragflag", "knowitall", "blameshift"]

    included_vals: List[float] = []
    for a in llm_agents:
        raw_name = str(a.get("agent_name", ""))
        nl = raw_name.lower()

        # skip explicit exclusions
        if _name_has(nl, exclude_needles):
            continue

        # take score safely
        try:
            s = float(a.get("score", 0.0))
        except Exception:
            s = 0.0

        # invert negative-signal agents
        if _name_has(nl, invert_needles):
            s = 10.0 - s

        included_vals.append(s)

    if not included_vals:
        return round(float(fallback_parser_humility), 1)

    final = sum(included_vals) / len(included_vals)
    return round(max(0.0, min(10.0, final)), 1)


# -------------------------- html builder --------------------------

def build_full_report_html(
    analysis_payload: Dict[str, Any],
    llm_agents: List[Dict[str, Any]] | None = None
) -> str:
    """
    PDF layout per latest request:

      - Core Behavioral Traits: show ONLY Humility, computed from LLM agents only,
        excluding IDontKnow, Pronoun*, ShareCredit, and PraiseHandling/PreciseHandling.
        (BragFlag/KnowItAll/BlameShift are inverted as 10 - score.)
      - Detailed Response Analysis (per question): show Humility score, Learning score,
        and Feedback (text). No Mistakes section.
      - LLM Agents table: hide IDontKnow and PronounRatio rows; show others with names
        cleaned (drop 'Agent' suffix).
    """
    candidate = analysis_payload.get("candidate_name", "Candidate")
    gen_at = analysis_payload.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    overall = analysis_payload.get("overall_scores", {}) or {}
    turns = analysis_payload.get("turns", []) or []

    llm_agents = llm_agents or []

    # 1) Humility (Core) — from LLM agents only (with exclusions/inversions)
    parser_humility_avg = float(overall.get("humility", 0.0))
    humility_final = _humility_from_llm_only(llm_agents, fallback_parser_humility=parser_humility_avg)

    # 2) LLM agents table — hide IDontKnow & PronounRatio in display
    hide_in_table_needles = ["idontknow", "pronoun"]
    llm_agents_display = []
    for a in llm_agents:
        raw_name = str(a.get("agent_name", ""))
        nl = raw_name.lower()
        if _name_has(nl, hide_in_table_needles):
            continue
        llm_agents_display.append({
            "agent_name": _clean_agent_name(raw_name),
            "score": float(a.get("score", 0.0)),
            "evidence": a.get("evidence", "") or ""
        })

    # -------------------------- styles --------------------------
    style = """
    <style>
      :root{
        --bg:#0f1115; --card:#171a21; --ink:#e6e6e6; --muted:#9aa0a6;
        --primary:#4a90e2; --ok:#2ecc71; --warn:#f1c40f; --bad:#e74c3c; --border:#2a2f3a
      }
      body{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;}
      .wrap{max-width:980px;margin:0 auto;}
      .hdr{border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:20px}
      h1{margin:0 0 6px 0;color:var(--primary)}
      .meta{color:var(--muted);font-size:14px}
      .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
      .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
      .score{font-size:28px;font-weight:800;margin-top:8px}
      .ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
      .sec h2{color:var(--primary);margin:24px 0 8px 0}
      .q{color:var(--primary);font-weight:600}
      .ans{background:#1c212b;border-left:4px solid var(--primary);padding:12px;border-radius:8px;margin:8px 0}
      .trait{background:#181c25;border-left:3px solid var(--primary);padding:10px;border-radius:6px;margin:8px 0;color:#cfd3da}
      table{width:100%;border-collapse:collapse;border:1px solid var(--border)}
      th,td{border-bottom:1px solid var(--border);padding:8px;text-align:left}
      th{color:var(--muted);font-weight:600}
      .fbtext{background:#1b202a;border-left:3px solid #58a6ff;padding:10px;border-radius:6px;margin-top:8px;color:#cfd3da}
      .note{color:var(--muted);font-size:12px;margin-top:6px}
    </style>
    """

    # ---------------------- core section (Humility only) ----------------------
    core_html = f"""
    <div class="card">
      <div>Humility</div>
      <div class="score {_score_class(humility_final)}">{humility_final:.1f}/10</div>
      <div class="note">
        Computed from agent signals only; excludes IDontKnow, Pronoun*, ShareCredit, and PraiseHandling.
        Anti‑humility agents (BragFlag/KnowItAll/BlameShift) are inverted.
      </div>
    </div>"""

    # ---------------------- LLM Agents table (filtered) ----------------------
    llm_rows = "".join(
        f"<tr><td>{a['agent_name']}</td><td>{a['score']:.2f}</td>"
        f"<td><div style='color:#aab0b7;font-size:12px'>{(a['evidence'] or '').strip()}</div></td></tr>"
        for a in llm_agents_display
    )
    llm_html = f"""
    <div class="sec">
      <h2>LLM Agents — Detailed Scores</h2>
      <div class="card">
        <table>
          <thead><tr><th>Agent</th><th>Score</th><th>Evidence</th></tr></thead>
          <tbody>{llm_rows}</tbody>
        </table>
      </div>
    </div>"""

    # ---------------------- per-question section ----------------------
    # Show: Humility score + Learning score + Feedback text (no Mistakes)
    turns_html = ""
    for t in turns:
        hum = float(t["scores"]["humility"])
        lrn = float(t["scores"]["learning"])
        hum_ev = t["evidence"]["humility"]
        lrn_ev = t["evidence"]["learning"]
        fb_text = t.get("feedback_text", "")
        turns_html += f"""
        <div class="card">
          <div class="q">Q{t['index']}. {t['question']}</div>
          <div class="ans"><b>Response:</b> {t['answer']}</div>
          <div class="trait"><b>Humility</b><span style="float:right">{hum:.1f}/10</span>
              <div style="color:#aab0b7;font-size:12px;margin-top:4px">{hum_ev}</div>
          </div>
          <div class="trait"><b>Learning</b><span style="float:right">{lrn:.1f}/10</span>
              <div style="color:#aab0b7;font-size:12px;margin-top:4px">{lrn_ev}</div>
          </div>
          <div class="fbtext"><b>Feedback</b><br>{fb_text}</div>
        </div>"""

    # ---------------------- assemble ----------------------
    html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Interview Report</title>{style}</head>
    <body><div class="wrap">
        <div class="hdr">
          <h1>Comprehensive Interview Analysis</h1>
          <div class="meta">{candidate} &nbsp;•&nbsp; Generated: {gen_at}</div>
        </div>

        <div class="sec">
          <h2>Core Behavioral Traits</h2>
          <div class="grid">{core_html}</div>
        </div>

        {llm_html}

        <div class="sec">
          <h2>Detailed Response Analysis</h2>
          <div class="grid" style="grid-template-columns:1fr">{turns_html}</div>
        </div>
    </div></body></html>
    """
    return html


# -------------------------- save / export --------------------------

def save_html_report(html: str, candidate_name: str, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

def create_pdf_report_from_html(html: str, candidate_name: str, out_dir: str = "reports") -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"interview_report_{_safe_filename(candidate_name)}_{ts}.pdf")
    if _WEASY_AVAILABLE:
        HTML(string=html).write_pdf(pdf_path)
        return pdf_path
    return None
