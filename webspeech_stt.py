import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import datetime
import base64

BACKEND = "http://127.0.0.1:8000"  # your FastAPI base

st.set_page_config(page_title="Live STT – Web Speech API", layout="wide")
st.title("🎤 Live Speech‑to‑Text (Browser Web Speech API)")

# ─────────────────────────────
# 1.  HTML + JS component
# ─────────────────────────────
lang = st.selectbox(
    "Choose recognition language",
    ["en-US", "hi-IN", "fr-FR", "de-DE", "es-ES"]
)

component_value = components.html(
    f"""
<style>
#btns button {{
  margin-right: 8px; padding: 6px 12px; font-size: 14px;
}}
#interim {{ color: #999; }}
</style>

<div id="btns">
  <button onclick="startRec()">Start 🎙️</button>
  <button onclick="stopRec()">Stop ⏹️</button>
  <button onclick="clearRec()">Clear 🗑️</button>
</div>
<p id="status">Status: stopped</p>
<h4>Transcript</h4>
<div id="final"></div>
<div id="interim"></div>

<script>
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) {{
  document.getElementById('status').innerText = "Web Speech API not supported in this browser.";
}} else {{
  const rec = new SpeechRecognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "{lang}";
  let finalText = "";
  rec.onstart = () => Streamlit.setComponentValue({{status:"recording"}});
  rec.onend   = () => Streamlit.setComponentValue({{status:"stopped"}});
  rec.onerror = (e)=>console.error(e);

  rec.onresult = (e) => {{
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; ++i) {{
      const res = e.results[i];
      if (res.isFinal) {{
         const conf = (res[0].confidence*100).toFixed(1);
         finalText += res[0].transcript + ` <span style='color:#0a0'>(`+conf+`%)</span> `;
      }} else {{
         interim += res[0].transcript + " ";
      }}
    }}
    document.getElementById("final").innerHTML = finalText;
    document.getElementById("interim").innerText = interim;
    Streamlit.setComponentValue({{status:"recording", final:finalText, interim}});
  }};

  window.startRec = () => rec.start();
  window.stopRec  = () => rec.stop();
  window.clearRec = () => {{
      finalText = "";
      document.getElementById("final").innerHTML = "";
      document.getElementById("interim").innerText = "";
      Streamlit.setComponentValue({{status:"cleared", final:""}});
  }};
}}
</script>
""",
    height=310
)

# ─────────────────────────────
# 2.  Show live status & text
# ─────────────────────────────
if component_value:
    status = component_value.get("status", "stopped")
    st.write(f"**Status:** {status}")
    transcript = component_value.get("final", "")
else:
    transcript = ""

# ─────────────────────────────
# 3.  Download button
# ─────────────────────────────
if transcript:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    b64 = base64.b64encode(transcript.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="transcript_{ts}.txt">💾 Download .txt</a>'
    st.markdown(href, unsafe_allow_html=True)

# ─────────────────────────────
# 4.  Analyse via your backend
# ─────────────────────────────
st.markdown("---")
st.header("Send to backend analysis")

if st.button("📤 Analyse transcript") and transcript.strip():
    with st.spinner("Contacting backend…"):
        payload = {"transcript": transcript, "question": "N/A"}
        try:
            r = requests.post(f"{BACKEND}/analyze", json=payload, timeout=60)
            r.raise_for_status()
            scores = r.json()["scores"]
            st.success("Analysis complete!")
            for agent, detail in scores.items():
                st.metric(agent, f"{detail['score']}")
                with st.expander("Evidence"):
                    st.write(detail["evidence"])
        except Exception as e:
            st.error(f"Backend error: {e}")
else:
    st.write("_Nothing to analyse yet._")
