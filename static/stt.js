(() => {
    /* 0. Setup ------------------------------------------------------ */
    const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/stt_stream`;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const txTextarea = document.getElementById("transcript");
    const status = document.getElementById("statusText");
    const recBtn  = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
  
    /* 1. Decide path ------------------------------------------------ */
    if (SpeechRecognition) {
      // --- Browser native path (Chrome, Edge, Safari) ---------------
      const recog = new SpeechRecognition();
      recog.continuous = true; recog.interimResults = true;
      recog.onresult = e => {
        let interim = "", final = "";
        for (let i = e.resultIndex; i < e.results.length; i++)
          (e.results[i].isFinal ? final : interim) += e.results[i][0].transcript;
        window.updateTranscript(final, interim);       // function is already in HTML
      };
      recBtn.onclick = () => { recog.lang = document.getElementById("languageSelect").value; recog.start(); ui(true); };
      stopBtn.onclick = () => { recog.stop(); ui(false); };
  
    } else {
      // --- Web‑socket path (Firefox + mobiles) ----------------------
      let ws, media, context, processor;
      recBtn.onclick = async () => {
        ws = new WebSocket(WS_URL);
        ws.onmessage = m => {
          const {partial, final} = JSON.parse(m.data);
          window.updateTranscript(final || "", final ? "" : partial);
        };
        await wsOpen(ws);
  
        context = new AudioContext({sampleRate: 16000});
        const stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true}});
        media = context.createMediaStreamSource(stream);
        processor = context.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = e => {
          const pcm = e.inputBuffer.getChannelData(0);          // float32 −1..1
          const int16 = floatTo16BitPCM(pcm);
          ws.send(int16);
        };
        media.connect(processor); processor.connect(context.destination);
        ui(true);
      };
      stopBtn.onclick = () => { processor?.disconnect(); media?.disconnect();
                                 context?.close(); ws?.close(); ui(false); };
    }
  
    /* helpers */
    function ui(rec) { recBtn.disabled = rec; stopBtn.disabled = !rec; status.textContent = rec?"Listening…":"Ready"; }
    const wsOpen = ws => new Promise(r => ws.onopen = r);
    const floatTo16BitPCM = f32 => {
        const buf = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; ++i)
            buf[i] = Math.max(-1, Math.min(1, f32[i])) * 0x7FFF;
        return buf.buffer;
    };
  })();
  