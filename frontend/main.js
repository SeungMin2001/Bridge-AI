const btn = document.getElementById("start");

const wf = document.getElementById('wf');
const statusLbl = document.getElementById('statusLbl');
const liveItem = document.getElementById('liveItem');
const liveText = document.getElementById('liveText');
const tsList = document.getElementById('tsList');

let isRecording = false;
let ws = null;
let audioContext = null;
let stream = null;

// UI 파형 그리기 및 상태 변경 기능
function toggleWaveformUI(show) {
  if (show) {
    statusLbl.style.display = 'none';
    for (let i = 0; i < 20; i++) {
      const b = document.createElement('div');
      b.className = 'wb';
      b.style.setProperty('--h', (Math.random() * 16 + 4).toFixed(1) + 'px');
      b.style.setProperty('--d', (Math.random() * 0.5 + 0.4).toFixed(2) + 's');
      b.style.animationDelay = (Math.random() * 0.4) + 's';
      wf.appendChild(b);
    }
  } else {
    statusLbl.style.display = 'block';
    const wbs = wf.querySelectorAll('.wb');
    wbs.forEach(wb => wb.remove());
  }
}

btn.onclick = async () => {
  if (isRecording) {
    // 녹음 중지 시
    isRecording = false;
    btn.classList.remove('active'); 
    liveItem.classList.remove('recording-active'); 
    toggleWaveformUI(false);

    if (ws) ws.close();
    if (audioContext) audioContext.close();
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    return;
  }

  // 녹음 시작 시
  isRecording = true;
  btn.classList.add('active'); 
  liveItem.style.display = 'flex'; 
  liveItem.classList.add('recording-active'); 
  toggleWaveformUI(true);

  ws = new WebSocket("ws://100.104.164.84:8000/ws");

  ws.onmessage = (event) => {
    console.log(event.data);
    
    // 백엔드에서 받은 텍스트를 UI에 표시 (누적)
    if (liveText.textContent && event.data.trim() !== '') {
       liveText.textContent += ' ' + event.data;
    } else {
       liveText.textContent = event.data;
    }
    
    // 자동 스크롤
    tsList.scrollTop = tsList.scrollHeight;
  };

  ws.onopen = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext();
      await audioContext.audioWorklet.addModule("pcm-worklet.js");

      const source = audioContext.createMediaStreamSource(stream);
      const processor = new AudioWorkletNode(audioContext, "pcm-worklet");

      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.port.onmessage = (event) => {
        const data = float32ToInt16(event.data);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data.buffer);
        }
      };
    } catch (err) {
      console.error("마이크 접근 또는 오디오 처리 중 오류:", err);
      // 에러 발생 시 UI 초기화
      isRecording = false;
      btn.classList.remove('active'); 
      liveItem.classList.remove('recording-active'); 
      toggleWaveformUI(false);
    }
  };

  ws.onclose = () => {
    // 연결 종료 시 UI 초기화
    if (isRecording) {
      isRecording = false;
      btn.classList.remove('active'); 
      liveItem.classList.remove('recording-active'); 
      toggleWaveformUI(false);
    }
  };
};

function float32ToInt16(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return int16Array;
}
