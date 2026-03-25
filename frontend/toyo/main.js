const btn = document.getElementById("start");
const recordingTimer = document.getElementById("recording-timer");
const recordingTimeText = document.getElementById("recording-time");

let ws = null;
let audioContext = null;
let stream = null;
let processor = null;
let source = null;
let timerInterval = null;
let recordingSeconds = 0;

function updateTimer() {
  recordingSeconds++;
  const minutes = Math.floor(recordingSeconds / 60);
  const seconds = recordingSeconds % 60;
  recordingTimeText.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function stopRecording() {
  if (recordingTimer) {
    recordingTimer.classList.add('hidden');
    recordingTimer.classList.remove('flex');
    btn.style.display = 'block';
  }

  clearInterval(timerInterval);

  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

if (recordingTimer) {
  recordingTimer.onclick = () => {
    stopRecording();
  };
}

btn.onclick = async () => {
  btn.style.display = 'none';
  if (recordingTimer) {
    recordingTimer.classList.remove('hidden');
    recordingTimer.classList.add('flex');
  }
  
  recordingSeconds = 0;
  if (recordingTimeText) recordingTimeText.textContent = "0:00";
  timerInterval = setInterval(updateTimer, 1000);

  ws = new WebSocket("ws://100.104.164.84:8000/ws");

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.text && data.text.trim() !== "") {
        addTranscriptionBubble(data.text);
      }
    } catch (e) {
      console.log("Error parsing JSON:", e);
    }
  };

  ws.onopen = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new AudioContext();
      await audioContext.audioWorklet.addModule("pcm-worklet.js");

      source = audioContext.createMediaStreamSource(stream);
      processor = new AudioWorkletNode(audioContext, "pcm-worklet");

      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.port.onmessage = (event) => {
        const data = float32ToInt16(event.data);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data.buffer);
        }
      };
    } catch (error) {
      console.error("Error accessing microphone:", error);
      stopRecording();
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

let lastBubbleTime = 0;
let currentMsgContainer = null;
let typingTaskQueue = [];
let isTyping = false;

function processTypingQueue() {
  if (typingTaskQueue.length === 0) {
    isTyping = false;
    return;
  }

  isTyping = true;
  const task = typingTaskQueue[0];

  if (task.text.length > 0) {
    const char = task.text.substring(0, 1);
    task.text = task.text.substring(1);
    task.container.textContent += char;

    const tsList = document.getElementById('tsList');
    tsList.scrollTop = tsList.scrollHeight;

    // Type faster if queue is getting backed up
    let speed = 20;
    const totalCharsPending = typingTaskQueue.reduce((acc, t) => acc + t.text.length, 0);
    if (totalCharsPending > 30) {
      speed = 10;
    }

    setTimeout(processTypingQueue, speed);
  } else {
    typingTaskQueue.shift();
    setTimeout(processTypingQueue, 5);
  }
}

function addTranscriptionBubble(text) {
  const tsList = document.getElementById('tsList');
  const emptyState = document.getElementById('emptyState');
  if (emptyState) emptyState.style.display = 'none';

  const now = new Date();

  // Create new container if 3 seconds passed
  if (!currentMsgContainer || (now.getTime() - lastBubbleTime >= 3000)) {
    const timeString = now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
    const bubbleHTML = `
      <div class="flex flex-col gap-1.5 mt-2">
        <span class="text-[11px] font-bold text-[#aeaeb2] px-1.5">${timeString}</span>
        <div class="flex items-center gap-2 px-1.5 mb-1">
          <div class="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
            <span class="text-[10px] font-bold text-blue-600">나</span>
          </div>
          <span class="text-[11px] font-bold text-[#1d1d1f]">나</span>
        </div>
        <div class="message-bubble px-3.5 py-3 text-[13px] leading-[1.6]"></div>
      </div>
    `;

    const div = document.createElement('div');
    div.innerHTML = bubbleHTML.trim();
    tsList.appendChild(div.firstChild);
    currentMsgContainer = tsList.lastElementChild.querySelector('.message-bubble');
  }

  // Prepend space if existing text is present
  let prependSpace = false;
  const hasTextInCurrent = currentMsgContainer.textContent.length > 0;
  const hasTaskForCurrent = typingTaskQueue.some(t => t.container === currentMsgContainer && t.text.length > 0);
  if (hasTextInCurrent || hasTaskForCurrent) {
    prependSpace = true;
  }

  const formattedText = prependSpace ? " " + text : text;

  typingTaskQueue.push({
    container: currentMsgContainer,
    text: formattedText
  });

  lastBubbleTime = now.getTime();

  if (!isTyping) {
    processTypingQueue();
  }
}
