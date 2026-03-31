<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import Workspace from './pages/Workspace/Workspace.vue'
import Home from './pages/Home/Home.vue'
import Workfolder from './pages/Workfolder/Workfolder.vue'
import AiHistory from './pages/AiHistory/AiHistory.vue'

// --- 상태 관리 (State) ---
const currentView = ref('home')
const isRecording = ref(false)
const recordingSeconds = ref(0)
const transcriptions = ref([])
const activeFileName = ref('강의1')
const isRightSidebarVisible = ref(true)

// --- 추가된 통합 데이터 상태 ---
const summaryNotes = ref([
  { id: 1, text: "데모 데이터: AI가 전사한 내용을 여기에 정리할 수 있습니다.", source: "AI 분석 결과", time: "12:00 PM" }
])
const aiInput = ref('')

// --- 통합 데이터 상태 (Home & Sidebar 공유) ---
const fileTree = ref([])
const favorites = ref(new Set())

onMounted(() => {
  const savedTree = localStorage.getItem('lecto_file_tree')
  if (savedTree) fileTree.value = JSON.parse(savedTree)
  
  const savedFavs = localStorage.getItem('lecto_favorites')
  if (savedFavs) favorites.value = new Set(JSON.parse(savedFavs))
})

watch(fileTree, (newVal) => {
  localStorage.setItem('lecto_file_tree', JSON.stringify(newVal))
}, { deep: true })

watch(favorites, (newVal) => {
  localStorage.setItem('lecto_favorites', JSON.stringify(Array.from(newVal)))
}, { deep: true })

// --- 참조 관리 (Refs) ---
let timer = null
let ws = null
let audioContext = null
let stream = null
let processor = null
let audioSource = null
let lastBubbleTime = 0
let mockTimers = []

const float32ToInt16 = (float32Array) => {
  const int16Array = new Int16Array(float32Array.length)
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]))
    int16Array[i] = s < 0 ? s * 32768 : s * 32767
  }
  return int16Array
}

const stopRecording = () => {
  isRecording.value = false
  clearInterval(timer)

  mockTimers.forEach(t => clearTimeout(t))
  mockTimers = []

  if (processor) { processor.disconnect(); processor = null }
  if (audioSource) { audioSource.disconnect(); audioSource = null }
  if (audioContext) { audioContext.close(); audioContext = null }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null }
  if (ws) { ws.close(); ws = null }
}

const addTranscriptionBubble = (text) => {
  const now = new Date()
  const timeSpan = now.getTime() - lastBubbleTime

  if (transcriptions.value.length === 0 || timeSpan >= 3000) {
    transcriptions.value.push({
      time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
      text: text
    })
  } else {
    const lastIdx = transcriptions.value.length - 1
    transcriptions.value[lastIdx].text += " " + text
  }
  
  lastBubbleTime = now.getTime()
}

const startRecording = async () => {
  isRecording.value = true
  recordingSeconds.value = 0

  timer = setInterval(() => {
    recordingSeconds.value++
  }, 1000)

  // Mock Data 설정
  // 전사 테스트
  const USE_MOCK_DATA = false

  if (USE_MOCK_DATA) {
    const t1 = setTimeout(() => {
      addTranscriptionBubble("안녕하세요, 실시간 음성 전사 테스트 중입니다.")
    }, 3000)
    const t2 = setTimeout(() => {
      addTranscriptionBubble("현재는 백엔드 연결 없이 샘플 데이터가 출력되고 있습니다.")
    }, 7000)
    mockTimers = [t1, t2]
    return
  }

  // --- 실시간 백엔드 연결 (WebSocket) ---
  // 웹 서켓과 백엔드하고 연동- aki
  // 
  ws = new WebSocket("ws://100.104.164.84:8000/ws")

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.text && data.text.trim() !== "") {
        addTranscriptionBubble(data.text)
      }
    } catch (e) {
      console.log("Error parsing JSON:", e)
    }
  }

  ws.onopen = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioContext = new AudioContext()
      await audioContext.audioWorklet.addModule("/pcm-worklet.js")

      audioSource = audioContext.createMediaStreamSource(stream)
      processor = new AudioWorkletNode(audioContext, "pcm-worklet")

      audioSource.connect(processor)
      processor.connect(audioContext.destination)

      processor.port.onmessage = (event) => {
        const data = float32ToInt16(event.data)
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data.buffer)
        }
      }
    } catch (error) {
      console.error("Error accessing microphone:", error)
      stopRecording()
    }
  }
}

onUnmounted(() => {
  stopRecording()
})

const formatTime = () => {
  const minutes = Math.floor(recordingSeconds.value / 60)
  const seconds = recordingSeconds.value % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

const handleFileSelect = (id, node) => {
  if (node) activeFileName.value = node.name
}

const handleRightSidebarToggle = () => {
  isRightSidebarVisible.value = !isRightSidebarVisible.value
}

const handleAddToNote = (text, source) => {
  const now = new Date()
  summaryNotes.value.push({
    id: Date.now(),
    text: text,
    source: source || 'AI 분석 결과',
    time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
  })
}

const handleAskAi = (word) => {
  aiInput.value = word
  isRightSidebarVisible.value = true
}

const handleNavigate = (view) => {
  currentView.value = view
}
</script>

<template>
  <AiHistory 
    v-if="currentView === 'ai-history'" 
    @navigateBack="handleNavigate('home')" 
  />

  <Workfolder 
    v-else-if="currentView === 'workfolder'"
    :fileTree="fileTree"
    :favorites="favorites"
    @update:fileTree="fileTree = $event"
    @update:favorites="favorites = $event"
    @navigate="handleNavigate"
  />

  <Home 
    v-else-if="currentView === 'home'"
    :fileTree="fileTree"
    :favorites="favorites"
    @update:fileTree="fileTree = $event"
    @update:favorites="favorites = $event"
    @navigate="handleNavigate"
  />

  <Workspace 
    v-else
    :fileTree="fileTree"
    :favorites="favorites"
    :transcriptions="transcriptions"
    :isRecording="isRecording"
    :recordingTimeText="formatTime()"
    :activeFileName="activeFileName"
    :isRightSidebarVisible="isRightSidebarVisible"
    :summaryNotes="summaryNotes"
    :aiInput="aiInput"
    @update:fileTree="fileTree = $event"
    @update:favorites="favorites = $event"
    @update:aiInput="aiInput = $event"
    @navigateHome="handleNavigate('home')"
    @fileSelect="handleFileSelect"
    @startRecording="startRecording"
    @stopRecording="stopRecording"
    @rightSidebarToggle="handleRightSidebarToggle"
    @addToNote="handleAddToNote"
    @askAi="handleAskAi"
  />
</template>
