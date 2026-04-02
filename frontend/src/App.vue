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

let segIdCounter = 0

const addTranscriptionBubble = (text, status = 'confirmed') => {
  const now = new Date()
  const timeSpan = now.getTime() - lastBubbleTime
  const segId = ++segIdCounter

  if (transcriptions.value.length === 0 || timeSpan >= 3000) {
    transcriptions.value.push({
      time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
      text: text,
      segments: [{ id: segId, text, status }]
    })
  } else {
    const lastIdx = transcriptions.value.length - 1
    const t = transcriptions.value[lastIdx]
    t.segments.push({ id: segId, text, status })
    t.text = t.segments.map(s => s.text).join(' ')
  }

  lastBubbleTime = now.getTime()
  return segId
}

const updateLastSegment = (rawText, correctedText) => {
  if (transcriptions.value.length === 0) return
  const t = transcriptions.value[transcriptions.value.length - 1]
  // 가장 최근의 pending 세그먼트를 역순으로 찾음 (rawText 우선, 없으면 가장 최근 pending)
  let segIdx = [...t.segments].map((s, i) => i).reverse().find(
    i => t.segments[i].status === 'pending' && t.segments[i].rawText === rawText
  )
  // fallback: rawText 매칭 없을 경우 가장 최근 pending 사용
  if (segIdx === undefined) {
    segIdx = [...t.segments].map((s, i) => i).reverse().find(
      i => t.segments[i].status === 'pending'
    )
  }
  if (segIdx !== undefined) {
    // Vue 반응성을 위해 splice로 객체 교체
    t.segments.splice(segIdx, 1, {
      ...t.segments[segIdx],
      text: correctedText,
      status: 'confirmed'
    })
    t.text = t.segments.map(s => s.text).join(' ')
  }
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
        if (data.type === 'corrected') {
          // pending 상태가 최소 400ms 동안 보이도록 딜레이 후 교정 적용
          const rawText = data.raw_text
          const correctedText = data.text
          setTimeout(() => {
            updateLastSegment(rawText, correctedText)
          }, 400)
        } else {
          // raw 텍스트 추가: rawText 저장 + 자동 confirmed 타임아웃 설정
          const segId = ++segIdCounter
          const now = new Date()
          const timeSpan = now.getTime() - lastBubbleTime
          const rawText = data.text

          if (transcriptions.value.length === 0 || timeSpan >= 3000) {
            transcriptions.value.push({
              time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
              text: rawText,
              segments: [{ id: segId, text: rawText, rawText, status: 'pending' }]
            })
          } else {
            const lastIdx = transcriptions.value.length - 1
            const t = transcriptions.value[lastIdx]
            t.segments.push({ id: segId, text: rawText, rawText, status: 'pending' })
            t.text = t.segments.map(s => s.text).join(' ')
          }
          lastBubbleTime = now.getTime()

          // 백엔드가 corrected를 보내지 않는 경우(교정 불필요 or 교정 disabled)를 위한 자동 confirmed 처리
          // 2.5초 후에도 pending 상태이면 confirmed로 자동 전환
          setTimeout(() => {
            for (const trans of transcriptions.value) {
              const segIdx = trans.segments.findIndex(s => s.id === segId && s.status === 'pending')
              if (segIdx !== -1) {
                trans.segments.splice(segIdx, 1, {
                  ...trans.segments[segIdx],
                  status: 'confirmed'
                })
                trans.text = trans.segments.map(s => s.text).join(' ')
                break
              }
            }
          }, 2500)
        }
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
