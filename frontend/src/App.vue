<!-- 애플리케이션의 루트 컴포넌트로, 전체 레이아웃과 라우팅, 전역 상태(음성 녹음, 파일 트리 등)를 관리합니다. -->
<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import Workspace from './pages/Workspace/Workspace.vue'
import Home from './pages/Home/Home.vue'
import Workfolder from './pages/Workfolder/Workfolder.vue'
import AiHistory from './pages/AiHistory/AiHistory.vue'

// --- 상태 관리 (State) ---
const currentView = ref('home')
const isRecording = ref(false)
const isRecordingPaused = ref(false)
const recordingSeconds = ref(0)
const transcriptions = ref([])
const activeFileName = ref('강의1')
const activeFileId = ref('lecture-1')
const isRightSidebarVisible = ref(true)
const currentPreviewMaterial = ref(null)

// --- 추가된 통합 데이터 상태 ---
const summaryNotes = ref([
  { id: 1, text: "데모 데이터: AI가 전사한 내용을 여기에 정리할 수 있습니다.", source: "AI 분석 결과", time: "12:00 PM" }
])
const aiInput = ref('')

// --- 통합 데이터 상태 (Home & Sidebar 공유) ---
const fileTree = ref([])
const favorites = ref(new Set())

const createLectureOneNode = () => ({
  id: 'lecture-1',
  type: 'file',
  name: '강의1',
  content: '',
  attachments: []
})

const ensureLectureOneFile = (nodes) => {
  const list = Array.isArray(nodes) ? [...nodes] : []
  const existingIndex = list.findIndex(node => node?.id === 'lecture-1' || node?.name === '강의1')

  if (existingIndex === -1) {
    return [createLectureOneNode(), ...list]
  }

  const existingNode = list[existingIndex]
  list[existingIndex] = {
    ...existingNode,
    id: existingNode.id || 'lecture-1',
    content: existingNode.content || '',
    attachments: Array.isArray(existingNode.attachments) ? existingNode.attachments : []
  }

  return list
}

const updateNodeById = (nodes, targetId, updater) => {
  return nodes.map((node) => {
    if (node.id === targetId) {
      return updater(node)
    }

    if (node.children) {
      return {
        ...node,
        children: updateNodeById(node.children, targetId, updater)
      }
    }

    return node
  })
}

const findNodeById = (nodes, targetId) => {
  for (const node of nodes) {
    if (node.id === targetId) return node
    if (node.children) {
      const found = findNodeById(node.children, targetId)
      if (found) return found
    }
  }
  return null
}

const currentFileNode = computed(() => findNodeById(fileTree.value, activeFileId.value))
const currentAttachments = computed(() => currentFileNode.value?.attachments || [])

onMounted(() => {
  const savedTree = localStorage.getItem('lecto_file_tree')
  if (savedTree) {
    fileTree.value = ensureLectureOneFile(JSON.parse(savedTree))
  } else {
    fileTree.value = ensureLectureOneFile([])
  }
  
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

const syncRecordingTimer = () => {
  clearInterval(timer)
  timer = null

  if (!isRecording.value || isRecordingPaused.value) return

  timer = setInterval(() => {
    recordingSeconds.value++
  }, 1000)
}

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
  isRecordingPaused.value = false
  clearInterval(timer)
  timer = null

  mockTimers.forEach(t => clearTimeout(t))
  mockTimers = []

  if (processor) { processor.disconnect(); processor = null }
  if (audioSource) { audioSource.disconnect(); audioSource = null }
  if (audioContext) { audioContext.close(); audioContext = null }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null }
  if (ws) { ws.close(); ws = null }
}

// 전사 테스트 관련코드
const addTranscriptionBubble = (text, isMock = false) => {
  const now = new Date()
  transcriptions.value.push({
    time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
    text: text,
    segments: [{ 
      id: Date.now() + Math.random(),
      text: text, 
      status: isMock ? 'confirmed' : 'pending' 
    }]
  })
}
//전사 테스트 코드 끝

const startRecording = async () => {
  isRecording.value = true
  isRecordingPaused.value = false
  recordingSeconds.value = 0

  syncRecordingTimer()

  // Mock Data 설정
  // 전사 테스트 모드
  //백엔드 킬때 false로
  const USE_MOCK_DATA = true

  if (USE_MOCK_DATA) {
    const t1 = setTimeout(() => {
      addTranscriptionBubble("안녕하세요, 실시간 음성 전사 테스트 중입니다.", true)
    }, 3000)
    const t2 = setTimeout(() => {
      addTranscriptionBubble("현재는 백엔드 연결 없이 샘플 데이터가 출력되고 있습니다.", true)
    }, 7000)
    mockTimers = [t1, t2]
    return
  }

  // --- 실시간 백엔드 연결 (WebSocket) ---
  // 웹 서켓과 백엔드하고 연동- aki
  // 
  ws = new WebSocket("ws://100.104.164.84:8000/ws")

  let segIdCounter = 0

  // rawText → segId 매핑: corrected 메시지가 올 때 정확하게 세그먼트를 찾기 위함
  const pendingSegmentMap = new Map()
  // segId → setTimeout handle: corrected 도착 시 자동 confirm 취소
  const autoConfirmTimers = new Map()

  // segId로 전체 버블에서 세그먼트를 찾아 업데이트
  const applyCorrection = (segId, correctedText) => {
    for (const trans of transcriptions.value) {
      const segIdx = trans.segments.findIndex(s => s.id === segId)
      if (segIdx !== -1) {
        trans.segments.splice(segIdx, 1, {
          ...trans.segments[segIdx],
          text: correctedText,
          status: 'confirmed'
        })
        trans.text = trans.segments.map(s => s.text).join(' ')
        break
      }
    }
  }

  // segId로 전체 버블에서 세그먼트를 pending→confirmed 자동 전환
  const autoConfirmSegment = (segId) => {
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
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.text && data.text.trim() !== "") {
        if (data.type === 'corrected') {
          const rawText = data.raw_text
          const correctedText = data.text
          const segId = pendingSegmentMap.get(rawText)

          if (segId !== undefined) {
            // corrected가 왔으므로 자동 confirm 타이머 취소
            clearTimeout(autoConfirmTimers.get(segId))
            autoConfirmTimers.delete(segId)
            pendingSegmentMap.delete(rawText)

            // pending 상태가 최소 400ms 보이도록 딜레이 후 교정 적용
            setTimeout(() => {
              applyCorrection(segId, correctedText)
            }, 400)
          }
        } else {
          // raw 텍스트 수신
          const now = new Date()
          const timeSpan = now.getTime() - lastBubbleTime
          const rawText = data.text
          const segId = ++segIdCounter

          // 세그먼트 추가
          if (transcriptions.value.length === 0 || timeSpan >= 3000) {
            transcriptions.value.push({
              time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
              text: rawText,
              segments: [{ id: segId, text: rawText, status: 'pending' }]
            })
          } else {
            const lastIdx = transcriptions.value.length - 1
            const t = transcriptions.value[lastIdx]
            t.segments.push({ id: segId, text: rawText, status: 'pending' })
            t.text = t.segments.map(s => s.text).join(' ')
          }
          lastBubbleTime = now.getTime()

          // rawText → segId 등록 (corrected 메시지 도착 시 정확한 매칭에 사용)
          pendingSegmentMap.set(rawText, segId)

          // 백엔드가 corrected를 보내지 않는 경우를 위한 자동 confirmed 처리
          // (교정 결과가 raw와 동일하면 백엔드는 corrected를 전송하지 않음)
          const timer = setTimeout(() => {
            autoConfirmTimers.delete(segId)
            pendingSegmentMap.delete(rawText)
            autoConfirmSegment(segId)
          }, 5000) // 교정 모델이 느릴 수 있으므로 5초 여유
          autoConfirmTimers.set(segId, timer)
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
        if (!isRecordingPaused.value && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data.buffer)
        }
      }
    } catch (error) {
      console.error("Error accessing microphone:", error)
      stopRecording()
    }
  }
}

const pauseRecording = async () => {
  if (!isRecording.value || isRecordingPaused.value) return

  isRecordingPaused.value = true
  syncRecordingTimer()

  mockTimers.forEach(t => clearTimeout(t))
  mockTimers = []

  if (audioContext && audioContext.state === 'running') {
    await audioContext.suspend()
  }
}

const resumeRecording = async () => {
  if (!isRecording.value || !isRecordingPaused.value) return

  isRecordingPaused.value = false
  syncRecordingTimer()

  if (audioContext && audioContext.state === 'suspended') {
    await audioContext.resume()
  }
}

onUnmounted(() => {
  stopRecording()
})

const formatTime = () => {
  const hours = Math.floor(recordingSeconds.value / 3600)
  const minutes = Math.floor(recordingSeconds.value / 60)
  const seconds = recordingSeconds.value % 60
  return `${hours.toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
}

const handleFileSelect = (id, node) => {
  if (node) {
    activeFileId.value = id
    activeFileName.value = node.name
  }
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

const handleUploadLectureMaterials = (files) => {
  const file = files[0]
  if (!file) return

  const uploadedAt = new Date().toISOString()
  const nextAttachment = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: file.name,
    size: file.size,
    type: file.type,
    uploadedAt,
    url: URL.createObjectURL(file),
    sourceFile: file
  }

  fileTree.value = updateNodeById(
    ensureLectureOneFile(fileTree.value),
    'lecture-1',
    (node) => ({
      ...node,
      attachments: [nextAttachment, ...(node.attachments || [])]
    })
  )

  currentPreviewMaterial.value = nextAttachment
  activeFileId.value = 'lecture-1'
  activeFileName.value = '강의1'
}

const handleClosePreviewMaterial = () => {
  currentPreviewMaterial.value = null
}

const handleOpenStoredMaterial = (materialId) => {
  const target = currentAttachments.value.find((item) => item.id === materialId)
  if (!target) return

  currentPreviewMaterial.value = target
}

const handleDeleteStoredMaterial = (materialId) => {
  fileTree.value = updateNodeById(
    ensureLectureOneFile(fileTree.value),
    activeFileId.value,
    (node) => ({
      ...node,
      attachments: (node.attachments || []).filter((item) => item.id !== materialId)
    })
  )

  if (currentPreviewMaterial.value?.id === materialId) {
    currentPreviewMaterial.value = null
  }
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
    @update:fileTree="fileTree = ensureLectureOneFile($event)"
    @update:favorites="favorites = $event"
    @navigate="handleNavigate"
  />

  <Home 
    v-else-if="currentView === 'home'"
    :fileTree="fileTree"
    :favorites="favorites"
    @update:fileTree="fileTree = ensureLectureOneFile($event)"
    @update:favorites="favorites = $event"
    @navigate="handleNavigate"
  />

  <Workspace 
    v-else
    :fileTree="fileTree"
    :favorites="favorites"
    :transcriptions="transcriptions"
    :isRecording="isRecording"
    :isRecordingPaused="isRecordingPaused"
    :recordingTimeText="formatTime()"
    :activeFileName="activeFileName"
    :activeFileId="activeFileId"
    :currentAttachments="currentAttachments"
    :currentPreviewMaterial="currentPreviewMaterial"
    :isRightSidebarVisible="isRightSidebarVisible"
    :summaryNotes="summaryNotes"
    :aiInput="aiInput"
    @update:fileTree="fileTree = ensureLectureOneFile($event)"
    @update:favorites="favorites = $event"
    @update:aiInput="aiInput = $event"
    @navigateHome="handleNavigate('home')"
    @fileSelect="handleFileSelect"
    @startRecording="startRecording"
    @pauseRecording="pauseRecording"
    @resumeRecording="resumeRecording"
    @stopRecording="stopRecording"
    @rightSidebarToggle="handleRightSidebarToggle"
    @addToNote="handleAddToNote"
    @askAi="handleAskAi"
    @uploadLectureMaterials="handleUploadLectureMaterials"
    @closePreviewMaterial="handleClosePreviewMaterial"
    @openStoredMaterial="handleOpenStoredMaterial"
    @deleteStoredMaterial="handleDeleteStoredMaterial"
  />
</template>
