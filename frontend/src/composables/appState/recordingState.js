import { computed, ref } from 'vue'

const USE_MOCK_DATA = true
const mockTranscriptPlan = [
  { text: '안녕하세요, 실시간 음성 전사 테스트 중입니다.', delay: 3000 },
  { text: '현재는 백엔드 연결 없이 샘플 데이터가 출력되고 있습니다.', delay: 7000 }
]

export function useRecordingState() {
  const isRecording = ref(false)
  const isRecordingPaused = ref(false)
  const recordingSeconds = ref(0)
  const transcriptions = ref([])
  const recordingTimeText = computed(() => {
    const hours = Math.floor(recordingSeconds.value / 3600)
    const minutes = Math.floor(recordingSeconds.value / 60)
    const seconds = recordingSeconds.value % 60
    return `${hours.toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
  })

  let timer = null
  let ws = null
  let audioContext = null
  let stream = null
  let processor = null
  let audioSource = null
  let lastBubbleTime = 0
  let mockTimers = []
  let mockTranscriptQueue = []

  const clearMockTimers = () => {
    mockTimers.forEach(({ timeoutId }) => clearTimeout(timeoutId))
    mockTimers = []
  }

  const resetMockTranscriptQueue = () => {
    clearMockTimers()
    mockTranscriptQueue = mockTranscriptPlan.map((item, index) => ({
      id: index,
      text: item.text,
      delay: item.delay,
      remaining: item.delay,
      startedAt: null,
      fired: false
    }))
  }

  const addTranscriptionBubble = (text, isMock = false) => {
    const now = new Date()
    transcriptions.value.push({
      time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
      text,
      segments: [{
        id: Date.now() + Math.random(),
        text,
        status: isMock ? 'confirmed' : 'pending'
      }]
    })
  }

  const scheduleMockTranscriptions = () => {
    clearMockTimers()

    mockTranscriptQueue
      .filter((item) => !item.fired)
      .forEach((item) => {
        item.startedAt = Date.now()
        const timeoutId = setTimeout(() => {
          item.fired = true
          item.remaining = 0
          addTranscriptionBubble(item.text, true)
          mockTimers = mockTimers.filter((entry) => entry.id !== item.id)
        }, item.remaining)

        mockTimers.push({ id: item.id, timeoutId })
      })
  }

  const syncRecordingTimer = () => {
    clearInterval(timer)
    timer = null

    if (!isRecording.value || isRecordingPaused.value) return

    timer = setInterval(() => {
      recordingSeconds.value += 1
    }, 1000)
  }

  const float32ToInt16 = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length)
    for (let i = 0; i < float32Array.length; i += 1) {
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

    clearMockTimers()
    mockTranscriptQueue = []

    if (processor) { processor.disconnect(); processor = null }
    if (audioSource) { audioSource.disconnect(); audioSource = null }
    if (audioContext) { audioContext.close(); audioContext = null }
    if (stream) { stream.getTracks().forEach((track) => track.stop()); stream = null }
    if (ws) { ws.close(); ws = null }
  }

  const startRecording = async () => {
    isRecording.value = true
    isRecordingPaused.value = false
    recordingSeconds.value = 0

    syncRecordingTimer()

    if (USE_MOCK_DATA) {
      resetMockTranscriptQueue()
      scheduleMockTranscriptions()
      return
    }

    ws = new WebSocket('ws://100.104.164.84:8000/ws')

    let segIdCounter = 0
    const pendingSegmentMap = new Map()
    const autoConfirmTimers = new Map()

    const applyCorrection = (segId, correctedText) => {
      for (const trans of transcriptions.value) {
        const segIdx = trans.segments.findIndex((segment) => segment.id === segId)
        if (segIdx !== -1) {
          trans.segments.splice(segIdx, 1, {
            ...trans.segments[segIdx],
            text: correctedText,
            status: 'confirmed'
          })
          trans.text = trans.segments.map((segment) => segment.text).join(' ')
          break
        }
      }
    }

    const autoConfirmSegment = (segId) => {
      for (const trans of transcriptions.value) {
        const segIdx = trans.segments.findIndex((segment) => segment.id === segId && segment.status === 'pending')
        if (segIdx !== -1) {
          trans.segments.splice(segIdx, 1, {
            ...trans.segments[segIdx],
            status: 'confirmed'
          })
          trans.text = trans.segments.map((segment) => segment.text).join(' ')
          break
        }
      }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (!data.text || data.text.trim() === '') return

        if (data.type === 'corrected') {
          const rawText = data.raw_text
          const correctedText = data.text
          const segId = pendingSegmentMap.get(rawText)

          if (segId !== undefined) {
            clearTimeout(autoConfirmTimers.get(segId))
            autoConfirmTimers.delete(segId)
            pendingSegmentMap.delete(rawText)

            setTimeout(() => {
              applyCorrection(segId, correctedText)
            }, 400)
          }
          return
        }

        const now = new Date()
        const timeSpan = now.getTime() - lastBubbleTime
        const rawText = data.text
        const segId = ++segIdCounter

        if (transcriptions.value.length === 0 || timeSpan >= 3000) {
          transcriptions.value.push({
            time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
            text: rawText,
            segments: [{ id: segId, text: rawText, status: 'pending' }]
          })
        } else {
          const lastIdx = transcriptions.value.length - 1
          const transcript = transcriptions.value[lastIdx]
          transcript.segments.push({ id: segId, text: rawText, status: 'pending' })
          transcript.text = transcript.segments.map((segment) => segment.text).join(' ')
        }
        lastBubbleTime = now.getTime()

        pendingSegmentMap.set(rawText, segId)

        const confirmTimer = setTimeout(() => {
          autoConfirmTimers.delete(segId)
          pendingSegmentMap.delete(rawText)
          autoConfirmSegment(segId)
        }, 5000)
        autoConfirmTimers.set(segId, confirmTimer)
      } catch (error) {
        console.log('Error parsing JSON:', error)
      }
    }

    ws.onopen = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        audioContext = new AudioContext()
        await audioContext.audioWorklet.addModule('/pcm-worklet.js')

        audioSource = audioContext.createMediaStreamSource(stream)
        processor = new AudioWorkletNode(audioContext, 'pcm-worklet')

        audioSource.connect(processor)
        processor.connect(audioContext.destination)

        processor.port.onmessage = (event) => {
          const data = float32ToInt16(event.data)
          if (!isRecordingPaused.value && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(data.buffer)
          }
        }
      } catch (error) {
        console.error('Error accessing microphone:', error)
        stopRecording()
      }
    }
  }

  const pauseRecording = async () => {
    if (!isRecording.value || isRecordingPaused.value) return

    isRecordingPaused.value = true
    syncRecordingTimer()

    if (USE_MOCK_DATA) {
      const now = Date.now()
      mockTranscriptQueue.forEach((item) => {
        if (item.fired || item.startedAt == null) return
        const elapsed = now - item.startedAt
        item.remaining = Math.max(0, item.remaining - elapsed)
        item.startedAt = null
      })
      clearMockTimers()
    }

    if (audioContext && audioContext.state === 'running') {
      await audioContext.suspend()
    }
  }

  const resumeRecording = async () => {
    if (!isRecording.value || !isRecordingPaused.value) return

    isRecordingPaused.value = false
    syncRecordingTimer()

    if (USE_MOCK_DATA) {
      scheduleMockTranscriptions()
      return
    }

    if (audioContext && audioContext.state === 'suspended') {
      await audioContext.resume()
    }
  }

  return {
    transcriptions,
    isRecording,
    isRecordingPaused,
    recordingTimeText,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording
  }
}
