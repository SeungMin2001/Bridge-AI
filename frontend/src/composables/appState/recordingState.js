import { computed, ref } from 'vue'

// 녹음 버튼 상태, 타이머, 실시간 전사 목록, WebSocket 음성 전송을 관리합니다.
const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_RECORDING === 'true'
const isWorkspaceUuid = (value = '') => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value)
const FINALIZE_TIMEOUT_MS = 180000

const logSpeakerFlow = (step, payload = {}) => {
  if (!import.meta.env.DEV) return
  console.debug(`[speaker-flow] ${step}`, payload)
}

const previewText = (text = '') => String(text).replace(/\s+/g, ' ').trim().slice(0, 80)

const formatElapsedTime = (seconds = 0) => {
  const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0))
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainSeconds = String(safeSeconds % 60).padStart(2, '0')
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${remainSeconds}`
  return `${minutes}:${remainSeconds}`
}

const numberOrNull = (value) => {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

// 신창영: 수정 이유 - 백엔드가 speaker_id만 늦게 보내도 화면에는 사람이 읽기 쉬운 화자 라벨로 표시하기 위함입니다.
const formatSpeakerLabel = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return `화자 ${Number.parseInt(match[1], 10) + 1}`
  return normalized || null
}

const createRecordingId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `recording-${crypto.randomUUID()}`
  }
  return `recording-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

const getRecordingWebSocketUrl = (sessionId = '', recordingId = '', options = {}) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const params = new URLSearchParams()
  if (isWorkspaceUuid(sessionId)) params.set('session_id', sessionId)
  if (recordingId) params.set('recording_id', recordingId)
  params.set('diarize', options.diarizationEnabled ? 'true' : 'false')
  const query = params.toString()
  return `${protocol}//${window.location.host}/ws${query ? `?${query}` : ''}`
}

const getMicrophoneUnavailableReason = () => {
  if (typeof window !== 'undefined') {
    const localhostHosts = new Set(['localhost', '127.0.0.1', '::1'])
    if (!window.isSecureContext && !localhostHosts.has(window.location.hostname)) {
      return '마이크 권한은 HTTPS 또는 localhost 접속에서만 사용할 수 있습니다. 프론트는 http://localhost:5173 으로 열어주세요.'
    }
  }
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return '현재 브라우저가 마이크 입력 API를 제공하지 않습니다.'
  }
  return ''
}

const getAudioContextConstructor = () => {
  if (typeof window === 'undefined') return null
  return window.AudioContext || window.webkitAudioContext || null
}
const mockTranscriptPlanByMode = {
  lecture: [
    { speakerId: 'speaker-me', speaker: '나', text: '안녕하세요, 실시간 음성 전사 테스트 중입니다.', delay: 3000 },
    { speakerId: 'speaker-a', speaker: '화자 A', text: '저도 테스트 회의에 참여했습니다. 다른 화자가 말하면 카드가 자동으로 추가되어야 합니다.', delay: 6000 },
    { speakerId: 'speaker-me', speaker: '나', text: '현재는 백엔드 연결 없이 샘플 데이터가 출력되고 있습니다.', delay: 9000 },
    { speakerId: 'speaker-b', speaker: '화자 B', text: '새 화자가 들어왔을 때 AI 요약 탭에 별도 카드가 생기는지 확인해보겠습니다.', delay: 12000 },
    { speakerId: 'speaker-a', speaker: '화자 A', text: '제가 다시 말하면 기존 화자 A 카드에 발화가 누적되어야 합니다.', delay: 15000 }
  ],
  meeting: [
    { speakerId: 'speaker-a', speaker: '화자 A', text: '오늘 회의에서는 실시간 전사와 화자별 요약 화면을 먼저 확인해보겠습니다.', delay: 2600 },
    { speakerId: 'speaker-b', speaker: '화자 B', text: '좋아요. 백엔드가 없더라도 테스트 데이터로 화자가 늘어나는 흐름을 볼 수 있으면 충분할 것 같습니다.', delay: 5200 },
    { speakerId: 'speaker-a', speaker: '화자 A', text: '우선 전사 데이터에 화자 아이디를 붙이고 AI 요약 탭에서는 그 아이디 기준으로 묶으면 됩니다.', delay: 8200 },
    { speakerId: 'speaker-c', speaker: '화자 C', text: '저는 새 화자가 들어왔을 때 카드가 자동으로 추가되는지 확인하고 싶습니다.', delay: 11200 },
    { speakerId: 'speaker-b', speaker: '화자 B', text: '각 화자 카드에는 방금 말한 내용이 짧게 정리되고 발화 수와 마지막 시간이 보이면 좋겠습니다.', delay: 14200 },
    { speakerId: 'speaker-a', speaker: '화자 A', text: '나중에 실제 화자 분리 모델이 붙으면 같은 데이터 구조로 교체하면 됩니다.', delay: 17200 }
  ]
}

export function useRecordingState() {
  const isRecording = ref(false)
  const isRecordingPaused = ref(false)
  const recordingSeconds = ref(0)
  const transcriptions = ref([])
  const recordingMode = ref('lecture')
  const activeRecordingId = ref('')
  const activeRecordingStartedAt = ref('')
  const diarizationEnabled = ref(false)
  const diarizationStatus = ref('idle')
  const recordingAudioLevel = ref(0)
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
  let currentSessionId = ''
  let finalizeResolve = null
  let finalizeTimer = null

  // 목업 전사 출력을 예약한 타이머를 모두 해제합니다.
  const clearMockTimers = () => {
    mockTimers.forEach(({ timeoutId }) => clearTimeout(timeoutId))
    mockTimers = []
  }

  // 녹음을 새로 시작할 때 목업 전사 큐를 초기 상태로 되돌립니다.
  const resetMockTranscriptQueue = () => {
    clearMockTimers()
    const plan = mockTranscriptPlanByMode[recordingMode.value] || mockTranscriptPlanByMode.lecture
    mockTranscriptQueue = plan.map((item, index) => ({
      id: index,
      speakerId: item.speakerId || null,
      speaker: item.speaker || null,
      text: item.text,
      delay: item.delay,
      remaining: item.delay,
      startedAt: null,
      fired: false
    }))
  }

  // 전사 탭에 말풍선 형태의 전사 결과를 추가합니다.
  const addTranscriptionBubble = (text, isMock = false, speaker = null, speakerId = null) => {
    const start = recordingSeconds.value
    transcriptions.value.push({
      recordingId: activeRecordingId.value,
      time: formatElapsedTime(start),
      speakerId,
      speaker,
      text,
      start,
      end: start,
      segments: [{
        id: Date.now() + Math.random(),
        text,
        start,
        end: start,
        status: isMock ? 'confirmed' : 'pending'
      }]
    })
  }

  // 백엔드 없이 테스트할 수 있도록 목업 전사 문장을 지연 출력합니다.
  const scheduleMockTranscriptions = () => {
    clearMockTimers()

    mockTranscriptQueue
      .filter((item) => !item.fired)
      .forEach((item) => {
        item.startedAt = Date.now()
        const timeoutId = setTimeout(() => {
          item.fired = true
          item.remaining = 0
          addTranscriptionBubble(item.text, true, item.speaker, item.speakerId)
          mockTimers = mockTimers.filter((entry) => entry.id !== item.id)

          // DB에 실험용 데이터 전송
          if (currentSessionId && isWorkspaceUuid(currentSessionId)) {
            fetch(`http://127.0.0.1:8001/api/mock/transcripts`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                session_id: currentSessionId,
                recording_id: activeRecordingId.value,
                text: item.text,
                speakerId: item.speakerId,
                speaker: item.speaker
              })
            }).catch(err => console.error('[mock] mock transcript save failed', err))
          }
        }, item.remaining)

        mockTimers.push({ id: item.id, timeoutId })
      })
  }

  // 녹음 중이고 일시정지가 아닐 때만 녹음 시간을 1초씩 증가시킵니다.
  const syncRecordingTimer = () => {
    clearInterval(timer)
    timer = null

    if (!isRecording.value || isRecordingPaused.value) return

    timer = setInterval(() => {
      recordingSeconds.value += 1
    }, 1000)
  }

  // 브라우저 AudioWorklet의 Float32 PCM 데이터를 백엔드가 받는 Int16 PCM으로 변환합니다.
  const float32ToInt16 = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length)
    for (let i = 0; i < float32Array.length; i += 1) {
      const s = Math.max(-1, Math.min(1, float32Array[i]))
      int16Array[i] = s < 0 ? s * 32768 : s * 32767
    }
    return int16Array
  }

  // 신창영: 수정 이유 - 헤더의 녹음 점을 가짜 애니메이션이 아니라 실제 마이크 입력 음량에 맞춰 움직이게 합니다.
  const updateRecordingAudioLevel = (float32Array) => {
    if (!isRecording.value || isRecordingPaused.value || !float32Array?.length) {
      recordingAudioLevel.value = 0
      return
    }

    let sum = 0
    for (let i = 0; i < float32Array.length; i += 1) {
      sum += float32Array[i] * float32Array[i]
    }

    const rms = Math.sqrt(sum / float32Array.length)
    const normalized = Math.min(1, Math.max(0, (rms - 0.008) * 14))
    recordingAudioLevel.value = (recordingAudioLevel.value * 0.62) + (normalized * 0.38)
  }

  const resolveBackendFinalize = (payload = {}) => {
    if (finalizeTimer) {
      clearTimeout(finalizeTimer)
      finalizeTimer = null
    }
    const resolve = finalizeResolve
    finalizeResolve = null
    if (resolve) resolve(payload)
  }

  const waitForBackendFinalize = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.resolve({ status: 'skipped' })
    }

    return new Promise((resolve) => {
      finalizeResolve = resolve
      finalizeTimer = setTimeout(() => {
        resolveBackendFinalize({ status: 'timeout' })
      }, FINALIZE_TIMEOUT_MS)
      try {
        ws.send(JSON.stringify({ type: 'finalize' }))
      } catch (error) {
        console.warn('[recording] finalize request failed:', error)
        resolveBackendFinalize({ status: 'send_failed' })
      }
    })
  }

  const closeAudioResources = () => {
    if (processor) { processor.disconnect(); processor = null }
    if (audioSource) { audioSource.disconnect(); audioSource = null }
    if (audioContext) { audioContext.close(); audioContext = null }
    if (stream) { stream.getTracks().forEach((track) => track.stop()); stream = null }
  }

  // 녹음/마이크/WebSocket/목업 타이머 등 사용 중인 리소스를 모두 정리합니다.
  const stopRecording = async (options = {}) => {
    const shouldFinalize = options.finalize === true && !USE_MOCK_DATA
    isRecording.value = false
    isRecordingPaused.value = false
    clearInterval(timer)
    timer = null

    clearMockTimers()
    mockTranscriptQueue = []

    closeAudioResources()
    // 신창영: 수정 이유 - 녹음 종료 시 원본 음성 파일 저장과 화자분리 최종 보정을 백엔드에서 마친 뒤 닫습니다.
    const finalizeResult = shouldFinalize
      ? await waitForBackendFinalize()
      : { status: 'skipped' }

    const snapshot = JSON.parse(JSON.stringify(transcriptions.value || []))
    if (ws) { ws.close(); ws = null }
    resolveBackendFinalize({ status: 'closed' })
    // transcriptions.value = [] // 신창잉: 녹음이 끝나면 실시간 전사 목록을 비웁니다.
    activeRecordingId.value = ''
    activeRecordingStartedAt.value = ''
    diarizationStatus.value = 'idle'
    recordingAudioLevel.value = 0
    return {
      finalizeResult,
      transcriptions: snapshot
    }
  }

  // 녹음을 시작하고, 목업 모드가 아니면 마이크 음성을 WebSocket으로 전송합니다.
  const startRecording = async (mode = 'lecture', sessionId = '', recordingId = '', options = {}) => {
    recordingMode.value = mode
    isRecording.value = true
    isRecordingPaused.value = false
    recordingSeconds.value = 0
    transcriptions.value = []
    lastBubbleTime = 0
    currentSessionId = sessionId
    activeRecordingId.value = recordingId || createRecordingId()
    activeRecordingStartedAt.value = new Date().toISOString()
    diarizationEnabled.value = options.diarizationEnabled === true
    diarizationStatus.value = diarizationEnabled.value ? 'bootstrapping' : 'disabled'

    syncRecordingTimer()

    if (USE_MOCK_DATA) {
      resetMockTranscriptQueue()
      scheduleMockTranscriptions()
      return
    }

    ws = new WebSocket(getRecordingWebSocketUrl(sessionId, activeRecordingId.value, {
      diarizationEnabled: diarizationEnabled.value
    }))

    let segIdCounter = 0
    const pendingSegmentMap = new Map()
    const savedSegmentMap = new Map()
    const autoConfirmTimers = new Map()

    // 백엔드에서 교정된 텍스트가 오면 기존 pending segment를 confirmed로 바꿉니다.
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

    // 일정 시간 교정 결과가 없으면 raw 전사를 확정 상태로 바꿉니다.
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

    const applySavedTranscript = (segId, savedData) => {
      for (const trans of transcriptions.value) {
        const segIdx = trans.segments.findIndex((segment) => segment.id === segId)
        if (segIdx !== -1) {
          trans.segments.splice(segIdx, 1, {
            ...trans.segments[segIdx],
            transcript_id: savedData.transcript_id || savedData.transcriptId || '',
            transcriptId: savedData.transcript_id || savedData.transcriptId || '',
            chunk_index: savedData.chunk_index,
            start: savedData.start_time ?? trans.segments[segIdx].start,
            end: savedData.end_time ?? trans.segments[segIdx].end,
            status: 'confirmed'
          })
          trans.text = trans.segments.map((segment) => segment.text).join(' ')
          break
        }
      }
    }

    // 신창영: 수정 이유 - 전사는 먼저 표시하고, 화자분리 결과가 늦게 오면 기존 전사 segment에 speaker_id를 덧붙입니다.
    const applySpeakerUpdates = (items = []) => {
      items.forEach((item) => {
        const chunkId = item.chunk_id || item.chunkId || ''
        const speakerId = item.speaker_id || item.speakerId || ''
        if (!speakerId) return

        const speaker = item.speaker || formatSpeakerLabel(speakerId)
        const start = Number(item.start_time ?? item.startTime)
        const end = Number(item.end_time ?? item.endTime)

        transcriptions.value.forEach((trans) => {
          let matched = false
          trans.segments.forEach((segment) => {
            const sameChunk = chunkId && segment.chunkId === chunkId
            const sameTime = Number.isFinite(start) && Number.isFinite(end) &&
              Math.abs(Number(segment.start) - start) < 0.05 &&
              Math.abs(Number(segment.end) - end) < 0.05

            if (sameChunk || sameTime) {
              segment.speakerId = speakerId
              segment.speaker = speaker
              matched = true
            }
          })

          if (matched) {
            const knownSpeakers = new Set(
              trans.segments
                .map((segment) => segment.speakerId)
                .filter(Boolean)
            )
            if (knownSpeakers.size === 1) {
              trans.speakerId = speakerId
              trans.speaker = speaker
            } else if (!trans.speakerId) {
              trans.speakerId = speakerId
              trans.speaker = speaker
            }
          }
        })
      })
    }

    // 백엔드 WebSocket에서 raw/corrected 전사 메시지를 받아 UI 상태에 반영합니다.
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'diarization') {
          // 8003 화자분리 서버의 세그먼트 결과입니다. 화면에는 직접 표시하지 않고 디버그 흐름 확인에만 씁니다.
          diarizationStatus.value = 'active'
          logSpeakerFlow('backend -> frontend diarization', {
            speakers: Array.from(new Set((data.segments || []).map((segment) => segment.speaker))),
            segmentCount: data.segments?.length || 0,
            segments: data.segments || []
          })
          return
        }

        if (data.type === 'diarization_status') {
          diarizationEnabled.value = data.diarization_enabled === true
          diarizationStatus.value = data.status || (diarizationEnabled.value ? 'active' : 'disabled')
          return
        }

        if (data.type === 'speaker_update') {
          applySpeakerUpdates(data.items || [])
          logSpeakerFlow('backend -> frontend speaker update', {
            updateCount: data.items?.length || 0
          })
          return
        }

        if (data.type === 'finalized') {
          diarizationStatus.value = data.status === 'ok' ? 'finalized' : diarizationStatus.value
          resolveBackendFinalize(data)
          return
        }

        if (!data.text || data.text.trim() === '') return

        if (data.type === 'saved') {
          const segId = savedSegmentMap.get(data.chunk_id) || savedSegmentMap.get(data.raw_text) || savedSegmentMap.get(data.text)
          if (segId !== undefined) {
            applySavedTranscript(segId, data)
            savedSegmentMap.delete(data.chunk_id)
            savedSegmentMap.delete(data.raw_text)
            savedSegmentMap.delete(data.text)
          }
          return
        }

        if (data.type === 'corrected') {
          const rawText = data.raw_text
          const correctedText = data.text
          const segId = pendingSegmentMap.get(data.chunk_id) || pendingSegmentMap.get(rawText)
          logSpeakerFlow('backend -> frontend corrected transcript', {
            speakerId: data.speaker_id || data.speakerId || null,
            text: previewText(correctedText)
          })

          if (segId !== undefined) {
            clearTimeout(autoConfirmTimers.get(segId))
            autoConfirmTimers.delete(segId)
            pendingSegmentMap.delete(rawText)

            setTimeout(() => {
              applyCorrection(segId, correctedText)
            }, 400)
            pendingSegmentMap.delete(data.chunk_id)
          }
          return
        }

        const now = new Date()
        const timeSpan = now.getTime() - lastBubbleTime
        const rawText = data.text
        const segId = ++segIdCounter
        // 8000 백엔드가 STT 결과에 붙여준 speaker_id를 전사 상태에 보관해 요약 탭에서 사용합니다.
        const speakerId = data.speaker_id || data.speakerId || null
        const speaker = data.speaker || formatSpeakerLabel(speakerId)
        const recordingId = data.recording_id || data.recordingId || activeRecordingId.value
        const chunkId = data.chunk_id || data.chunkId || ''
        const start = data.start_time ?? data.startTime ?? null
        const end = data.end_time ?? data.endTime ?? null
        const displayStart = numberOrNull(start) ?? recordingSeconds.value
        if (diarizationEnabled.value && diarizationStatus.value === 'bootstrapping') {
          diarizationStatus.value = 'active'
        }
        const lastTranscript = transcriptions.value[transcriptions.value.length - 1]
        const shouldCreateBubble = (
          !lastTranscript ||
          timeSpan >= 3000 ||
          lastTranscript.segments.length >= 5 ||
          lastTranscript.speakerId !== speakerId ||
          lastTranscript.speaker !== speaker
        )
        // 신창영: 수정 이유 - 화자분리 결과가 늦게 오더라도 화면 말풍선은 자연스럽게 이어 붙이고,
        // 각 segment의 chunkId/start/end는 유지해서 나중에 speaker_update로 화자만 보정합니다.
        logSpeakerFlow('backend -> frontend raw transcript', {
          speakerId,
          speaker,
          recordingId,
          action: shouldCreateBubble ? 'new-bubble' : 'append-bubble',
          text: previewText(rawText)
        })

        if (shouldCreateBubble) {
          transcriptions.value.push({
            recordingId,
            time: formatElapsedTime(displayStart),
            speakerId,
            speaker,
            text: rawText,
            start,
            end,
            segments: [{ id: segId, chunkId, text: rawText, start, end, speakerId, speaker, status: 'pending' }]
          })
        } else {
          const lastIdx = transcriptions.value.length - 1
          const transcript = transcriptions.value[lastIdx]
          transcript.end = end ?? transcript.end
          transcript.segments.push({ id: segId, chunkId, text: rawText, start, end, speakerId, speaker, status: 'pending' })
          transcript.text = transcript.segments.map((segment) => segment.text).join(' ')
        }
        lastBubbleTime = now.getTime()

        if (chunkId) {
          pendingSegmentMap.set(chunkId, segId)
          savedSegmentMap.set(chunkId, segId)
        }
        pendingSegmentMap.set(rawText, segId)
        savedSegmentMap.set(rawText, segId)

        const confirmTimer = setTimeout(() => {
          autoConfirmTimers.delete(segId)
          pendingSegmentMap.delete(chunkId)
          pendingSegmentMap.delete(rawText)
          autoConfirmSegment(segId)
        }, 5000)
        autoConfirmTimers.set(segId, confirmTimer)
      } catch (error) {
        console.log('Error parsing JSON:', error)
      }
    }

    ws.onclose = () => {
      resolveBackendFinalize({ status: 'closed' })
    }

    // WebSocket 연결 후 마이크 권한을 얻고 AudioWorklet으로 PCM 청크를 전송합니다.
    ws.onopen = async () => {
      try {
        const microphoneUnavailableReason = getMicrophoneUnavailableReason()
        if (microphoneUnavailableReason) throw new Error(microphoneUnavailableReason)

        const AudioContextConstructor = getAudioContextConstructor()
        if (!AudioContextConstructor) throw new Error('현재 브라우저가 Web Audio API를 제공하지 않습니다.')

        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        audioContext = new AudioContextConstructor()
        await audioContext.audioWorklet.addModule('/pcm-worklet.js')

        audioSource = audioContext.createMediaStreamSource(stream)
        processor = new AudioWorkletNode(audioContext, 'pcm-worklet')

        audioSource.connect(processor)
        processor.connect(audioContext.destination)

        processor.port.onmessage = (event) => {
          updateRecordingAudioLevel(event.data)
          const data = float32ToInt16(event.data)
          if (isRecording.value && !isRecordingPaused.value && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(data.buffer)
          }
        }
      } catch (error) {
        console.error('Error accessing microphone:', error)
        stopRecording()
      }
    }
  }

  // 녹음 시간을 멈추고, 목업/실제 오디오 스트림도 일시정지합니다.
  const pauseRecording = async () => {
    if (!isRecording.value || isRecordingPaused.value) return

    isRecordingPaused.value = true
    recordingAudioLevel.value = 0
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

  // 일시정지된 녹음을 재개하고, 목업/실제 오디오 스트림도 다시 진행합니다.
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
    recordingMode,
    recordingTimeText,
    recordingAudioLevel,
    activeRecordingId,
    activeRecordingStartedAt,
    diarizationEnabled,
    diarizationStatus,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording
  }
}
