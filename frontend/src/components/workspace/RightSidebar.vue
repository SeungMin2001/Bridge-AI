<!-- 워크스페이스 내에서 AI와 실시간으로 채팅하며 노트를 정리할 수 있는 오른쪽 채팅 패널입니다. -->
<script setup>
import { computed, ref, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import { marked } from 'marked'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  visible: { type: Boolean, default: true },
  aiInput: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  chatSource: { type: Object, default: null }
})

const emit = defineEmits(['update:aiInput', 'openEvidenceSource'])

const { 
  messages, 
  addMessage, 
  updateLastAiMessage,
  openCitePopover
} = useChat()
const isLoading = ref(false)
const isThinkingMode = ref(false)
const aiTextarea = ref(null)
const isSending = ref(false) // 중복 전송 방지용 플래그

// 🚀 [환경 설정] 백엔드 연동 모드 전환 플래그
// true: 백엔드 연결 없이 지정된 한국어 데모 데이터로 즉시 응답합니다.
// false: 실제 백엔드 서버(http://100.104.164.84:8000)로 통신합니다.
// 백엔드 사용시 여부분 주석 처리 조심
const USE_DEMO_DATA = false

function getActiveSessionId() {
  const id = props.activeFileId || ''
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id) ? id : null
}

const workspaceChatSources = computed(() => {
  if (!props.chatSource) return []
  if (Array.isArray(props.chatSource.sources)) return props.chatSource.sources
  return [props.chatSource]
})

const hasWorkspaceChatSource = computed(() => workspaceChatSources.value.length > 0)

function unique(values = []) {
  return Array.from(new Set(values.filter(Boolean).map((item) => String(item))))
}

function shouldUseWorkspaceWideSearch(question = '') {
  const text = String(question || '').replace(/\s+/g, ' ').trim()
  if (!text) return false

  const currentScopeTerms = [
    '현재 파일',
    '이 파일',
    '여기 파일',
    '현재 여기에',
    '여기에 저장',
    '선택된',
    '열려 있는',
    '열려있는',
    '지금 파일',
    '이 강의',
    '이 자료',
    '이 내용',
    '여기 내용',
    '현재 내용'
  ]
  if (currentScopeTerms.some((term) => text.includes(term))) return false

  const pageScopedTerms = [
    /\d+\s*페이지/,
    /\d+\s*p\b/i,
    /p\.\s*\d+/i,
    /pdf/i,
  ]
  if (pageScopedTerms.some((pattern) => pattern.test(text))) return false

  const explicitGlobalTerms = [
    '전체 파일',
    '전체파일',
    '모든 파일',
    '모든파일',
    '전체 자료',
    '모든 자료',
    '워크스페이스 전체',
    '전체 워크스페이스',
    '다른 파일',
    '다른파일',
    '파일들',
    '전체에서',
    '모든 곳'
  ]
  if (explicitGlobalTerms.some((term) => text.includes(term))) return true

  const activeTitle = String(props.chatSource?.title || '').replace(/전체 자료$/, '').replace(/\s+/g, '').trim()
  const sourceHintMatch = text.match(/^(.{2,30}?)(?:의|에서|에는|에)\s+.+/)
  const sourceHint = String(sourceHintMatch?.[1] || '').replace(/\s+/g, '').trim()
  const genericSourceHints = new Set(['이', '그', '저', '현재', '여기', '오늘', '강의', '자료', '내용', '파일'])
  if (
    sourceHint &&
    !genericSourceHints.has(sourceHint) &&
    (!activeTitle || (!activeTitle.includes(sourceHint) && !sourceHint.includes(activeTitle)))
  ) {
    return true
  }

  const locatorTerms = ['어디', '어느', '찾아', '찾아줘', '찾을', '검색', '근거', '링크', '보여', '보여줘']
  const targetTerms = ['파일', '녹음', '녹음본', '전사', '자료', '부분', '구간', '문장', '대목', '내용', '말']
  const mentionTerms = ['언급', '나오', '포함', '있는', '있어', '말했', '다룬', '등장']
  const strongMentionTerms = ['언급', '나오', '포함', '말했', '다룬', '등장']
  const knowledgeQuestionTerms = [
    '누구', '무엇', '뭐야', '뭐여', '뭐냐', '뭐임', '뭐에요', '뭐예요',
    '뭔가', '뭔데', '무슨', '의미', '정의', '설명', '알려', '개념', '뜻'
  ]
  const lookupStopwords = new Set([
    '혹시', '무엇', '뭐', '어디', '어느', '위치', '찾아', '검색', '언급', '부분', '구간',
    '파일', '녹음', '녹음본', '전사', '자료', '내용', '말', '나오', '포함', '있는',
    '있어', '관련', '해당', '대한', '대해', '대해서', '특정', '단어', '표현',
    '키워드', '근거', '링크', '보여', '보여줘'
  ])
  const hasLookupCandidate = (text.match(/[A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30}/g) || [])
    .some((term) => !lookupStopwords.has(term))
  const hasSubjectHint = (
    /[A-Za-z][A-Za-z0-9_+#.-]*/.test(text) ||
    /[가-힣A-Za-z0-9_+#.-]{2,30}\s*(에\s*대한|에대한|에\s*대해|에대해|에\s*대해서|에대해서|라고|이라는|라는)/.test(text) ||
    /(단어|표현|키워드)\s*['"“”‘’]?\s*[가-힣A-Za-z0-9_+#.-]{2,30}/.test(text) ||
    hasLookupCandidate
  )
  const hasLocator = locatorTerms.some((term) => text.includes(term))
  const hasMention = mentionTerms.some((term) => text.includes(term))
  const hasStrongMention = strongMentionTerms.some((term) => text.includes(term))
  const hasTarget = targetTerms.some((term) => text.includes(term))
  const hasKnowledgeQuestion = knowledgeQuestionTerms.some((term) => text.includes(term))

  if (hasSubjectHint && hasKnowledgeQuestion) return true

  return (
    hasSubjectHint &&
    hasMention &&
    (hasLocator || hasTarget || hasStrongMention || text.includes('에 대한') || text.includes('에대한') || text.includes('라고'))
  )
}

function buildSourceFilter() {
  if (!hasWorkspaceChatSource.value) return null

  const materialIds = []
  const storedNames = []
  const recordingIds = []
  const transcriptIds = []

  workspaceChatSources.value.forEach((source) => {
    const material = source.material || {}
    if (source.type === 'material' || material.id || material.storedName) {
      if (material.id || source.materialId) materialIds.push(material.id || source.materialId)
      if (material.storedName || source.storedName) storedNames.push(material.storedName || source.storedName)
      if (source.id && String(source.id).toLowerCase().endsWith('.pdf')) storedNames.push(source.id)
      else if (source.id && !material.id && !source.materialId) materialIds.push(source.id)
    }

    if (source.type === 'recording' || source.recordingId) {
      recordingIds.push(source.recordingId || source.id)
    }

    ;(source.transcriptIds || []).forEach((id) => transcriptIds.push(id))
  })

  ;(props.chatSource?.transcriptIds || []).forEach((id) => transcriptIds.push(id))
  ;(props.chatSource?.recordings || []).forEach((recording) => {
    recordingIds.push(recording?.id || recording?.recordingId)
  })

  const filter = {
    material_ids: unique(materialIds),
    stored_names: unique(storedNames),
    recording_ids: unique(recordingIds),
    transcript_ids: unique(transcriptIds),
  }

  return Object.values(filter).some((items) => items.length) ? filter : null
}

async function sendMessage() {
  const question = props.aiInput.trim()
  if (!question || isSending.value) return // 전송 중이거나 빈 메시지면 무시
  
  isSending.value = true

  addMessage({ role: 'user', text: question })
  emit('update:aiInput', '')
  isLoading.value = true

  const idx = messages.value.length
  messages.value.push({ role: 'ai', text: '', thinking: '', citations: [], phase: 'streaming' })

  /*
  // 1. 데모(목업) 모드 동작 (비활성화)
  if (USE_DEMO_DATA) {
    console.log('[테스트 모드] USE_DEMO_DATA가 true이므로 미리 설정된 데모 데이터를 출력합니다.')
    setTimeout(() => {
      updateLastAiMessage({
        role: 'ai',
        thinking: 'CPU의 정의와 주요 역할을 강의 자료에서 검색했습니다...',
        text: 'CPU(중앙 처리 장치)는 컴퓨터의 두뇌 역할을 하며, 프로그램의 명령어를 해석하고 실행하는 핵심 하드웨어입니다. CPU 내부에는 초고속 임시 저장 공간인 **레지스터** 가 있어, 연산 과정에서 필요한 데이터를 매우 빠르게 접근하고 처리할 수 있습니다.',
        citations: [
          {
            transcript_id: "dd110001-0000-0000-1004",
            text: "CPU 는 명령을 읽고 실행하며 레지스터는 초고속 임시 저장 공간이다.",
            citation: "1 주차 - 데이터 표현과 메모리 > 6:00~8:00",
            session_title: "1주차 - 데이터 표현과 메모리",
            full_transcript: "오늘 수업 시작하겠습니다! 여러분 컴퓨터의 구조에 대해 많이 들어보셨죠?\n그 중에서 가장 핵심이 되는 부품이 뭘까요? 네 맞습니다. CPU입니다.\n\nCPU 는 명령을 읽고 실행하며 레지스터는 초고속 임시 저장 공간이다. 이 점을 꼭 기억하셔야 합니다.\n이러한 구조 덕분에 우리가 원하는 프로그램이 순식간에 처리될 수 있는 것이죠."
          },
          {
            transcript_id: "dd110002-0000-0001-2005",
            text: "운영체제는 하드웨어와 사용자 사이를 중개한다.",
            citation: "컴퓨터공학개론 > 2 주차 - 프로세스와 스레드 > 0:00~2:00",
            session_title: "2주차 - 프로세스와 스레드",
            full_transcript: "자, 지난 시간에는 하드웨어에 대해 배웠죠.\n오늘은 소프트웨어를 배워봅시다. 특히 운영체제에 집중할 건데요.\n운영체제는 하드웨어와 사용자 사이를 중개한다. 이게 가장 중요한 역할입니다.\n마우스 클릭만으로 복잡한 연산이 처리되는게 다 운영체제 덕분이죠."
          }
        ],
        phase: 'done',
      })
      isLoading.value = false
    }, 800)
    return
  }
  //데모(목업) 모드 동작 (비활성화)
*/
  // 2. 실제 백엔드 서버 연동 모드 (SSE 스트리밍)
  const t0 = performance.now()
  let ttftLogged = false
  const workspaceWideSearch = shouldUseWorkspaceWideSearch(question)

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        is_thinking: isThinkingMode.value,
        session_id: workspaceWideSearch ? null : props.chatSource?.sessionId || getActiveSessionId(),
        source_filter: workspaceWideSearch ? null : buildSourceFilter()
      }),
    })

    if (!res.ok) throw new Error(`서버 응답 오류 (상태 코드: ${res.status})`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let streamedText = ''
    let streamedCitations = []
    let tokenCount = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6)
        if (payload === '[DONE]') break

        try {
          const data = JSON.parse(payload)
          if (data.type === 'citations') {
            streamedCitations = data.citations
            updateLastAiMessage({ role: 'ai', text: streamedText, thinking: '', citations: streamedCitations, phase: 'streaming' })
          } else if (data.type === 'token') {
            if (!ttftLogged) {
              console.log(`⏱️ [TTFT] 첫 토큰까지: ${(performance.now() - t0).toFixed(0)}ms`)
              ttftLogged = true
            }
            tokenCount++
            streamedText += data.token
            updateLastAiMessage({ role: 'ai', text: streamedText, thinking: '', citations: streamedCitations, phase: 'streaming' })
          } else if (data.type === 'error') {
            streamedText += `\n오류: ${data.error}`
            updateLastAiMessage({ role: 'ai', text: streamedText, thinking: '', citations: streamedCitations, phase: 'streaming' })
          }
        } catch (parseErr) {
          // SSE 파싱 실패 시 무시
        }
      }
    }

    const totalMs = performance.now() - t0
    console.log(`⏱️ [응답완료] 총: ${totalMs.toFixed(0)}ms | 토큰: ${tokenCount}개 | 속도: ${(tokenCount / (totalMs / 1000)).toFixed(1)} tok/s`)

    updateLastAiMessage({ role: 'ai', text: streamedText, thinking: '', citations: streamedCitations, phase: 'done' })

  } catch (e) {
    console.error('[오류] 실제 백엔드 서버 연결에 실패했습니다.', e)
    updateLastAiMessage({
      role: 'ai',
      thinking: '',
      text: '⚠️ 현재 백엔드 서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인해 주세요.',
      citations: [],
      phase: 'done',
    })
  } finally {
    isLoading.value = false
    isSending.value = false // 전송 완료 후 플래그 해제
    // 전송 후 텍스트에어리어 높이 초기화
    nextTick(() => {
      if (aiTextarea.value) aiTextarea.value.style.height = 'auto'
    })
  }
}

function handleInput(e) {
  emit('update:aiInput', e.target.value)
  // 높이 자동 조절
  nextTick(() => {
    if (aiTextarea.value) {
      aiTextarea.value.style.height = 'auto'
      aiTextarea.value.style.height = aiTextarea.value.scrollHeight + 'px'
    }
  })
}

function handleEnter(e) {
  // 한국어 IME 중복 전송 방지를 위한 엄격한 체크 (keyCode 229는 조합 중을 의미)
  if (e.isComposing || e.keyCode === 229) return
  if (e.shiftKey) return 
  
  e.preventDefault()
  sendMessage()
}

function renderTextWithCitations(text) {
  if (!text) return ''
  // 이미지 스타일을 위해 인라인 [1] 마커 제거 후 마크다운 렌더링
  let processedText = text
    .replace(/\[\d+\]/g, '')
    .replace(/\s*\[출처[:：]?[^\]]*\][^\n]*(?=\n|$)/g, '')
    .trim()
  return marked.parse(processedText)
}

function getSourceChips(msg) {
  const citations = Array.isArray(msg?.citations) ? msg.citations : []
  const seen = new Set()

  return citations.filter((cite, index) => {
    const key = getCitationKey(cite, index)
    if (!key || seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function getCitationKey(cite = {}, index = 0) {
  return String(
    cite.citation
    || cite.material_id
    || cite.transcript_id
    || cite.recording_id
    || cite.stored_name
    || cite.text
    || index
  )
}

function isMaterialCitation(cite = {}) {
  return cite?.source_type === 'material'
}

function compactSourceLabel(value = '', fallback = '근거 자료') {
  const label = String(value || '').replace(/\s+/g, ' ').trim()
  if (!label) return fallback
  return label.length > 28 ? `${label.slice(0, 28).trim()}...` : label
}

function formatCitationSeconds(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value)) return ''
  const totalSeconds = Math.max(0, Math.floor(value))
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, '0')}`
}

function transcriptChipLabel(cite = {}, index = 0) {
  const title = cite.recording_title || cite.session_title || cite.file_title || `전사 ${index + 1}`
  const start = formatCitationSeconds(cite.start_time)
  const end = formatCitationSeconds(cite.end_time)
  if (start && end) return compactSourceLabel(`${title} > ${start}~${end}`, `전사 ${index + 1}`)
  return compactSourceLabel(cite.citation || title, `전사 ${index + 1}`)
}

function sourceChipLabel(cite = {}, index = 0) {
  if (isMaterialCitation(cite)) {
    const title = cite.material_name || cite.file_title || cite.stored_name || `PDF ${index + 1}`
    const page = Number(cite.page || 0)
    return compactSourceLabel(page > 0 ? `${title} p.${page}` : title, `PDF ${index + 1}`)
  }
  return transcriptChipLabel(cite, index)
}

function sourceChipIcon(cite = {}) {
  return isMaterialCitation(cite) ? 'picture_as_pdf' : 'graphic_eq'
}

function clampPosition(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function handleCitationClick(event, cite) {
  if (!cite) return

  const rect = event?.currentTarget?.getBoundingClientRect?.()
  if (!rect) {
    openCitePopover(cite, Math.max(16, window.innerWidth - 380), 96)
    return
  }

  const popoverWidth = 340
  const estimatedPopoverHeight = 430
  const viewportInset = 16
  const maxLeft = Math.max(viewportInset, window.innerWidth - popoverWidth - viewportInset)
  const left = clampPosition(rect.left, viewportInset, maxLeft)

  let top = rect.bottom + 10
  if (top + estimatedPopoverHeight > window.innerHeight - viewportInset) {
    top = Math.max(viewportInset, rect.top - estimatedPopoverHeight - 10)
  }

  openCitePopover(cite, left, top)
}

const width = ref(420)
const isResizing = ref(false)

const handleMouseDown = (e) => {
  isResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
}

const handleMouseMove = (e) => {
  if (!isResizing.value) return
  const newWidth = window.innerWidth - e.clientX - 12
  if (newWidth > 180 && newWidth < 600) width.value = newWidth
}

const handleMouseUp = () => {
  if (!isResizing.value) return
  isResizing.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.body.classList.remove('is-resizing')
}

onMounted(() => {
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})

const scrollContainer = ref(null)

const scrollToBottom = async () => {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({
      top: scrollContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

watch(messages, () => {
  scrollToBottom()
}, { deep: true })

</script>

<template>
  <div
    class="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
    id="resizer-right"
    :class="{ 'is-collapsed': !visible }"
    @mousedown="handleMouseDown"
  >
    <div class="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
  </div>

  <aside
    class="h-full shrink-0 overflow-hidden rounded-[24px] transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)]"
    id="right-sidebar"
    :class="{ 'sidebar-collapsed': !visible }"
    :style="{ width: visible ? `${width}px` : '0px', minWidth: visible ? `${width}px` : '0px', maxWidth: visible ? `${width}px` : '0px' }"
  >
    <div class="card workspace-right-sidebar-card h-full flex flex-col p-4 pt-3.5 relative min-w-[300px]">
      <transition name="fade-slide-switch" mode="out-in">
        <div v-if="messages.length === 0" key="initial-ui" class="flex-1 flex flex-col items-center justify-center px-2">
          <div class="mb-6 flex items-center justify-center">
            <img src="/images/image.png" alt="AI chat" class="w-20 h-auto object-contain" />
          </div>
          <h3 class="text-[18px] font-bold text-[#1d1d1f] mb-8">무엇을 도와드릴까요?</h3>
          <div class="w-full flex flex-col gap-3 mb-10">
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-[22px] text-left">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">description</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">강의 노트 요약하기</span>
            </button>
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-[22px] text-left" @click="emit('update:aiInput', '핵심 개념 퀴즈 생성해줘')">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">quiz</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">핵심 개념 퀴즈 생성</span>
            </button>
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-[22px] text-left">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">translate</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">외국어 자료 번역</span>
            </button>
          </div>
        </div>

        <div v-else key="chat-history" class="flex-1 flex flex-col gap-6 mb-4 overflow-y-auto custom-scrollbar px-1" ref="scrollContainer">
          <transition-group name="chat-bubble">
            <div
              v-for="(msg, i) in messages"
              :key="i"
              :class="msg.role === 'ai' ? 'w-full flex flex-col' : 'w-full flex flex-col items-end gap-1.5'"
            >
              <!-- 사용자 말풍선 -->
              <template v-if="msg.role === 'user'">
                <div class="user-bubble px-4 py-2.5 rounded-[18px] text-white text-[14px] leading-relaxed w-fit max-w-[85%]">
                  {{ msg.text }}
                </div>
              </template>

              <!-- AI 답변 (문서 스타일) -->
              <template v-else>
                <!-- 최종 답변 본문 -->
                <div 
                  v-if="msg.text" 
                  :class="{ 'answer-fade-in': msg.phase === 'answering' || msg.phase === 'done' }"
                  class="ai-doc-feed text-[#1d1d1f] text-[14px] leading-[1.7]"
                  v-html="renderTextWithCitations(msg.text)"
                >
                </div>

                <div v-if="msg.phase === 'done' && getSourceChips(msg).length" class="answer-citation-row" aria-label="근거 자료">
                  <button
                    v-for="(cite, ci) in getSourceChips(msg)"
                    :key="getCitationKey(cite, ci)"
                    type="button"
                    class="answer-citation-pill"
                    :class="{ 'is-material': isMaterialCitation(cite) }"
                    :title="cite.citation || cite.text || sourceChipLabel(cite, ci)"
                    @click="handleCitationClick($event, cite)"
                  >
                    <span class="material-symbols-outlined">{{ sourceChipIcon(cite) }}</span>
                    <span>{{ sourceChipLabel(cite, ci) }}</span>
                  </button>
                </div>

                <div
                  v-if="(msg.phase === 'streaming' || msg.phase === 'thinking') && !msg.text && !msg.thinking"
                  class="ai-stream-wait"
                  role="status"
                  aria-live="polite"
                >
                  <span class="ai-typing-dots" aria-hidden="true">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                </div>
              </template>
            </div>
          </transition-group>
        </div>
      </transition>
        <div class="mt-auto px-1 pb-2">
          <!-- 🎨 다듬어진 프리미엄 입력창 디자인 -->
          <div class="chat-input-glow rounded-[26px] p-3.5 transition-all">
            <textarea
              class="w-full bg-transparent border-none focus:ring-0 p-0 text-[14px] text-[#1d1d1f] placeholder-[#aeaeb2] min-h-[24px] max-h-[120px] resize-none leading-relaxed custom-scrollbar"
              placeholder="무엇이든 물어보세요..."
              rows="1"
              ref="aiTextarea"
              :value="aiInput"
              @input="handleInput"
              @keydown.enter.prevent="handleEnter"
            ></textarea>
            
            <div class="flex items-center justify-between mt-2 pt-1 border-t border-[#f2f2f7]/50">
              <!-- 왼쪽 도구: 첨부 아이콘 -->
              <button class="w-8 h-8 flex items-center justify-center text-[#8e8e93] hover:text-[#1d1d1f] hover:bg-[#f2f2f7] rounded-full transition-all">
                <span class="material-symbols-outlined text-[20px]">attach_file</span>
              </button>

              <div class="flex items-center gap-2">
                <!-- Thinking 모드 버튼 (동작 위주 아이콘) -->
                <button 
                  class="flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all border-none cursor-pointer"
                  :class="isThinkingMode ? 'bg-[#3b82f6]/10 text-[#3b82f6]' : 'bg-[#f2f2f7] text-[#8e8e93] hover:bg-[#e5e5ea]'"
                  @click="isThinkingMode = !isThinkingMode"
                  title="Thinking Mode"
                >
                  <span class="material-symbols-outlined text-[18px]" :class="{ 'animate-pulse': isThinkingMode }">psychology</span>
                  <span class="text-[11px] font-bold tracking-tight">Thinking</span>
                </button>

                <!-- 전송 버튼 -->
                <button 
                  class="w-8 h-8 rounded-full flex items-center justify-center transition-all border-none"
                  :class="aiInput.trim() ? 'bg-[#3b82f6] text-white shadow-sm' : 'bg-[#d1d1d6] text-white cursor-not-allowed'"
                  @click="sendMessage"
                  :disabled="!aiInput.trim()"
                >
                  <span class="material-symbols-outlined text-[18px]">arrow_upward</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
  </aside>
</template>

<style scoped>
.workspace-right-sidebar-card {
  background: var(--workspace-sidebar-card-bg);
  border: 1px solid var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.workspace-right-sidebar-card::before {
  background: var(--workspace-sidebar-card-overlay);
}

.workspace-right-sidebar-card::after {
  border-color: var(--workspace-sidebar-card-inner-border);
}

/* 화면 전환 애니메이션 */
.fade-slide-switch-enter-active,
.fade-slide-switch-leave-active {
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.fade-slide-switch-enter-from {
  opacity: 0;
  transform: translateY(20px);
}
.fade-slide-switch-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

/* 채팅 말풍선 등장 애니메이션 */
.chat-bubble-enter-active {
  transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.chat-bubble-enter-from {
  opacity: 0;
  transform: translateY(15px) scale(0.95);
}

/* 리스트 레이아웃 부드러운 이동 */
.chat-bubble-move {
  transition: transform 0.4s ease;
}

.ai-stream-wait {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  max-width: 100%;
  min-height: 32px;
  margin-top: 4px;
  padding: 5px 2px;
}

.ai-typing-dots {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 24px;
}

.ai-typing-dots span {
  width: 13px;
  height: 13px;
  border-radius: 999px;
  background: #aaa7a3;
  animation: ai-typing-dot 1.05s ease-in-out infinite;
}

.ai-typing-dots span:nth-child(2) {
  animation-delay: 0.16s;
}

.ai-typing-dots span:nth-child(3) {
  animation-delay: 0.32s;
}

@keyframes ai-typing-dot {
  0%, 80%, 100% {
    opacity: 0.58;
    transform: translateY(0) scale(0.86);
  }
  40% {
    opacity: 1;
    transform: translateY(-4px) scale(1);
  }
}

@media (prefers-reduced-motion: reduce) {
  .ai-typing-dots span {
    animation: none;
  }
}

/* 답변 끝에 붙는 ChatGPT 스타일 근거 pill */
.answer-citation-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px;
  margin-top: 10px;
  margin-bottom: 2px;
}

.answer-citation-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: min(100%, 220px);
  padding: 6px 10px;
  color: #475569;
  background: rgba(241, 245, 249, 0.92);
  border: 1px solid rgba(226, 232, 240, 0.96);
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 750;
  line-height: 1.25;
  cursor: pointer;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.86) inset;
  transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.answer-citation-pill:hover {
  color: #111827;
  background: rgba(226, 232, 240, 0.98);
  border-color: rgba(203, 213, 225, 1);
  transform: translateY(-1px);
}

.answer-citation-pill.is-material {
  color: #1d4ed8;
  background: rgba(239, 246, 255, 0.96);
  border-color: rgba(191, 219, 254, 0.98);
}

.answer-citation-pill.is-material:hover {
  color: #1e40af;
  background: rgba(219, 234, 254, 0.98);
}

.answer-citation-pill .material-symbols-outlined {
  font-size: 15px;
  flex: 0 0 auto;
}

.answer-citation-pill span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

:deep(.ai-doc-feed p) {
  margin-bottom: 0.5em;
}
:deep(.ai-doc-feed p:last-child) {
  margin-bottom: 0;
}

</style>
