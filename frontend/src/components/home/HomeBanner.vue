<!-- 홈 화면 상단에 표시되는 환영 문구와 광고/안내 배너를 포함하는 컴포넌트입니다. -->
<script setup>
import { ref, defineEmits, nextTick } from 'vue'
import MultimodalInput from './MultimodalInput.vue'
import LoadingHourglass from '../ui/LoadingHourglass.vue'
import { marked } from 'marked'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  recentFiles: { type: Array, default: () => [] }
})

const emit = defineEmits(['sendMessage', 'openReference', 'openRecentFile'])

const messages = ref([])
const isGenerating = ref(false)
const chatScrollRef = ref(null)
const abortController = ref(null)
const expandedReferenceMessages = ref(new Set())

// true: 백엔드 없이 홈 AI 채팅에서 데모 응답을 표시합니다.
// false: 실제 /chat/stream 엔드포인트를 호출합니다.
const USE_DEMO_DATA = false

const scrollToBottom = () => {
  nextTick(() => {
    if (chatScrollRef.value) {
      chatScrollRef.value.scrollTop = chatScrollRef.value.scrollHeight
    }
  })
}

const getFileIcon = (type) => {
  switch (type) {
    case 'pdf': return 'picture_as_pdf'
    case 'ppt': return 'slideshow'
    case 'doc': return 'description'
    case 'meeting': return 'groups_2'
    case 'lecture': return 'article'
    default: return 'insert_drive_file'
  }
}

const colorWithAlpha = (color = '#6366f1', alpha = 0.12) => {
  const hex = String(color || '').trim()
  const fullHex = /^#[0-9a-fA-F]{6}$/.test(hex)
    ? hex
    : (/^#[0-9a-fA-F]{3}$/.test(hex)
        ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
        : '#6366f1')
  const value = fullHex.slice(1)
  const red = parseInt(value.slice(0, 2), 16)
  const green = parseInt(value.slice(2, 4), 16)
  const blue = parseInt(value.slice(4, 6), 16)
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

const getRecentFileNode = (file = {}) => file.node || file
const getRecentFileColor = (file = {}) => getRecentFileNode(file)?.color || '#6366f1'
const isRecentMeetingFile = (file = {}) => {
  const node = getRecentFileNode(file)
  return node?.fileKind === 'meeting' || node?.tag === '회의' || file?.type === 'meeting'
}
const getRecentFileIcon = (file = {}) => {
  const node = getRecentFileNode(file)
  if (node?.fileIcon) return node.fileIcon
  if (node?.tag === '프로젝트') return 'workspaces'
  if (node?.tag === '개인') return 'person'
  if (node?.tag === '중요') return 'priority_high'
  return isRecentMeetingFile(file) ? 'groups_2' : 'article'
}
const getRecentFileTag = (file = {}) => {
  const node = getRecentFileNode(file)
  return node?.tag || (isRecentMeetingFile(file) ? '회의' : '강의')
}

const cleanAssistantContent = (content = '') => {
  return String(content)
    .split('\n')
    .filter((line) => !/^\s*(\[?\s*(출처|참고\s*출처)\s*\]?|출처\s*\d+)\s*[:：]/i.test(line.trim()))
    .join('\n')
    .replace(/\s*\[출처\s*\d+\]/g, '')
    .replace(/\s*\[출처[:：]?[^\]]*\]\s*$/i, '')
    .trim()
}

const isAssistantTyping = (msg = {}) => {
  return msg.role === 'assistant' && msg.phase === 'streaming' && !cleanAssistantContent(msg.content).trim()
}

const renderAssistantContent = (content = '') => {
  return marked.parse(cleanAssistantContent(content) || '')
}

const mapCitationsToReferences = (citations = []) => {
  return citations.map((cite, index) => ({
    id: cite.transcript_id || `cite-${index}`,
    title: cite.citation || cite.session_title || `근거 ${index + 1}`,
    script: cite.full_transcript || cite.text || '',
    raw: cite,
  }))
}

const toggleReferenceMessage = (index) => {
  const next = new Set(expandedReferenceMessages.value)
  if (next.has(index)) next.delete(index)
  else next.add(index)
  expandedReferenceMessages.value = next
}

const isReferenceMessageExpanded = (index) => expandedReferenceMessages.value.has(index)

const demoResponse = {
  content: 'CPU(중앙 처리 장치)는 컴퓨터의 두뇌 역할을 하며, 프로그램의 명령어를 해석하고 실행하는 핵심 하드웨어입니다. CPU 내부에는 초고속 임시 저장 공간인 레지스터가 있어 연산 과정에 필요한 데이터를 매우 빠르게 처리할 수 있습니다.',
  citations: [
    {
      transcript_id: 'dd110001-0000-0000-1004',
      text: 'CPU 는 명령을 읽고 실행하며 레지스터는 초고속 임시 저장 공간이다.',
      citation: '1 주차 - 데이터 표현과 메모리 > 6:00~8:00',
      session_title: '1주차 - 데이터 표현과 메모리',
      full_transcript: '오늘 수업 시작하겠습니다! 여러분 컴퓨터의 구조에 대해 많이 들어보셨죠?\n그 중에서 가장 핵심이 되는 부품이 뭘까요? 네 맞습니다. CPU입니다.\n\nCPU 는 명령을 읽고 실행하며 레지스터는 초고속 임시 저장 공간이다. 이 점을 꼭 기억하셔야 합니다.\n이러한 구조 덕분에 우리가 원하는 프로그램이 순식간에 처리될 수 있는 것이죠.'
    },
    {
      transcript_id: 'dd110002-0000-0001-2005',
      text: '운영체제는 하드웨어와 사용자 사이를 중개한다.',
      citation: '컴퓨터공학개론 > 2 주차 - 프로세스와 스레드 > 0:00~2:00',
      session_title: '2주차 - 프로세스와 스레드',
      full_transcript: '자, 지난 시간에는 하드웨어에 대해 배웠죠.\n오늘은 소프트웨어를 배워봅시다. 특히 운영체제에 집중할 건데요.\n운영체제는 하드웨어와 사용자 사이를 중개한다. 이게 가장 중요한 역할입니다.\n마우스 클릭만으로 복잡한 연산이 처리되는게 다 운영체제 덕분이죠.'
    }
  ]
}

const onSendMessage = async (params) => {
  messages.value.push({ role: 'user', content: params.input, attachments: params.attachments })
  emit('sendMessage', params)
  scrollToBottom()

  isGenerating.value = true

  const aiMessage = {
    role: 'assistant',
    content: '',
    references: [],
    isRevealing: true,
    phase: 'streaming',
  }
  messages.value.push(aiMessage)
  scrollToBottom()

  abortController.value = new AbortController()

  try {
    if (USE_DEMO_DATA) {
      await new Promise((resolve) => setTimeout(resolve, 700))

      const current = messages.value[messages.value.length - 1]
      messages.value[messages.value.length - 1] = {
        ...current,
        content: demoResponse.content,
        references: mapCitationsToReferences(demoResponse.citations),
        phase: 'done',
      }
      return
    }

    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: params.input,
        is_thinking: false,
      }),
      signal: abortController.value.signal,
    })

    if (!res.ok) throw new Error(`서버 응답 오류 (상태 코드: ${res.status})`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let streamedText = ''
    let streamedCitations = []

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

        const data = JSON.parse(payload)
        if (data.type === 'citations') {
          streamedCitations = data.citations || []
        } else if (data.type === 'token') {
          streamedText += data.token
        } else if (data.type === 'error') {
          streamedText += `\n오류: ${data.error}`
        }

        const current = messages.value[messages.value.length - 1]
        messages.value[messages.value.length - 1] = {
          ...current,
          content: streamedText,
          references: mapCitationsToReferences(streamedCitations),
          phase: 'streaming',
        }
        scrollToBottom()
      }
    }

    const current = messages.value[messages.value.length - 1]
    messages.value[messages.value.length - 1] = {
      ...current,
      content: streamedText,
      references: mapCitationsToReferences(streamedCitations),
      phase: 'done',
    }
  } catch (error) {
    const aborted = error?.name === 'AbortError'
    const current = messages.value[messages.value.length - 1]
    messages.value[messages.value.length - 1] = {
      ...current,
      content: aborted ? '응답 생성을 중단했습니다.' : '오류가 발생했습니다. 서버 연결을 확인해주세요.',
      references: [],
      phase: 'done',
    }
  } finally {
    isGenerating.value = false
    abortController.value = null
    scrollToBottom()

    const current = messages.value[messages.value.length - 1]
    if (current?.role === 'assistant') {
      setTimeout(() => {
        current.isRevealing = false
      }, 600)
    }
  }
}

const onStopGenerating = () => {
  abortController.value?.abort()
  isGenerating.value = false
}
</script>

<template>
  <div class="w-full h-full relative transition-all duration-700">
    
    <!-- Chat History -->
    <Transition name="fade">
      <div v-if="messages.length > 0" 
           ref="chatScrollRef"
           class="absolute inset-0 overflow-y-auto px-4 w-full flex flex-col items-center custom-scrollbar z-10">
        <div class="w-full max-w-[700px] flex flex-col gap-8 pt-12 pb-[160px]">
          
          <div v-for="(msg, idx) in messages" :key="idx" 
               :class="['flex w-full', msg.role === 'user' ? 'justify-end' : 'justify-start']">

            <div v-if="isAssistantTyping(msg)" class="home-chat-typing-shell">
              <div class="home-typing-dots" role="status" aria-label="답변 생성 중">
                <LoadingHourglass
                  class="chat-loading-dots-lottie"
                  src="/animations/Loading%20Dots%20Blue.json"
                  width="160px"
                  height="90px"
                  :content-scale="4.3"
                  fallback-icon="more_horiz"
                />
              </div>
            </div>

            <div
              v-else-if="msg.role === 'user'"
              class="home-chat-bubble home-chat-bubble-user max-w-[85%] rounded-[18px] px-4 py-2.5 text-[14px] leading-relaxed break-words"
            >
              <div v-if="msg.attachments?.length" class="flex gap-2 mb-3">
                <div v-for="att in msg.attachments" :key="att.url" class="w-14 h-14 rounded-lg overflow-hidden border border-black/10">
                  <img :src="att.url" class="w-full h-full object-cover" />
                </div>
              </div>
              <div class="whitespace-pre-wrap">{{ msg.content }}</div>
            </div>

            <article
              v-else
              class="home-assistant-response"
              :class="{ 'reveal-message': msg.isRevealing }"
            >
              <div
                :class="['home-assistant-markdown', msg.isRevealing ? 'reveal-content' : '']"
                v-html="renderAssistantContent(msg.content)"
              ></div>

              <!-- Reference Links -->
              <div v-if="msg.phase === 'done' && msg.references && msg.references.length > 0" :class="['home-answer-references', msg.isRevealing ? 'reveal-content reveal-delay-2' : '']">
                <button
                  type="button"
                  class="home-reference-toggle"
                  @click="toggleReferenceMessage(idx)"
                >
                  <span class="material-symbols-outlined text-[15px]">link</span>
                  <span>참고한 전사 {{ msg.references.length }}개 보기</span>
                  <span
                    class="material-symbols-outlined home-reference-chevron"
                    :class="{ 'is-open': isReferenceMessageExpanded(idx) }"
                  >
                    expand_more
                  </span>
                </button>

                <div
                  v-if="isReferenceMessageExpanded(idx)"
                  class="home-reference-dropdown flex flex-wrap gap-2 mt-3"
                >
                  <button 
                    v-for="ref in msg.references" 
                    :key="ref.id" 
                    @click="emit('openReference', ref)"
                    class="home-reference-chip"
                  >
                    <span class="material-symbols-outlined text-[14px]">link</span>
                    {{ ref.title }}
                  </button>
                </div>
              </div>
            </article>
          </div>

          <!-- Thinking Dots Animation (Shown before the AI message starts typing) -->
          <Transition name="fade-fast">
            <div v-if="isGenerating && messages.length > 0 && messages[messages.length-1].role === 'user'" 
                 class="flex w-full gap-3 flex-row items-start">
              <div class="home-chat-typing-shell">
                <div class="home-typing-dots" role="status" aria-label="답변 생성 중">
                  <LoadingHourglass
                    class="chat-loading-dots-lottie"
                    src="/animations/Loading%20Dots%20Blue.json"
                    width="160px"
                    height="90px"
                    :content-scale="4.3"
                    fallback-icon="more_horiz"
                  />
                </div>
              </div>
            </div>
          </Transition>

        </div>
      </div>
    </Transition>

    <!-- Layout structure for Title & Input (Flex-grow animation) -->
    <div class="absolute inset-0 flex flex-col items-center pointer-events-none z-20 transition-all duration-700">
      
      <!-- Top dynamic space -->
      <div 
        class="w-full flex-shrink-0 transition-all duration-700" 
        :style="{
          flexGrow: 0,
          height: messages.length > 0 ? '0px' : '190px',
          transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)'
        }"
      ></div>

      <!-- Title (Hides when chat starts) -->
      <Transition name="fade">
        <div v-if="messages.length === 0" class="flex flex-col items-center text-center gap-3 pb-[50px] pointer-events-auto shrink-0 w-full transition-all duration-500">
          <LoadingHourglass
            class="home-hero-animation"
            src="/animations/welcome.json"
            width="clamp(420px, 48vw, 760px)"
            height="clamp(118px, 14vw, 214px)"
            fallback-icon="waving_hand"
          />
        </div>
      </Transition>

      <!-- Input component -->
      <div :class="['home-input-stage', messages.length > 0 ? 'is-chatting' : 'is-idle']">
        <MultimodalInput 
          :is-generating="isGenerating"
          @sendMessage="onSendMessage"
          @stopGenerating="onStopGenerating"
        />
      </div>

      <!-- Recent Files Section -->
      <div v-if="messages.length === 0 && props.recentFiles.length" class="home-recent-files animate-fade-in-up" style="animation-duration: 0.6s; animation-delay: 0.2s; animation-fill-mode: both;">
        <h3>최근 연 파일</h3>
        <div class="home-recent-file-grid">
          <button
            v-for="file in props.recentFiles"
            :key="file.id"
            type="button"
            class="home-recent-file-card"
            @click="emit('openRecentFile', file)"
          >
            <div
              class="home-recent-file-paper"
              :style="{
                background: colorWithAlpha(getRecentFileColor(file), 0.14),
                '--recent-file-color': getRecentFileColor(file)
              }"
            >
              <div class="home-recent-file-lines"></div>
              <div class="home-recent-file-content">
                <div class="home-recent-file-meta">
                  <div class="home-recent-file-icon">
                    <span class="material-symbols-outlined">
                      {{ getRecentFileIcon(file) }}
                    </span>
                  </div>
                  <span
                    class="home-recent-file-tag"
                    :style="{ color: getRecentFileColor(file) }"
                  >
                    {{ getRecentFileTag(file) }}
                  </span>
                </div>
                <div class="home-recent-file-bottom">
                  <strong>{{ file.name }}</strong>
                  <span>{{ file.date }}</span>
                </div>
              </div>
            </div>
          </button>
        </div>
      </div>

      <!-- Bottom dynamic space -->
      <div 
        class="w-full flex-shrink-0 transition-all duration-700"
        :style="{ 
           flexGrow: 0, 
           height: messages.length > 0 ? '32px' : '0px',
           transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)' 
        }"
      ></div>

    </div>
    
  </div>
</template>

<style scoped>
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes thinking-dot {
  0%, 100% { transform: translateY(0); opacity: 0.5; }
  50% { transform: translateY(-4px); opacity: 1; }
}

@keyframes pulse-slow {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.8; transform: scale(1.05); }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.animate-spin-slow {
  animation: spin-slow 3s linear infinite;
}

.animate-thinking-dot {
  animation: thinking-dot 1.2s ease-in-out infinite;
}

.animate-pulse-slow {
  animation: pulse-slow 2s ease-in-out infinite;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.4s ease, transform 0.4s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.98);
}

.fade-fast-enter-active,
.fade-fast-leave-active {
  transition: opacity 0.2s ease;
}
.fade-fast-enter-from,
.fade-fast-leave-to {
  opacity: 0;
}

.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 10px;
}

.home-input-stage {
  pointer-events: auto;
  z-index: 50;
  transition:
    width 1.15s cubic-bezier(0.19, 1, 0.22, 1),
    transform 1.15s cubic-bezier(0.19, 1, 0.22, 1),
    bottom 1.15s cubic-bezier(0.19, 1, 0.22, 1),
    opacity 0.55s ease;
}

.home-input-stage.is-idle {
  width: 100%;
  max-width: 820px;
  flex-shrink: 0;
}

.home-input-stage.is-chatting {
  position: absolute;
  left: 50%;
  bottom: 42px;
  width: min(820px, calc(100% - 48px));
  transform: translateX(-50%);
  animation: homeInputSettle 1.15s cubic-bezier(0.19, 1, 0.22, 1) both;
}

@keyframes homeInputSettle {
  from {
    opacity: 0.92;
    transform: translate(-50%, -42px) scale(0.985);
  }
  to {
    opacity: 1;
    transform: translate(-50%, 0) scale(1);
  }
}

/* ── AI 메시지 등장 애니메이션 ── */
@keyframes revealMessage {
  from {
    opacity: 0;
    transform: translateY(12px) scale(0.97);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@keyframes revealContent {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.reveal-message {
  animation: revealMessage 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.reveal-content {
  animation: revealContent 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.reveal-delay-1 {
  opacity: 0;
  animation-delay: 0.15s;
}

.reveal-delay-2 {
  opacity: 0;
  animation-delay: 0.3s;
}

.home-chat-bubble {
  border: 1px solid var(--copy-line);
  box-shadow: 0 18px 40px rgba(24, 28, 35, 0.06);
}

.home-chat-bubble-assistant {
  background: var(--copy-surface);
}

.home-chat-bubble-user {
  color: #fff;
  background: linear-gradient(160deg, rgba(55, 53, 73, 0.94), rgba(34, 42, 68, 0.9));
  border-color: transparent;
  border-bottom-right-radius: 4px;
  box-shadow: none;
}

.home-assistant-response {
  width: 100%;
  margin-left: 0;
  padding-top: 18px;
  color: var(--copy-text);
}

.home-assistant-markdown {
  font-size: 16px;
  font-weight: 500;
  line-height: 1.78;
  letter-spacing: 0;
  word-break: keep-all;
}

.home-assistant-markdown :deep(h1),
.home-assistant-markdown :deep(h2),
.home-assistant-markdown :deep(h3) {
  margin: 0 0 18px;
  color: var(--copy-text);
  font-weight: 900;
  line-height: 1.22;
  letter-spacing: 0;
}

.home-assistant-markdown :deep(h1) {
  font-size: 26px;
}

.home-assistant-markdown :deep(h2) {
  margin-top: 32px;
  font-size: 23px;
}

.home-assistant-markdown :deep(h3) {
  margin-top: 26px;
  font-size: 20px;
}

.home-assistant-markdown :deep(p) {
  margin: 0 0 18px;
}

.home-assistant-markdown :deep(strong) {
  font-weight: 900;
}

.home-assistant-markdown :deep(ul),
.home-assistant-markdown :deep(ol) {
  margin: 8px 0 24px 24px;
  padding: 0;
}

.home-assistant-markdown :deep(li) {
  margin: 8px 0;
  padding-left: 6px;
}

.home-answer-references {
  margin-top: 28px;
  padding-top: 18px;
  border-top: 1px solid rgba(24, 28, 35, 0.08);
}

.home-chat-typing-shell {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  max-width: 85%;
  min-height: 34px;
  margin-left: 0;
  padding: 22px 4px 6px;
}

.home-typing-dots {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 55px;
  height: 24px;
  overflow: visible;
}

.home-typing-dots :deep(.chat-loading-dots-lottie) {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}

.home-hero-title {
  color: var(--copy-text);
  font-size: 38px;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.15;
}

.home-hero-animation {
  display: block;
  user-select: none;
  pointer-events: none;
}

.home-recent-files {
  width: 100%;
  max-width: 700px;
  pointer-events: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 48px;
  flex-shrink: 0;
}

.home-recent-files h3 {
  padding: 0 2px;
  color: var(--copy-muted-strong);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.02em;
}

.home-recent-file-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.home-recent-file-card {
  min-width: 0;
  height: 118px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: transform 0.2s ease, filter 0.2s ease;
}

.home-recent-file-card:hover {
  transform: translateY(-3px);
  filter: brightness(1.02);
}

.home-recent-file-paper {
  position: relative;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 18px;
  box-shadow: 0 18px 42px rgba(24, 28, 35, 0.05);
}

.home-recent-file-lines {
  display: none;
}

.home-recent-file-content {
  position: relative;
  z-index: 1;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 14px;
}

.home-recent-file-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.home-recent-file-icon {
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.72);
  color: var(--copy-text);
}

.home-recent-file-icon .material-symbols-outlined {
  font-size: 18px;
  font-variation-settings: 'FILL' 0;
}

.home-recent-file-tag {
  height: 18px;
  display: inline-flex;
  align-items: center;
  max-width: 112px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.64);
  font-size: 10px;
  font-weight: 950;
  letter-spacing: 0.04em;
}

.home-recent-file-bottom {
  margin-top: auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.home-recent-file-bottom strong {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #1d1d1f;
  font-size: 13px;
  font-weight: 850;
  line-height: 1.25;
}

.home-recent-file-bottom span {
  margin-top: 4px;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 650;
  line-height: 1.25;
}

.home-reference-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  color: #4b6a4e;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid #dce8d3;
  background: #eef4e8;
  border-radius: 999px;
  padding: 5px 12px;
  line-height: 1.4;
  white-space: nowrap;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.home-reference-toggle:hover {
  background: #dce8d3;
  border-color: #b8cfae;
  box-shadow: 0 1px 4px rgba(72, 101, 74, 0.15);
}

.home-reference-chevron {
  font-size: 17px;
  transition: transform 0.18s ease;
}

.home-reference-chevron.is-open {
  transform: rotate(180deg);
}

.home-reference-dropdown {
  animation: revealContent 0.22s ease forwards;
}

.home-reference-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-width: 0;
  max-width: 100%;
  padding: 5px 12px;
  color: #4b6a4e;
  background: #eef4e8;
  border: 1px solid #dce8d3;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.home-reference-chip:hover {
  background: #dce8d3;
  border-color: #b8cfae;
  box-shadow: 0 1px 4px rgba(72, 101, 74, 0.15);
}

.home-reference-chip .material-symbols-outlined {
  color: #4b6a4e;
  flex: 0 0 auto;
}

</style>
