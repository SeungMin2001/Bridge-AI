<!-- 워크스페이스 내에서 AI와 실시간으로 채팅하며 노트를 정리할 수 있는 오른쪽 채팅 패널입니다. -->
<script setup>
import { ref, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import LoadingHourglass from '../ui/LoadingHourglass.vue'
import CitationInlineText from './citations/CitationInlineText.vue'

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
const workspaceChatbotAnimationRef = ref(null)
let workspaceChatbotTimer = null

async function sendMessage() {
  const question = props.aiInput.trim()
  if (!question || isSending.value) return // 전송 중이거나 빈 메시지면 무시
  
  isSending.value = true

  addMessage({ role: 'user', text: question })
  emit('update:aiInput', '')
  isLoading.value = true

  messages.value.push({ role: 'ai', text: '', thinking: '', citations: [], phase: 'streaming' })

  // 2. 실제 백엔드 서버 연동 모드 (SSE 스트리밍)
  const t0 = performance.now()
  let ttftLogged = false
  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        is_thinking: isThinkingMode.value,
        // AI 질문은 특정 파일/폴더 선택과 무관하게 전체 전사문과 PDF 자료를 검색한다.
        // 선택 파일 범위는 요약/퀴즈 기능에서만 사용한다.
        session_id: null,
        source_filter: null
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

function clampPosition(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function handleCitationClick({ cite, target }) {
  if (!cite) return
  openCitationPopover(cite, target)
}

function openCitationPopover(cite, target) {
  const rect = target?.getBoundingClientRect?.()
  if (!rect) {
    openCitePopover(cite, Math.max(16, window.innerWidth - 380), 96)
    return
  }

  const popoverWidth = 390
  const estimatedPopoverHeight = Math.min(620, window.innerHeight - 32)
  const viewportInset = 16
  const maxLeft = Math.max(viewportInset, window.innerWidth - popoverWidth - viewportInset)
  const left = clampPosition(rect.left, viewportInset, maxLeft)

  let top = rect.bottom + 10
  if (top + estimatedPopoverHeight > window.innerHeight - viewportInset) {
    top = Math.max(viewportInset, rect.top - estimatedPopoverHeight - 10)
  }

  openCitePopover(cite, left, top)
}

function shouldShowCitations(msg = {}) {
  return msg.phase === 'done'
    && !isErrorAnswer(msg.text)
    && Array.isArray(msg.citations)
    && msg.citations.length > 0
}

function isErrorAnswer(text = '') {
  const value = String(text || '').trim()
  return value.startsWith('오류:')
    || value.startsWith('⚠️')
    || value.includes('All connection attempts failed')
}

const width = ref(360)
const isResizing = ref(false)

const stopWorkspaceChatbotTimer = () => {
  if (!workspaceChatbotTimer) return
  window.clearInterval(workspaceChatbotTimer)
  workspaceChatbotTimer = null
}

const startWorkspaceChatbotTimer = async () => {
  stopWorkspaceChatbotTimer()
  if (!props.visible || messages.value.length > 0) return

  await nextTick()
  workspaceChatbotTimer = window.setInterval(() => {
    workspaceChatbotAnimationRef.value?.playFromStart?.()
  }, 15000)
}

const handleMouseDown = (e) => {
  isResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
}

const handleMouseMove = (e) => {
  if (!isResizing.value) return
  const newWidth = window.innerWidth - e.clientX - 12
  const maxWidth = Math.max(280, Math.min(600, window.innerWidth - 24))
  const minWidth = Math.min(280, maxWidth)
  if (newWidth > minWidth && newWidth < maxWidth) width.value = newWidth
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
  startWorkspaceChatbotTimer()
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  stopWorkspaceChatbotTimer()
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

watch(
  () => [messages.value.length, props.visible],
  () => {
    startWorkspaceChatbotTimer()
  }
)

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
    class="workspace-right-sidebar h-full shrink-0 overflow-hidden rounded-[24px] transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)]"
    id="right-sidebar"
    :class="{ 'sidebar-collapsed': !visible }"
    :style="{ '--right-sidebar-width': visible ? `${width}px` : '0px' }"
  >
    <div class="card workspace-right-sidebar-card h-full flex flex-col p-4 pt-3.5 relative min-w-0">
      <transition name="fade-slide-switch" mode="out-in">
        <div v-if="messages.length === 0" key="initial-ui" class="flex-1 flex flex-col items-center justify-center px-2">
          <div class="mb-6 flex items-center justify-center">
            <LoadingHourglass
              ref="workspaceChatbotAnimationRef"
              class="workspace-chatbot-animation"
              src="/animations/Chatbot.json"
              width="112px"
              height="112px"
              :autoplay="false"
              :loop="false"
              fallback-icon="smart_toy"
            />
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
                <CitationInlineText
                  v-if="msg.text" 
                  :class="{ 'answer-fade-in': msg.phase === 'answering' || msg.phase === 'done' }"
                  class="ai-doc-feed text-[#1d1d1f] text-[14px] leading-[1.7]"
                  :text="msg.text"
                  :citations="msg.citations"
                  :enable-citations="shouldShowCitations(msg)"
                  @citationClick="handleCitationClick"
                />

                <div
                  v-if="(msg.phase === 'streaming' || msg.phase === 'thinking') && !msg.text && !msg.thinking"
                  class="ai-stream-wait"
                  role="status"
                  aria-live="polite"
                >
                  <span class="ai-typing-dots" aria-hidden="true">
                    <LoadingHourglass
                      class="chat-loading-dots-lottie"
                      src="/animations/Loading%20Dots%20Blue.json"
                      width="160px"
                      height="90px"
                      :content-scale="4.3"
                      fallback-icon="more_horiz"
                    />
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
.workspace-right-sidebar {
  width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
  min-width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
  max-width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
}

.workspace-right-sidebar-card {
  background: #fff;
  border: 1px solid rgba(226, 232, 240, 0.78);
  box-shadow: -10px 0 34px rgba(48, 42, 58, 0.05);
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}

.workspace-right-sidebar-card::before {
  opacity: 0;
}

.workspace-right-sidebar-card::after {
  border-color: rgba(226, 232, 240, 0.78);
}

.chat-input-glow {
  background: #f8fafc;
  border: 1.5px solid #d6dee9;
  box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}

.chat-input-glow:focus-within {
  background: #fff;
  border-color: #94a3b8;
  box-shadow: 0 20px 38px rgba(15, 23, 42, 0.11);
  transform: translateY(-1px);
}

.workspace-chatbot-animation {
  display: block;
  user-select: none;
  pointer-events: none;
}

@media (max-width: 1280px) {
  .workspace-right-sidebar {
    width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
    min-width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
    max-width: min(var(--right-sidebar-width, 360px), calc(100vw - 24px));
  }
}

@media (max-width: 1024px) {
  #resizer-right {
    display: none !important;
  }

  .workspace-right-sidebar {
    position: fixed;
    top: 12px;
    right: 12px;
    bottom: 12px;
    z-index: 120;
    width: min(420px, calc(100vw - 24px)) !important;
    min-width: min(420px, calc(100vw - 24px)) !important;
    max-width: min(420px, calc(100vw - 24px)) !important;
    height: auto !important;
    box-shadow: -18px 0 48px rgba(15, 23, 42, 0.16);
  }

  .workspace-right-sidebar.sidebar-collapsed {
    width: min(420px, calc(100vw - 24px)) !important;
    min-width: min(420px, calc(100vw - 24px)) !important;
    max-width: min(420px, calc(100vw - 24px)) !important;
    margin: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    transform: translateX(calc(100% + 24px));
  }
}

@media (max-width: 560px) {
  .workspace-right-sidebar {
    top: 8px;
    right: 8px;
    bottom: 8px;
    width: calc(100vw - 16px) !important;
    min-width: calc(100vw - 16px) !important;
    max-width: calc(100vw - 16px) !important;
    border-radius: 20px;
  }

  .workspace-right-sidebar-card {
    padding: 14px;
  }
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
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 55px;
  height: 24px;
  overflow: visible;
}

.ai-typing-dots :deep(.chat-loading-dots-lottie) {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}

</style>
