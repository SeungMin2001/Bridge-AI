<!-- 워크스페이스 내에서 AI와 실시간으로 채팅하며 노트를 정리할 수 있는 오른쪽 채팅 패널입니다. -->
<script setup>
import { ref, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useChat } from '../../composables/useChat'
import { marked } from 'marked'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  visible: { type: Boolean, default: true },
  aiInput: { type: String, default: '' },
  activeFileId: { type: String, default: '' }
})

const emit = defineEmits(['update:aiInput'])

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
const expandedCitationMessages = ref(new Set())

// 🚀 [환경 설정] 백엔드 연동 모드 전환 플래그
// true: 백엔드 연결 없이 지정된 한국어 데모 데이터로 즉시 응답합니다.
// false: 실제 백엔드 서버(http://100.104.164.84:8000)로 통신합니다.
// 백엔드 사용시 여부분 주석 처리 조심
const USE_DEMO_DATA = false

function getActiveSessionId() {
  const id = props.activeFileId || ''
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id) ? id : null
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

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        is_thinking: isThinkingMode.value,
        session_id: getActiveSessionId()
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

  return citations.filter((cite) => {
    if (!cite?.citation || seen.has(cite.citation)) return false
    seen.add(cite.citation)
    return true
  })
}

function toggleCitationMessage(index) {
  const next = new Set(expandedCitationMessages.value)
  if (next.has(index)) next.delete(index)
  else next.add(index)
  expandedCitationMessages.value = next
}

function isCitationMessageExpanded(index) {
  return expandedCitationMessages.value.has(index)
}

function handleCitationClick(event, cite) {
  if (!cite) return
  
  // 🎯 중앙 메인 컨텐츠 카드의 위치를 찾습니다.
  const mainCard = document.getElementById('tab-contents-container')
  if (!mainCard) {
    // 만약 요소를 못 찾는 경우 대비한 fallback
    openCitePopover(cite, window.innerWidth / 2 + 50, 100)
    return
  }

  const rect = mainCard.getBoundingClientRect()
  
  const popoverWidth = 340
  const edgeInset = 0
  const topInset = 0
  openCitePopover(cite, rect.right - popoverWidth - edgeInset, rect.top + topInset)
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
              :class="msg.role === 'ai' ? 'w-full flex flex-col' : 'user-bubble px-4 py-2.5 rounded-[18px] text-white text-[14px] leading-relaxed self-end w-fit max-w-[85%]'"
            >
              <!-- 사용자 말풍선 -->
              <template v-if="msg.role === 'user'">
                {{ msg.text }}
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

                <div v-if="msg.phase === 'done' && getSourceChips(msg).length" class="answer-source-summary">
                  <button
                    class="answer-source-toggle"
                    type="button"
                    @click="toggleCitationMessage(i)"
                  >
                    <span class="material-symbols-outlined text-[15px]">link</span>
                    <span>참고한 전사 {{ getSourceChips(msg).length }}개 보기</span>
                    <span
                      class="material-symbols-outlined answer-source-chevron"
                      :class="{ 'is-open': isCitationMessageExpanded(i) }"
                    >
                      expand_more
                    </span>
                  </button>
                </div>

                <div v-if="msg.phase === 'done' && getSourceChips(msg).length && isCitationMessageExpanded(i)" class="answer-source-panel">
                  <div class="answer-source-panel-head">
                    <span class="material-symbols-outlined">format_quote</span>
                    <span>근거 전사</span>
                  </div>
                  <div class="answer-source-list">
                    <button
                      v-for="cite in getSourceChips(msg)"
                      :key="cite.citation"
                      type="button"
                      class="answer-source-card"
                      @click="handleCitationClick($event, cite)"
                      :title="cite.citation"
                    >
                      <span class="answer-source-card-index"></span>
                      <span class="answer-source-card-main">
                        <span class="answer-source-card-text">{{ cite.text }}</span>
                        <span class="answer-source-card-meta">
                          <span class="material-symbols-outlined">link</span>
                          <span>{{ cite.citation }}</span>
                        </span>
                      </span>
                      <span class="material-symbols-outlined answer-source-card-arrow">open_in_new</span>
                    </button>
                  </div>
                </div>

                <!-- thinking 중 데이터가 오기 전 대기 -->
                <div v-if="msg.phase === 'thinking' && !msg.text && !msg.thinking" class="flex items-center gap-2 mt-1">
                  <span class="material-symbols-outlined text-[15px] thinking-spin text-[#8e8e93]">psychology</span>
                  <span class="text-[#8e8e93] text-[13px]">생각하는 중...</span>
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

/* ═══════════════════════════════════════
   전역 근거 배지(팝오버 트리거) 스타일
   ═══════════════════════════════════════ */
:deep(.cite-grounding-badge-wrap) {
  margin-top: 8px;
  margin-bottom: 4px;
}

:deep(.cite-chip-inline) {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  border-radius: 100px;
  background: #eef4e8;
  color: #4b6a4e;
  font-size: 12px;
  font-weight: 500;
  vertical-align: middle;
  white-space: nowrap;
  border: 1px solid #dce8d3;
  line-height: 1.4;
  cursor: pointer;
  transition: all 0.2s ease;
}

:deep(.cite-chip-inline:hover) {
  background: #dce8d3;
  border-color: #b8cfae;
  box-shadow: 0 1px 4px rgba(72, 101, 74, 0.15);
}

:deep(.cite-chip-extra) {
  font-size: 10px;
  color: #7a9a7c;
  font-weight: 600;
}

.answer-source-summary {
  margin-top: 14px;
}

.answer-source-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  padding: 5px 12px;
  border-radius: 100px;
  background: #eef4e8;
  color: #4b6a4e;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  border: 1px solid #dce8d3;
  line-height: 1.4;
  cursor: pointer;
  transition: all 0.2s ease;
}

.answer-source-toggle:hover {
  background: #dce8d3;
  border-color: #b8cfae;
  box-shadow: 0 1px 4px rgba(72, 101, 74, 0.15);
}

.answer-source-chevron {
  font-size: 15px;
  transition: transform 0.2s ease;
}

.answer-source-chevron.is-open {
  transform: rotate(180deg);
}

.answer-source-panel {
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid #eef2f7;
}

.answer-source-panel-head {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 10px;
  color: #334155;
  font-size: 13px;
  font-weight: 850;
}

.answer-source-panel-head .material-symbols-outlined {
  color: #64748b;
  font-size: 17px;
}

.answer-source-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.answer-source-card {
  width: 100%;
  display: grid;
  grid-template-columns: 6px minmax(0, 1fr) 18px;
  gap: 12px;
  padding: 13px 14px;
  text-align: left;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  cursor: pointer;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.answer-source-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
  transform: translateY(-1px);
}

.answer-source-card-index {
  width: 6px;
  min-height: 100%;
  border-radius: 999px;
  background: #dbeafe;
}

.answer-source-card-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 9px;
}

.answer-source-card-text {
  color: #334155;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.65;
  word-break: keep-all;
}

.answer-source-card-meta {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-width: 0;
  width: fit-content;
  max-width: 100%;
  padding: 5px 12px;
  color: #4b6a4e;
  background: #eef4e8;
  border: 1px solid #dce8d3;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.4;
}

.answer-source-card-meta .material-symbols-outlined {
  font-size: 15px;
  color: #4b6a4e;
  flex: 0 0 auto;
}

.answer-source-card-meta span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.answer-source-card-arrow {
  align-self: center;
  color: #94a3b8;
  font-size: 17px;
}

:deep(.ai-doc-feed p) {
  margin-bottom: 0.5em;
}
:deep(.ai-doc-feed p:last-child) {
  margin-bottom: 0;
}

</style>
