<!-- 홈 화면 상단에 표시되는 환영 문구와 광고/안내 배너를 포함하는 컴포넌트입니다. -->
<script setup>
import { ref, defineEmits, nextTick } from 'vue'
import MultimodalInput from './MultimodalInput.vue'

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

const cleanAssistantContent = (content = '') => {
  return String(content)
    .split('\n')
    .filter((line) => !/^\s*(출처|참고\s*출처)\s*[:：]/i.test(line.trim()))
    .join('\n')
    .replace(/\s*\[출처[:：]?[^\]]*\]\s*$/i, '')
    .trim()
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
        <div class="w-full max-w-[700px] flex flex-col gap-6 pt-12 pb-[160px]">
          
          <div v-for="(msg, idx) in messages" :key="idx" 
               :class="['flex w-full gap-3', msg.role === 'user' ? 'flex-row-reverse' : 'flex-row items-start']">
            
            <!-- AI Avatar (Animating when it's the latest message being generated) -->
            <div v-if="msg.role === 'assistant'" class="flex-shrink-0 mt-1">
              <div :class="[
                'w-8 h-8 rounded-full bg-[#4f46e5] flex items-center justify-center border border-indigo-200/70 overflow-hidden',
                isGenerating && idx === messages.length - 1 ? 'ring-2 ring-indigo-400/30' : ''
              ]">
                <span :class="[
                  'material-symbols-outlined text-[18px] text-white',
                  isGenerating && idx === messages.length - 1 ? 'animate-spin-slow' : ''
                ]" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
              </div>
            </div>

            <!-- Message Bubble -->
            <div :class="[
              'home-chat-bubble max-w-[85%] rounded-[24px] px-5 py-4 text-[15px] leading-relaxed break-words',
              msg.role === 'user' 
                ? 'home-chat-bubble-user text-white rounded-tr-none' 
                : 'home-chat-bubble-assistant text-[#1e293b] rounded-tl-none',
              msg.role === 'assistant' && msg.isRevealing ? 'reveal-message' : ''
            ]">
              <div v-if="msg.attachments?.length" class="flex gap-2 mb-3">
                <div v-for="att in msg.attachments" :key="att.url" class="w-14 h-14 rounded-lg overflow-hidden border border-black/10">
                  <img :src="att.url" class="w-full h-full object-cover" />
                </div>
              </div>
              <div :class="['whitespace-pre-wrap', msg.isRevealing ? 'reveal-content' : '']">{{ msg.role === 'assistant' ? cleanAssistantContent(msg.content) : msg.content }}</div>
              
              <!-- Reference Links -->
              <div v-if="msg.phase === 'done' && msg.references && msg.references.length > 0" :class="['mt-4 pt-4 border-t border-black/10', msg.isRevealing ? 'reveal-content reveal-delay-2' : '']">
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
                    class="flex items-center gap-1 text-[13px] bg-indigo-50/50 hover:bg-indigo-100 text-indigo-700 px-3 py-1.5 rounded-full border border-indigo-200/50 transition-colors shadow-sm font-medium"
                  >
                    <span class="material-symbols-outlined text-[14px]">link</span>
                    {{ ref.title }}
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Thinking Dots Animation (Shown before the AI message starts typing) -->
          <Transition name="fade-fast">
            <div v-if="isGenerating && messages.length > 0 && messages[messages.length-1].role === 'user'" 
                 class="flex w-full gap-3 flex-row items-start">
              <div class="flex-shrink-0 mt-1">
                <div class="w-8 h-8 rounded-full bg-[#4f46e5] flex items-center justify-center border border-indigo-200/70 ring-2 ring-indigo-400/30 overflow-hidden animate-pulse-slow">
                  <span class="material-symbols-outlined text-[18px] text-white animate-spin-slow" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
                </div>
              </div>
              <div class="bg-white/80 backdrop-blur-xl border border-slate-200 rounded-2xl rounded-tl-none px-6 py-4 flex items-center gap-1.5">
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-400 rounded-full animate-thinking-dot"></div>
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-500 rounded-full animate-thinking-dot [animation-delay:0.2s]"></div>
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-600 rounded-full animate-thinking-dot [animation-delay:0.4s]"></div>
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
        :style="{ flexGrow: 1, transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)' }"
      ></div>

      <!-- Title (Hides when chat starts) -->
      <Transition name="fade">
        <div v-if="messages.length === 0" class="flex flex-col items-center text-center gap-3 pb-8 pointer-events-auto shrink-0 w-full transition-all duration-500">
          <div class="flex items-center justify-center transform -rotate-6 transition-transform hover:rotate-0 duration-500">
            <img src="/images/banner_illust.png" alt="AI chat" class="w-36 h-auto object-contain" />
          </div>
          <div class="text-[32px] font-extrabold text-[#1d1d1f] tracking-tight leading-tight">무엇을 도와드릴까요?</div>
        </div>
      </Transition>

      <!-- Input component -->
      <div class="w-full max-w-[730px] pointer-events-auto flex-shrink-0 z-50 transition-all duration-700">
        <MultimodalInput 
          :is-generating="isGenerating"
          @sendMessage="onSendMessage"
          @stopGenerating="onStopGenerating"
        />
      </div>

      <!-- Recent Files Section -->
      <div v-if="messages.length === 0 && props.recentFiles.length" class="w-full max-w-[600px] pointer-events-auto flex flex-col gap-4 mt-12 opacity-80 animate-fade-in-up shrink-0" style="animation-duration: 0.6s; animation-delay: 0.2s; animation-fill-mode: both;">
        <h3 class="text-sm font-semibold text-[#64748b] px-2 uppercase tracking-wider font-sans">최근 연 파일</h3>
        <div class="flex gap-4">
          <button
            v-for="file in props.recentFiles"
            :key="file.id"
            type="button"
            class="flex-1 neo-card p-5 flex flex-col gap-3 cursor-pointer hover:-translate-y-1 hover:brightness-105 transition-all duration-300 text-left"
            @click="emit('openRecentFile', file)"
          >
            <div class="w-10 h-10 rounded-xl neo-inner flex items-center justify-center text-gray-500">
              <span class="material-symbols-outlined text-[20px]">{{ getFileIcon(file.type) }}</span>
            </div>
            <div class="flex flex-col">
              <span class="text-[14px] font-bold text-[#1e293b] truncate leading-tight">{{ file.name }}</span>
              <span class="text-[12px] text-[#64748b] mt-0.5">{{ file.date }}</span>
            </div>
          </button>
        </div>
      </div>

      <!-- Bottom dynamic space -->
      <div 
        class="w-full flex-shrink-0 transition-all duration-700"
        :style="{ 
           flexGrow: messages.length > 0 ? 0 : 1.2, 
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
  border: 1px solid rgba(226, 232, 240, 0.92);
  box-shadow: none;
}

.home-chat-bubble-assistant {
  background: rgba(248, 250, 252, 0.92);
}

.home-chat-bubble-user {
  background: #1f2937;
  border-color: #1f2937;
}

.home-reference-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #475569;
  font-size: 13px;
  font-weight: 800;
  border: 1px solid rgba(226, 232, 240, 0.92);
  background: rgba(255, 255, 255, 0.62);
  border-radius: 999px;
  padding: 7px 11px;
  transition: background 0.18s ease, border-color 0.18s ease;
}

.home-reference-toggle:hover {
  background: rgba(248, 250, 252, 0.95);
  border-color: rgba(199, 210, 254, 0.9);
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

</style>
