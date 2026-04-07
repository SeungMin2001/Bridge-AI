<script setup>
import { ref, defineEmits, nextTick } from 'vue'
import MultimodalInput from './MultimodalInput.vue'

const emit = defineEmits(['sendMessage', 'openReference'])

const messages = ref([])
const isGenerating = ref(false)
const chatScrollRef = ref(null)
const abortController = ref(null)

const scrollToBottom = () => {
  nextTick(() => {
    if (chatScrollRef.value) {
      chatScrollRef.value.scrollTop = chatScrollRef.value.scrollHeight
    }
  })
}

const recentFiles = ref([
  { id: '1', name: '자료구조 강의 노트.pdf', type: 'pdf', date: '오늘' },
  { id: '2', name: 'AI 프로젝트 기획서.docx', type: 'doc', date: '어제' },
  { id: '3', name: '중간고사 요약본.pptx', type: 'ppt', date: '2일 전' }
])

const getFileIcon = (type) => {
  switch (type) {
    case 'pdf': return 'picture_as_pdf'
    case 'ppt': return 'slideshow'
    case 'doc': return 'description'
    default: return 'insert_drive_file'
  }
}

const mapCitationsToReferences = (citations = []) => {
  return citations.map((cite, index) => ({
    id: cite.transcript_id || `cite-${index}`,
    title: cite.citation || cite.session_title || `근거 ${index + 1}`,
    script: cite.full_transcript || cite.text || '',
    raw: cite,
  }))
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
                'w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm border border-white/20 overflow-hidden',
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
              'max-w-[85%] rounded-[24px] px-5 py-4 text-[15px] leading-relaxed break-words',
              msg.role === 'user' 
                ? 'neo-active-btn text-white rounded-tr-none' 
                : 'neo-card text-[#1e293b] rounded-tl-none',
              msg.role === 'assistant' && msg.isRevealing ? 'reveal-message' : ''
            ]">
              <div v-if="msg.attachments?.length" class="flex gap-2 mb-3">
                <div v-for="att in msg.attachments" :key="att.url" class="w-14 h-14 rounded-lg overflow-hidden border border-black/10">
                  <img :src="att.url" class="w-full h-full object-cover" />
                </div>
              </div>
              <div :class="['whitespace-pre-wrap', msg.isRevealing ? 'reveal-content' : '']">{{ msg.content }}</div>
              
              <!-- Reference Links -->
              <div v-if="msg.phase === 'done' && msg.references && msg.references.length > 0" :class="['flex flex-wrap gap-2 mt-4 pt-4 border-t border-black/10', msg.isRevealing ? 'reveal-content reveal-delay-2' : '']">
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

          <!-- Thinking Dots Animation (Shown before the AI message starts typing) -->
          <Transition name="fade-fast">
            <div v-if="isGenerating && messages.length > 0 && messages[messages.length-1].role === 'user'" 
                 class="flex w-full gap-3 flex-row items-start">
              <div class="flex-shrink-0 mt-1">
                <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm border border-white/20 ring-2 ring-indigo-400/30 overflow-hidden animate-pulse-slow">
                  <span class="material-symbols-outlined text-[18px] text-white animate-spin-slow" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
                </div>
              </div>
              <div class="bg-white/80 backdrop-blur-xl border border-black/5 rounded-2xl rounded-tl-none px-6 py-4 shadow-sm flex items-center gap-1.5">
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
          <div class="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg transform -rotate-6 transition-transform hover:rotate-0 duration-500">
            <span class="material-symbols-outlined text-[28px] text-white" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
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
      <div v-if="messages.length === 0" class="w-full max-w-[600px] pointer-events-auto flex flex-col gap-4 mt-12 opacity-80 animate-fade-in-up shrink-0" style="animation-duration: 0.6s; animation-delay: 0.2s; animation-fill-mode: both;">
        <h3 class="text-sm font-semibold text-[#64748b] px-2 uppercase tracking-wider font-sans">최근 연 파일</h3>
        <div class="flex gap-4">
          <div v-for="file in recentFiles" :key="file.id" 
               class="flex-1 neo-card p-5 flex flex-col gap-3 cursor-pointer hover:-translate-y-1 hover:brightness-105 transition-all duration-300">
            <div class="w-10 h-10 rounded-xl neo-inner flex items-center justify-center text-gray-500">
              <span class="material-symbols-outlined text-[20px]">{{ getFileIcon(file.type) }}</span>
            </div>
            <div class="flex flex-col">
              <span class="text-[14px] font-bold text-[#1e293b] truncate leading-tight">{{ file.name }}</span>
              <span class="text-[12px] text-[#64748b] mt-0.5">{{ file.date }}</span>
            </div>
          </div>
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
</style>
