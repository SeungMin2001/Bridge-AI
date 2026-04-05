<script setup>
import { ref, nextTick, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: true },
  aiInput: { type: String, default: '' }
})

const emit = defineEmits(['update:aiInput'])

const messages = ref([])
const isLoading = ref(false)

async function sendMessage() {
  const question = props.aiInput.trim()
  if (!question) return

  messages.value.push({ role: 'user', text: question })
  emit('update:aiInput', '')
  isLoading.value = true

  const idx = messages.value.length
  messages.value.push({ role: 'ai', text: '', thinking: '', citations: [], phase: 'streaming' })

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

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
        const msg = messages.value[idx]

        if (data.type === 'citations') {
          messages.value[idx] = { ...msg, citations: data.citations }
        } else if (data.type === 'token') {
          messages.value[idx] = { ...msg, text: msg.text + data.token }
        } else if (data.type === 'error') {
          messages.value[idx] = { ...msg, text: msg.text + `\n오류: ${data.error}` }
        }
      }
    }

    const msg = messages.value[idx]
    messages.value[idx] = { ...msg, phase: 'done' }
  } catch (e) {
    messages.value[messages.value.length - 1] = {
      role: 'ai',
      thinking: '',
      text: '오류가 발생했습니다. 서버 연결을 확인해주세요.',
      citations: [],
      phase: 'done',
    }
  } finally {
    isLoading.value = false
  }
}

const width = ref(420)
const isResizing = ref(false)

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

const handleMouseDown = () => {
  isResizing.value = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.classList.add('is-resizing')
}
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
    class="h-full shrink-0 overflow-hidden transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)]"
    id="right-sidebar"
    :class="{ 'sidebar-collapsed': !visible }"
    :style="{ width: visible ? `${width}px` : '0px', minWidth: visible ? `${width}px` : '0px', maxWidth: visible ? `${width}px` : '0px' }"
  >
    <div class="card h-full flex flex-col p-4 pt-3.5 relative min-w-[300px]">
        <div class="flex-1 flex flex-col items-center justify-center px-2">
          <div class="w-14 h-14 rounded-2xl ai-gradient-bg flex items-center justify-center mb-6 shadow-lg">
            <span class="material-symbols-outlined text-white text-[32px]">auto_awesome</span>
          </div>
          <h3 class="text-[18px] font-bold text-[#1d1d1f] mb-8">무엇을 도와드릴까요?</h3>
          <div class="w-full flex flex-col gap-3 mb-10">
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">description</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">강의 노트 요약하기</span>
            </button>
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left" @click="emit('update:aiInput', '핵심 개념 퀴즈 생성해줘')">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">quiz</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">핵심 개념 퀴즈 생성</span>
            </button>
            <button class="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left">
              <span class="material-symbols-outlined text-[18px] text-[#8e8e93]">translate</span>
              <span class="text-[13px] font-medium text-[#1d1d1f]">외국어 자료 번역</span>
            </button>
          </div>
        </div>

        <div class="mt-auto">
          <div v-if="messages.length > 0" class="mb-3 flex flex-col gap-2 max-h-[300px] overflow-y-auto custom-scrollbar">
            <div
              v-for="(msg, i) in messages"
              :key="i"
              :class="['px-3 py-2 rounded-xl text-[13px] leading-relaxed', msg.role === 'ai' ? 'bg-[#f2f2f7] text-[#1d1d1f] self-start' : 'bg-[#373549] text-white self-end']"
            >
              <!-- Thinking 실시간 표시 -->
              <div v-if="msg.thinking" class="thinking-block mb-1.5">
                <div class="flex items-center gap-1 mb-1">
                  <span class="material-symbols-outlined text-[13px] text-[#8e8e93]" :class="{ 'thinking-spin': msg.phase === 'thinking' }">psychology</span>
                  <span class="text-[10px] font-semibold text-[#8e8e93]">
                    {{ msg.phase === 'thinking' ? '생각하는 중...' : '사고 완료' }}
                  </span>
                </div>
                <div class="thinking-content thinking-stream text-[11px]">
                  {{ msg.thinking }}
                </div>
              </div>
              <!-- 최종 답변 -->
              <div v-if="msg.text" :class="{ 'answer-fade-in': msg.phase === 'answering' || msg.phase === 'done' }">
                {{ msg.text }}
              </div>
              <!-- 출처 표시 -->
              <div v-if="msg.citations && msg.citations.length" class="mt-2 pt-2 border-t border-black/5">
                <div class="flex items-center gap-1 mb-1.5">
                  <span class="material-symbols-outlined text-[12px] text-[#8e8e93]">menu_book</span>
                  <span class="text-[10px] font-bold text-[#8e8e93]">참고 출처</span>
                </div>
                <div v-for="(cite, ci) in msg.citations" :key="ci" class="flex items-start gap-1.5 mb-1">
                  <span class="text-[10px] text-blue-500 font-bold mt-0.5">{{ ci + 1 }}</span>
                  <span class="text-[10px] text-[#636366] leading-[1.5]">{{ cite.citation }}</span>
                </div>
              </div>
              <!-- thinking 중 대기 -->
              <div v-if="msg.phase === 'thinking' && !msg.text && !msg.thinking" class="flex items-center gap-2">
                <span class="material-symbols-outlined text-[14px] thinking-spin">psychology</span>
                <span class="text-[#8e8e93]">생각하는 중...</span>
              </div>
            </div>
          </div>

          <div class="sidebar-search-bg rounded-[14px] px-4 py-3 flex items-center gap-3 border border-transparent focus-within:border-[#3b82f6] transition-all">
            <input
              class="bg-transparent border-none focus:ring-0 p-0 text-[13px] flex-1 text-[#1d1d1f] placeholder-[#aeaeb2]"
              placeholder="AI에게 질문하기..." type="text"
              :value="aiInput"
              @input="emit('update:aiInput', $event.target.value)"
              @keyup.enter="sendMessage"
            />
            <button class="text-[#3b82f6] hover:text-blue-700 transition-colors" @click="sendMessage">
              <span class="material-symbols-outlined text-[20px]">arrow_upward</span>
            </button>
          </div>
          <p class="text-[10px] text-center text-[#aeaeb2] mt-3">AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.</p>
        </div>
      </div>
  </aside>
</template>
