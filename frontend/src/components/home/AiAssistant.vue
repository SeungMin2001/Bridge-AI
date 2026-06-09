<!-- 사용자와 대화하며 질문에 답변하거나 요약 서비스를 제공하는 AI 비서 팝업 컴포넌트입니다. -->
<script setup>
import { ref } from 'vue'
import { useChat } from '../../composables/useChat'

defineProps({
  isOpen: Boolean,
  aiWinRef: Object,
  aiBtnRef: Object
})

const emit = defineEmits(['update:isOpen'])

const inputText = ref('')
const isLoading = ref(false)
const { messages, addMessage, updateLastAiMessage } = useChat()

if (!messages.value.length) {
  addMessage({
    role: 'ai',
    text: '안녕하세요! 어떤 것을 도와드릴까요? 강의 노트 요약이나 시험 문제 생성 등을 도와드릴 수 있습니다.',
    thinking: '',
    citations: [],
    phase: 'done'
  })
}

async function sendMessage() {
  const question = inputText.value.trim()
  if (!question) return

  addMessage({ role: 'user', text: question })
  inputText.value = ''
  isLoading.value = true

  // 로딩 표시용 임시 버블
  addMessage({ role: 'ai', text: '', thinking: '', citations: [], phase: 'thinking' })

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
    const data = await res.json()
    updateLastAiMessage({
      role: 'ai',
      thinking: data.thinking || '',
      text: data.answer || '',
      citations: data.citations || [],
      phase: 'done'
    })
  } catch (e) {
    console.error('[AI Chat] fetch error:', e)
    updateLastAiMessage({
      role: 'ai',
      thinking: '',
      text: '오류가 발생했습니다. 서버 연결을 확인해주세요.',
      citations: [],
      phase: 'done'
    })
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <button
    class="absolute bottom-8 right-8 w-[64px] h-[64px] bg-gradient-to-r from-[#3b82f6] to-[#5856d6] rounded-full shadow-[0_8px_24px_rgba(59,130,246,0.3)] flex items-center justify-center text-white hover:scale-105 active:scale-95 transition-all z-40"
    id="home-ai-btn"
    @click="emit('update:isOpen', !isOpen)"
  >
    <span class="material-symbols-outlined text-[28px]">auto_awesome</span>
  </button>

  <div :class="['home-ai-chat-window z-[60]', { open: isOpen }]">
    <div class="chat-header">
      <div class="flex items-center gap-2.5">
        <div class="w-8 h-8 bg-[#373549] rounded-lg flex items-center justify-center">
          <span class="material-symbols-outlined text-white text-[18px]">auto_awesome</span>
        </div>
        <span class="text-[16px] font-bold tracking-[-0.01em]">Lecto AI Assistant</span>
      </div>
      <button class="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]" @click="emit('update:isOpen', false)">
        <span class="material-symbols-outlined text-[20px]">close</span>
      </button>
    </div>

    <div class="chat-content custom-scrollbar">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="[
          'chat-bubble',
          msg.role === 'ai' ? 'bubble-ai' : 'bubble-user',
          msg.phase === 'thinking' && !msg.text && !msg.thinking ? 'bubble-typing' : ''
        ]"
      >
        <!-- Thinking 실시간 표시 -->
        <div v-if="msg.thinking" class="thinking-block mb-2">
          <div class="flex items-center gap-1 mb-1">
            <span class="material-symbols-outlined text-[14px] text-[#8e8e93]" :class="{ 'thinking-spin': msg.phase === 'thinking' }">psychology</span>
            <span class="text-[11px] font-semibold text-[#8e8e93]">
              {{ msg.phase === 'thinking' ? '생각하는 중...' : '사고 완료' }}
            </span>
          </div>
          <div class="thinking-content thinking-stream">
            {{ msg.thinking }}
          </div>
        </div>
        <!-- 최종 답변 -->
        <div v-if="msg.text" class="answer-text" :class="{ 'answer-fade-in': msg.phase === 'answering' || msg.phase === 'done' }">
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
        <!-- 아직 thinking 중이고 답변 없을 때 -->
        <div v-if="msg.phase === 'thinking' && !msg.text && !msg.thinking" class="thinking-loading">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <div class="chat-footer">
      <div class="chat-input-container">
        <input
          class="chat-input"
          placeholder="AI에게 질문해보세요..."
          type="text"
          v-model="inputText"
          @keyup.enter="sendMessage"
          :disabled="isLoading"
        />
        <button class="btn-ghost-icon p-1 text-[#373549]" @click="sendMessage" :disabled="isLoading">
          <span class="material-symbols-outlined text-[20px]">send</span>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-bubble.bubble-typing {
  padding: 6px 4px;
  background: transparent;
  border: 0;
  box-shadow: none;
}

.thinking-loading {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 24px;
}

.thinking-loading span {
  width: 13px;
  height: 13px;
  border-radius: 999px;
  background: #aaa7a3;
  animation: home-popup-typing-dot 1.05s ease-in-out infinite;
}

.thinking-loading span:nth-child(2) {
  animation-delay: 0.16s;
}

.thinking-loading span:nth-child(3) {
  animation-delay: 0.32s;
}

@keyframes home-popup-typing-dot {
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
  .thinking-loading span {
    animation: none;
  }
}
</style>
