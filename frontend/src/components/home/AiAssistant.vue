<script setup>
import { ref } from 'vue'

defineProps({
  isOpen: Boolean,
  aiWinRef: Object,
  aiBtnRef: Object
})

const emit = defineEmits(['update:isOpen'])

const inputText = ref('')
const isLoading = ref(false)
const messages = ref([
  { role: 'ai', text: '안녕하세요! 어떤 것을 도와드릴까요? 강의 노트 요약이나 시험 문제 생성 등을 도와드릴 수 있습니다.' }
])

async function sendMessage() {
  const question = inputText.value.trim()
  console.log('[AI Chat] sendMessage called, question:', question)
  if (!question) return

  messages.value.push({ role: 'user', text: question })
  inputText.value = ''
  isLoading.value = true

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
    const data = await res.json()
    console.log('[AI Chat] response:', data)
    messages.value.push({ role: 'ai', text: data.answer })
  } catch (e) {
    console.error('[AI Chat] fetch error:', e)
    messages.value.push({ role: 'ai', text: '오류가 발생했습니다. 서버 연결을 확인해주세요.' })
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
        :class="['chat-bubble', msg.role === 'ai' ? 'bubble-ai' : 'bubble-user']"
      >
        {{ msg.text }}
      </div>
      <div v-if="isLoading" class="chat-bubble bubble-ai">
        <span>...</span>
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
        />
        <button class="btn-ghost-icon p-1 text-[#373549]" @click="sendMessage">
          <span class="material-symbols-outlined text-[20px]">send</span>
        </button>
      </div>
    </div>
  </div>
</template>
