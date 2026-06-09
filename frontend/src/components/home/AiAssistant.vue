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
const {
  messages,
  chatSessionSummaries,
  activeChatSessionId,
  addMessage,
  updateLastAiMessage,
  switchChatSession,
  startNewChat,
  renameChatSession
} = useChat()
const isChatListOpen = ref(false)
const editingChatSessionId = ref('')
const editingChatTitle = ref('')

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

function handleNewChat() {
  if (isLoading.value) return
  startNewChat()
  isChatListOpen.value = false
  inputText.value = ''
}

function handleChatSessionSelect(sessionId) {
  if (isLoading.value) return
  switchChatSession(sessionId)
  isChatListOpen.value = false
  editingChatSessionId.value = ''
}

function startEditingChatTitle(session) {
  editingChatSessionId.value = session.id
  editingChatTitle.value = session.title
}

function commitChatTitleEdit() {
  if (!editingChatSessionId.value) return
  renameChatSession(editingChatSessionId.value, editingChatTitle.value)
  editingChatSessionId.value = ''
  editingChatTitle.value = ''
}

function cancelChatTitleEdit() {
  editingChatSessionId.value = ''
  editingChatTitle.value = ''
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

    <div class="home-chat-session-toolbar">
      <button
        type="button"
        class="home-chat-session-toggle"
        :disabled="isLoading"
        @click="isChatListOpen = !isChatListOpen"
      >
        <span class="material-symbols-outlined text-[15px]">forum</span>
        <span>{{ chatSessionSummaries.find((session) => session.id === activeChatSessionId)?.title || '새 채팅' }}</span>
        <span class="material-symbols-outlined text-[15px] ml-auto">
          {{ isChatListOpen ? 'expand_less' : 'expand_more' }}
        </span>
      </button>
      <button
        type="button"
        class="home-chat-session-new"
        :disabled="isLoading"
        @click="handleNewChat"
      >
        <span class="material-symbols-outlined text-[16px]">add</span>
        새 채팅
      </button>
    </div>
    <div v-if="isChatListOpen" class="home-chat-session-list-panel">
      <button
        type="button"
        class="home-chat-session-list-new"
        :disabled="isLoading"
        @click="handleNewChat"
      >
        <span class="material-symbols-outlined text-[16px]">add_comment</span>
        새로운 채팅 시작
      </button>
      <div class="home-chat-session-list-scroll custom-scrollbar">
        <div
          v-for="session in chatSessionSummaries"
          :key="session.id"
          class="home-chat-session-list-item"
          :class="{ 'is-active': session.id === activeChatSessionId }"
        >
          <button
            v-if="editingChatSessionId !== session.id"
            type="button"
            class="home-chat-session-title-button"
            :disabled="isLoading"
            @click="handleChatSessionSelect(session.id)"
          >
            <span class="material-symbols-outlined text-[15px]">chat_bubble</span>
            <span>{{ session.title }}</span>
          </button>
          <input
            v-else
            v-model="editingChatTitle"
            class="home-chat-session-title-input"
            type="text"
            maxlength="40"
            @keyup.enter="commitChatTitleEdit"
            @keyup.esc="cancelChatTitleEdit"
            @blur="commitChatTitleEdit"
          />
          <button
            type="button"
            class="home-chat-session-edit-button"
            :disabled="isLoading"
            @click="editingChatSessionId === session.id ? commitChatTitleEdit() : startEditingChatTitle(session)"
          >
            <span class="material-symbols-outlined text-[14px]">
              {{ editingChatSessionId === session.id ? 'check' : 'edit' }}
            </span>
          </button>
        </div>
      </div>
    </div>

    <div class="chat-content custom-scrollbar">
      <div v-if="messages.length === 0" class="home-chat-empty">
        <span class="material-symbols-outlined">chat_bubble</span>
        <p>새 채팅을 시작해 보세요.<br />강의 요약, 퀴즈, 개념 질문을 바로 도와드릴게요.</p>
      </div>
      <div
        v-else
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

.home-chat-session-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px 8px;
  border-bottom: 1px solid rgba(242, 242, 247, 0.92);
}

.home-chat-session-toggle {
  flex: 1 1 auto;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-width: 0;
  height: 32px;
  padding: 0 9px;
  border-radius: 999px;
  border: 1px solid #e5e7eb;
  background: #f8fafc;
  color: #1d1d1f;
  font-size: 11px;
  font-weight: 800;
  outline: none;
}

.home-chat-session-toggle span:nth-child(2) {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.home-chat-session-new {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  height: 32px;
  padding: 0 10px;
  border-radius: 999px;
  color: #ffffff;
  background: #373549;
  font-size: 11px;
  font-weight: 900;
}

.home-chat-session-new:disabled,
.home-chat-session-toggle:disabled {
  opacity: 0.56;
  cursor: not-allowed;
}

.home-chat-session-list-panel {
  margin: 0 14px 8px;
  padding: 8px;
  border-radius: 16px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.home-chat-session-list-new {
  width: 100%;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  border-radius: 12px;
  color: #1d1d1f;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  font-size: 11px;
  font-weight: 900;
}

.home-chat-session-list-scroll {
  max-height: 150px;
  overflow-y: auto;
  margin-top: 7px;
}

.home-chat-session-list-item {
  min-height: 36px;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 4px;
  border-radius: 12px;
}

.home-chat-session-list-item.is-active {
  background: #ffffff;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
}

.home-chat-session-title-button {
  flex: 1 1 auto;
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #374151;
  font-size: 11px;
  font-weight: 850;
  text-align: left;
}

.home-chat-session-title-button span:last-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.home-chat-session-title-input {
  flex: 1 1 auto;
  min-width: 0;
  height: 27px;
  padding: 0 8px;
  border-radius: 9px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #1d1d1f;
  font-size: 11px;
  font-weight: 850;
  outline: none;
}

.home-chat-session-edit-button {
  flex: 0 0 auto;
  width: 27px;
  height: 27px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 9px;
  color: #64748b;
}

.home-chat-session-edit-button:hover:not(:disabled) {
  background: #eef2f7;
  color: #1d1d1f;
}

.home-chat-empty {
  min-height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #8e8e93;
  text-align: center;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.55;
}

.home-chat-empty .material-symbols-outlined {
  color: #c7c7cc;
  font-size: 30px;
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
