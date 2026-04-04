<script setup>
import { ref, computed } from 'vue'
import { useChat } from '../../composables/useChat'

defineProps({
  isOpen: { type: Boolean, required: true }
})

const emit = defineEmits(['close'])
const { 
  sessions, 
  currentSessionId, 
  switchToSession, 
  createNewSession, 
  deleteSession 
} = useChat()

const currentSession = computed(() => {
  return sessions.value.find(s => s.id === currentSessionId.value)
})

const formatDate = (dateStr) => {
  const date = new Date(dateStr)
  return date.toLocaleDateString()
}

const handleSwitch = (id) => {
  switchToSession(id)
}

const handleNewChat = () => {
  createNewSession()
}
</script>

<template>
  <transition name="modal-fade">
    <div v-if="isOpen" class="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8">
      <!-- Backdrop -->
      <div class="absolute inset-0 bg-[#1d1d1f]/40 backdrop-blur-sm" @click="emit('close')"></div>
      
      <!-- Modal Content -->
      <div class="relative w-full max-w-5xl bg-white rounded-3xl shadow-2xl flex h-[85vh] overflow-hidden border border-[#f2f2f7]">
        
        <!-- Sidebar (Session List) -->
        <div class="w-[300px] bg-[#1d1d1f] flex flex-col border-r border-[#2d2d2f]">
          <div class="p-6">
            <h2 class="text-white text-[20px] font-bold mb-6 flex items-center gap-2">
              <span class="material-symbols-outlined text-white">chat</span>
              채팅
            </h2>
            <button 
              @click="handleNewChat"
              class="w-full py-3 px-4 rounded-xl bg-white/10 text-white text-[14px] font-bold flex items-center justify-center gap-2 hover:bg-white/20 transition-all border border-white/5"
            >
              <span class="material-symbols-outlined text-[18px]">add</span>
              새로운 대화 시작
            </button>
          </div>

          <div class="flex-1 overflow-y-auto px-4 pb-6 custom-scrollbar-dark">
            <div class="space-y-1">
              <div 
                v-for="session in sessions" 
                :key="session.id"
                @click="handleSwitch(session.id)"
                :class="[
                  'group p-3 rounded-xl cursor-pointer transition-all flex items-center justify-between',
                  session.id === currentSessionId ? 'bg-white/15 text-white' : 'text-gray-400 hover:bg-white/5 hover:text-white'
                ]"
              >
                <div class="flex flex-col min-w-0">
                  <span class="text-[13px] font-bold truncate">{{ session.title || '새로운 대화' }}</span>
                  <span class="text-[10px] opacity-50">{{ formatDate(session.createdAt) }}</span>
                </div>
                <button 
                  class="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 rounded-lg transition-all"
                  @click.stop="deleteSession(session.id)"
                >
                  <span class="material-symbols-outlined text-[16px] text-red-400">delete</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Main Content (Conversation) -->
        <div class="flex-1 flex flex-col bg-[#fcfcfd]">
          <!-- Header -->
          <div class="px-6 py-5 border-b border-[#f2f2f7] flex items-center justify-between bg-white/80 backdrop-blur-md sticky top-0 z-10">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-xl bg-[#f2f2f7] flex items-center justify-center shadow-sm">
                <span class="material-symbols-outlined text-[#1d1d1f] text-[22px]">history</span>
              </div>
              <div>
                <h2 class="text-[17px] font-bold text-[#1d1d1f] truncate max-w-[400px]">
                  {{ currentSession?.title || '새로운 대화' }}
                </h2>
                <p class="text-[11px] text-[#8e8e93] font-medium">현재 세션의 대화 내역입니다</p>
              </div>
            </div>
            <button 
              class="w-9 h-9 rounded-full hover:bg-[#f2f2f7] flex items-center justify-center transition-all bg-white border border-[#f2f2f7]"
              @click="emit('close')"
            >
              <span class="material-symbols-outlined text-[#8e8e93] text-[20px]">close</span>
            </button>
          </div>

          <!-- Body -->
          <div class="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar-light bg-white/40">
            <div v-if="!currentSession?.messages.length" class="h-full flex flex-col items-center justify-center py-20 text-center">
              <div class="w-20 h-20 bg-[#f2f2f7] rounded-full flex items-center justify-center mb-6 shadow-inner">
                <span class="material-symbols-outlined text-[#aeaeb2] text-[40px]">chat_bubble_outline</span>
              </div>
              <p class="text-[15px] text-[#8e8e93] font-medium leading-relaxed">대화 기록이 없습니다.<br>AI에게 질문을 시작해보세요!</p>
            </div>

            <div v-else v-for="(msg, i) in currentSession.messages" :key="i" class="flex flex-col gap-3">
              <div :class="['flex items-center gap-3 mb-1', msg.role === 'user' ? 'justify-end' : 'justify-start']">
                <span v-if="msg.role === 'user'" class="text-[10px] text-[#8e8e93] font-semibold">나</span>
                <div v-else class="flex items-center gap-1.5">
                  <div class="w-6 h-6 rounded-lg bg-[#1d1d1f] flex items-center justify-center">
                    <span class="material-symbols-outlined text-white text-[12px]">auto_awesome</span>
                  </div>
                  <span class="text-[10px] text-[#1d1d1f] font-bold uppercase tracking-wider">LectoAI</span>
                </div>
              </div>
              
              <div :class="[
                'px-5 py-4 rounded-[24px] text-[14px] leading-relaxed shadow-sm max-w-[80%]',
                msg.role === 'user' ? 'bg-[#1d1d1f] text-white self-end rounded-tr-none' : 'bg-white text-[#1d1d1f] self-start rounded-tl-none border border-[#f2f2f7]'
              ]">
                <div v-if="msg.thinking" class="mb-3 p-3 bg-[#f2f2f7] rounded-xl border border-[#e5e5ea]">
                  <div class="flex items-center gap-2 mb-2">
                    <span class="material-symbols-outlined text-[14px] text-[#8e8e93]">psychology</span>
                    <span class="text-[10px] font-bold text-[#8e8e93]">생각하는 중...</span>
                  </div>
                  <div class="text-[12px] text-[#1d1d1f]/70 whitespace-pre-wrap leading-snug italic">{{ msg.thinking }}</div>
                </div>
                <div class="whitespace-pre-wrap">{{ msg.text || (msg.phase === 'thinking' ? '응답을 생성하고 있습니다...' : '') }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-active .relative,
.modal-fade-leave-active .relative {
  transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.4s ease;
}

.modal-fade-enter-from .relative {
  transform: scale(0.95) translateY(30px);
  opacity: 0;
}

.modal-fade-leave-to .relative {
  transform: scale(0.95) translateY(30px);
  opacity: 0;
}

.custom-scrollbar-dark::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar-dark::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar-dark::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
}
.custom-scrollbar-dark::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}

.custom-scrollbar-light::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar-light::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar-light::-webkit-scrollbar-thumb {
  background: #e5e5ea;
  border-radius: 10px;
}
.custom-scrollbar-light::-webkit-scrollbar-thumb:hover {
  background: #d1d1d6;
}
</style>
