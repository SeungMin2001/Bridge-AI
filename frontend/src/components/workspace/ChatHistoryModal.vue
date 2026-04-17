<script setup>
import { useChat } from '../../composables/useChat'

defineProps({
  isOpen: { type: Boolean, required: true }
})

const emit = defineEmits(['close'])
const { messages, clearHistory } = useChat()

const formatDate = () => {
  const now = new Date()
  return now.toLocaleDateString() + ' ' + (now.getHours().toString().padStart(2, '0')) + ':' + (now.getMinutes().toString().padStart(2, '0'))
}
</script>

<template>
  <transition name="modal-fade">
    <div v-if="isOpen" class="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
      <!-- Backdrop -->
      <div class="absolute inset-0 bg-[#1d1d1f]/40 backdrop-blur-sm" @click="emit('close')"></div>
      
      <!-- Modal Content -->
      <div class="relative w-full max-w-2xl bg-white rounded-3xl shadow-2xl flex flex-col max-h-[80vh] overflow-hidden border border-[#f2f2f7]">
        <!-- Header -->
        <div class="px-6 py-5 border-b border-[#f2f2f7] flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-[#f2f2f7] flex items-center justify-center">
              <span class="material-symbols-outlined text-[#1d1d1f] text-[22px]">history</span>
            </div>
            <div>
              <h2 class="text-[17px] font-bold text-[#1d1d1f]">AI 채팅 히스토리</h2>
              <p class="text-[11px] text-[#8e8e93] font-medium">현재 세션의 대화 내역입니다</p>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <button 
              v-if="messages.length > 0"
              class="text-[12px] text-[#ff3b30] font-semibold px-3 py-1.5 hover:bg-[#ff3b30]/5 rounded-lg transition-colors"
              @click="clearHistory"
            >
              전체 삭제
            </button>
            <button 
              class="w-8 h-8 rounded-full hover:bg-[#f2f2f7] flex items-center justify-center transition-colors"
              @click="emit('close')"
            >
              <span class="material-symbols-outlined text-[#8e8e93] text-[20px]">close</span>
            </button>
          </div>
        </div>

        <!-- Body -->
        <div class="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          <div v-if="messages.length === 0" class="h-full flex flex-col items-center justify-center py-20 text-center">
            <div class="w-16 h-16 bg-[#f2f2f7] rounded-full flex items-center justify-center mb-4">
              <span class="material-symbols-outlined text-[#aeaeb2] text-[32px]">chat_bubble_outline</span>
            </div>
            <p class="text-[14px] text-[#8e8e93] font-medium">기록이 비어 있습니다.<br>AI와 대화를 시작해 보세요!</p>
          </div>

          <div v-else v-for="(msg, i) in messages" :key="i" class="flex flex-col gap-2">
            <div :class="['flex items-center gap-2 mb-1', msg.role === 'user' ? 'justify-end' : 'justify-start']">
              <span class="text-[10px] text-[#aeaeb2] font-semibold">{{ formatDate() }}</span>
              <span v-if="msg.role === 'user'" class="text-[10px] text-[#1d1d1f] font-bold">나</span>
              <span v-else class="text-[10px] text-[#3b82f6] font-bold uppercase">LectoAI</span>
            </div>
            
            <div :class="[
              'px-4 py-3 rounded-2xl text-[13px] leading-relaxed shadow-sm max-w-[85%]',
              msg.role === 'user' ? 'bg-[#1d1d1f] text-white self-end rounded-tr-none' : 'bg-[#f2f2f7] text-[#1d1d1f] self-start rounded-tl-none'
            ]">
              <div v-if="msg.thinking" class="mb-2 p-2 bg-white/50 rounded-lg border border-[#e5e5ea]">
                <div class="flex items-center gap-1.5 mb-1">
                  <span class="material-symbols-outlined text-[12px] text-[#8e8e93]">psychology</span>
                  <span class="text-[9px] font-bold text-[#8e8e93]">사고 과정</span>
                </div>
                <div class="text-[11px] text-[#1d1d1f]/60 whitespace-pre-wrap leading-tight italic">{{ msg.thinking }}</div>
              </div>
              <div class="whitespace-pre-wrap text-[13px] leading-relaxed">{{ msg.text || (msg.phase === 'thinking' ? '생각하는 중...' : '') }}</div>
              
              <!-- Citations -->
              <div v-if="msg.citations && msg.citations.length" class="mt-2 pt-2 border-t border-black/5">
                <div class="flex items-center gap-1 mb-1.5">
                  <span class="material-symbols-outlined text-[12px] text-[#8e8e93]">menu_book</span>
                  <span class="text-[10px] font-bold text-[#8e8e93]">참고 출처</span>
                </div>
                <div v-for="(cite, ci) in msg.citations" :key="ci" class="flex items-start gap-1.5 mb-1 last:mb-0">
                  <span class="text-[10px] text-blue-500 font-bold mt-0.5">{{ ci + 1 }}</span>
                  <span class="text-[10px] text-[#636366] leading-[1.5]">{{ cite.citation }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="px-6 py-4 bg-[#f2f2f7]/50 border-t border-[#f2f2f7] flex justify-center">
          <p class="text-[10px] text-[#aeaeb2]">AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.</p>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-active .relative,
.modal-fade-leave-active .relative {
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.modal-fade-enter-from .relative {
  transform: scale(0.9) translateY(20px);
}

.modal-fade-leave-to .relative {
  transform: scale(0.9) translateY(20px);
}

.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #d1d1d6;
  border-radius: 10px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: #aeaeb2;
}
</style>
