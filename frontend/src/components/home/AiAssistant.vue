<script setup>
defineProps({
  isOpen: Boolean,
  aiWinRef: Object,
  aiBtnRef: Object
})

const emit = defineEmits(['update:isOpen'])
</script>

<template>
  <button
    ref="aiBtnRef"
    class="absolute bottom-8 right-8 w-[64px] h-[64px] bg-gradient-to-r from-[#3b82f6] to-[#5856d6] rounded-full shadow-[0_8px_24px_rgba(59,130,246,0.3)] flex items-center justify-center text-white hover:scale-105 active:scale-95 transition-all z-40"
    id="home-ai-btn"
    @click="emit('update:isOpen', !isOpen)"
  >
    <span class="material-symbols-outlined text-[28px]">auto_awesome</span>
  </button>

  <div ref="aiWinRef" :class="['home-ai-chat-window z-[60]', { open: isOpen }]">
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
      <div class="chat-bubble bubble-ai">
        안녕하세요! 어떤 것을 도와드릴까요? <br />강의 노트 요약이나 시험 문제 생성 등을 도와드릴 수 있습니다.
      </div>
      <div class="chat-bubble bubble-user">
        지난 '컴퓨터 네트워크' 수업 내용을 요약해줘.
      </div>
      <div class="chat-bubble bubble-ai">
        네, '컴퓨터 네트워크' 폴더의 최신 노트를 분석하여 요약해 드릴게요. <br /><br />
        1. OSI 7계층의 구조와 각 계층의 역할<br />
        2. TCP/IP 4계층 모델과의 차이점<br />
        3. 데이터 캡슐화와 비캡슐화 과정...
      </div>
    </div>
    <div class="chat-footer">
      <div class="chat-input-container">
        <input class="chat-input" placeholder="AI에게 질문해보세요..." type="text" />
        <button class="btn-ghost-icon p-1 text-[#373549]">
          <span class="material-symbols-outlined text-[20px]">send</span>
        </button>
      </div>
    </div>
  </div>
</template>
