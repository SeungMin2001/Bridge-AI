<!-- 워크스페이스 상단에 선택된 단어의 뜻과 관련 액션을 표시하는 카드입니다. -->
<script setup>
defineProps({
  wordData: { type: Object, required: true }
})

defineEmits(['close', 'ask-ai', 'add-to-note'])
</script>

<template>
  <div class="word-info-card card shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-2.5">
        <div class="word-badge">
          <span class="material-symbols-outlined text-[14px]">dictionary</span>
        </div>
        <span class="text-[15px] font-extrabold text-[#1d1d1f] tracking-tight">{{ wordData.word }}</span>
      </div>
      <button class="p-1.5 rounded-full hover:bg-black/5 transition-colors cursor-pointer" @click="$emit('close')">
        <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">close</span>
      </button>
    </div>
    <p class="text-[13px] text-[#3a3a3c] leading-[1.7] font-medium mt-2 mb-0">
      {{ wordData.desc }}
    </p>
    <div class="flex items-center justify-between mt-2.5 pt-2.5 border-t border-black/5">
      <div class="flex items-center gap-1.5">
        <span class="material-symbols-outlined text-[13px] text-[#8e8e93]">link</span>
        <span class="text-[10px] font-bold text-[#8e8e93] uppercase tracking-wider">Source:</span>
        <span class="text-[10px] font-bold text-blue-500">{{ wordData.source }}</span>
      </div>
      <div class="flex gap-1.5">
        <button class="word-card-btn word-card-btn-primary" @click="$emit('ask-ai')">
          <span class="material-symbols-outlined text-[13px]">auto_awesome</span>
          AI 질문
        </button>
        <button class="word-card-btn word-card-btn-secondary" @click="$emit('add-to-note')">
          <span class="material-symbols-outlined text-[13px]">note_add</span>
          노트 추가
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.word-info-card {
  padding: 14px 18px;
  border-left: 1px solid var(--workspace-sidebar-card-border);
  background: var(--workspace-sidebar-card-bg);
  border-color: var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.word-info-card::before {
  background: var(--workspace-sidebar-card-overlay);
}

.word-info-card::after {
  border-color: var(--workspace-sidebar-card-inner-border);
}

.word-badge {
  width: 28px;
  height: 28px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.82);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  box-shadow: 0 10px 20px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.word-card-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  border-radius: 8px;
  border: none;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.word-card-btn-primary {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.72));
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.84);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.word-card-btn-primary:hover {
  background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(255, 255, 255, 0.78));
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(148, 163, 184, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.98);
}

.word-card-btn-secondary {
  background: rgba(255, 255, 255, 0.48);
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.72);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.word-card-btn-secondary:hover {
  background: rgba(255, 255, 255, 0.68);
  transform: translateY(-1px);
}
</style>
