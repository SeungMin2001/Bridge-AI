<!-- 워크스페이스 상단에 선택된 단어의 뜻과 관련 액션을 표시하는 카드입니다. -->
<script setup>
defineProps({
  wordData: { type: Object, required: true }
})

defineEmits(['close', 'ask-ai'])
</script>

<template>
  <div class="word-info-card card shrink-0">
    <div class="flex items-center justify-between gap-3">
      <div class="flex items-center gap-2.5">
        <div class="word-badge">
          <span class="material-symbols-outlined text-[14px]">dictionary</span>
        </div>
        <span class="text-[15px] font-extrabold text-[#1d1d1f] tracking-tight">{{ wordData.word }}</span>
      </div>
      <div class="word-card-header-actions">
        <button class="word-card-ai-link" @click="$emit('ask-ai')">
          <span class="material-symbols-outlined text-[13px]">auto_awesome</span>
          AI 질문
        </button>
        <button class="word-card-close-btn" @click="$emit('close')">
          <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">close</span>
        </button>
      </div>
    </div>
    <p class="text-[13px] text-[#3a3a3c] leading-[1.7] font-medium mt-2 mb-0">
      {{ wordData.desc }}
    </p>
    <p v-if="wordData.error" class="word-card-error">
      {{ wordData.error }}
    </p>
  </div>
</template>

<style scoped>
.word-info-card {
  padding: 14px 18px;
  border-left: 1px solid var(--workspace-sidebar-card-border);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 0 !important;
  background: var(--workspace-sidebar-card-bg);
  border-color: var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.word-info-card::before {
  border-radius: 0 !important;
  background: var(--workspace-sidebar-card-overlay);
}

.word-info-card::after {
  border-radius: 0 !important;
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

.word-card-ai-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-height: 30px;
  padding: 0 4px;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: #2f7df6;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0;
  cursor: pointer;
  transition: color 0.16s ease, text-decoration-color 0.16s ease;
}

.word-card-ai-link:hover {
  color: #1d4ed8;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.word-card-ai-link .material-symbols-outlined {
  color: #8e8e93;
  transition: color 0.16s ease;
}

.word-card-ai-link:hover .material-symbols-outlined {
  color: #1d4ed8;
}

.word-card-header-actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 6px;
}

.word-card-close-btn {
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 0;
  background: transparent;
  cursor: pointer;
  transition: background-color 0.16s ease;
}

.word-card-close-btn:hover {
  background: rgba(0, 0, 0, 0.05);
}

.word-card-error {
  margin: 8px 0 0;
  color: #dc2626;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.5;
}

</style>
