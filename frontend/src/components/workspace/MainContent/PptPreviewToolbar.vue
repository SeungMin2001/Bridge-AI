<!-- 워크스페이스 PPT 미리보기 안에서 슬라이드 이동과 현재 위치를 표시하는 툴바입니다. -->
<script setup>
defineProps({
  slideIndex: { type: Number, default: 0 },
  slideCount: { type: Number, default: 0 },
  loading: { type: Boolean, default: false }
})

defineEmits(['previous', 'next'])
</script>

<template>
  <div class="ppt-preview-toolbar">
    <button type="button" class="ppt-nav-btn" :disabled="slideIndex <= 0 || loading" @click="$emit('previous')">
      이전
    </button>
    <span class="ppt-slide-status">
      {{ slideCount ? `${slideIndex + 1} / ${slideCount}` : '슬라이드 준비 중' }}
    </span>
    <button type="button" class="ppt-nav-btn" :disabled="slideIndex >= slideCount - 1 || loading" @click="$emit('next')">
      다음
    </button>
  </div>
</template>

<style scoped>
.ppt-preview-toolbar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 10px;
}

.ppt-nav-btn {
  padding: 8px 14px;
  border-radius: 12px;
  border: 1px solid rgba(229, 231, 235, 0.9);
  background: rgba(255, 255, 255, 0.92);
  font-size: 12px;
  font-weight: 700;
  color: #374151;
}

.ppt-nav-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ppt-slide-status {
  font-size: 12px;
  font-weight: 800;
  color: #1d1d1f;
}
</style>
