<!-- 워크스페이스 자료 탭에서 저장된 강의 자료 목록과 삭제 액션을 표시하는 리스트입니다. -->
<script setup>
const props = defineProps({
  materialAttachments: { type: Array, default: () => [] }
})

defineEmits(['open', 'delete'])

const formatFileSize = (bytes = 0) => {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

const fileIconName = (fileName = '') => {
  const lower = fileName.toLowerCase()
  if (lower.endsWith('.pdf')) return 'picture_as_pdf'
  return 'slideshow'
}
</script>

<template>
  <div v-if="props.materialAttachments.length" class="flex flex-col gap-3">
    <button
      v-for="file in props.materialAttachments"
      :key="file.id"
      type="button"
      class="pdf-file-row text-left"
      @click="$emit('open', file.id)"
    >
      <div class="flex items-center gap-3 min-w-0">
        <div class="pdf-file-badge">
          <span class="material-symbols-outlined text-[18px]">{{ fileIconName(file.name) }}</span>
        </div>
        <div class="min-w-0">
          <p class="text-[14px] font-bold text-[#1d1d1f] truncate">{{ file.name }}</p>
          <p class="text-[12px] text-[#8e8e93] mt-0.5">{{ formatFileSize(file.size) }}</p>
        </div>
      </div>
      <div class="flex items-center gap-2 shrink-0">
        <span class="pdf-file-action shrink-0">메모 탭에서 열기</span>
        <button
          type="button"
          class="material-delete-btn"
          title="자료 삭제"
          @click.stop="$emit('delete', file.id)"
        >
          삭제
        </button>
      </div>
    </button>
  </div>
  <div v-else class="lecture-material-empty">
    아직 저장된 자료가 없습니다. 메모 탭에서 파일을 올리면 여기에 저장됩니다.
  </div>
</template>

<style scoped>
.lecture-material-empty {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.72);
  color: #8e8e93;
  font-size: 13px;
  line-height: 1.7;
}

.pdf-file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.72);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.8), rgba(248, 250, 252, 0.7));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.96), 0 12px 24px rgba(148, 163, 184, 0.08);
  transition: all 0.2s ease;
}

.pdf-file-row:hover {
  transform: translateY(-1px);
  border-color: rgba(59, 130, 246, 0.24);
}

.pdf-file-badge {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: rgba(239, 246, 255, 0.95);
}

.pdf-file-action {
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 700;
  color: #4b5563;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(229, 231, 235, 0.9);
}

.material-delete-btn {
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid rgba(248, 113, 113, 0.24);
  background: rgba(254, 242, 242, 0.96);
  color: #b91c1c;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.material-delete-btn:hover {
  background: rgba(254, 226, 226, 0.98);
  border-color: rgba(239, 68, 68, 0.34);
}
</style>
