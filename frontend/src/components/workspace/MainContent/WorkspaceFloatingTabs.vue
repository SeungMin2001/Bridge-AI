<!-- 워크스페이스 중앙 영역 하단에서 탭 전환을 담당하는 플로팅 탭바입니다. -->
<script setup>
defineProps({
  tabs: { type: Array, default: () => [] },
  activeTab: { type: String, required: true }
})

defineEmits(['change'])
</script>

<template>
  <div class="workspace-tab-float-wrap">
    <div class="workspace-tab-float">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        class="workspace-tab-chip"
        :class="{ 'is-active': activeTab === tab.key }"
        @click="$emit('change', tab.key)"
      >
        {{ tab.label }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.workspace-tab-float {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: #eee4d8;
  border: 1px solid rgba(228, 217, 203, 0.95);
  box-shadow:
    0 18px 36px rgba(208, 194, 177, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.34);
}

.workspace-tab-float::before {
  display: none;
}

.workspace-tab-float::after {
  display: none;
}

.workspace-tab-float-wrap {
  position: absolute;
  left: 50%;
  bottom: 28px;
  transform: translateX(-50%);
  z-index: 15;
}

.workspace-tab-chip {
  position: relative;
  z-index: 1;
  min-width: 0;
  padding: 12px 18px;
  border-radius: 999px;
  border: 1px solid transparent;
  background: transparent;
  color: rgba(86, 86, 92, 0.68);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: -0.02em;
  white-space: nowrap;
  word-break: keep-all;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: color 0.24s ease, transform 0.24s ease, box-shadow 0.24s ease, background 0.24s ease;
  cursor: pointer;
}

.workspace-tab-chip:hover {
  color: rgba(29, 29, 31, 0.84);
  background: rgba(255, 255, 255, 0.22);
}

.workspace-tab-chip.is-active {
  background:
    radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.96), transparent 48%),
    linear-gradient(180deg, rgba(251, 248, 243, 0.95), rgba(239, 230, 217, 0.88));
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.94);
  box-shadow:
    0 18px 28px rgba(211, 198, 180, 0.28),
    0 6px 18px rgba(255, 255, 255, 0.38),
    inset 0 1px 0 rgba(255, 255, 255, 0.98),
    inset 0 -2px 6px rgba(213, 197, 176, 0.3);
  backdrop-filter: blur(16px) saturate(150%);
  -webkit-backdrop-filter: blur(16px) saturate(150%);
}

.workspace-tab-chip.is-active::before {
  content: '';
  position: absolute;
  inset: 2px 6px auto;
  height: 52%;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.18));
  opacity: 0.95;
  pointer-events: none;
}

.workspace-tab-chip.is-active::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.38),
    inset 0 -10px 16px rgba(222, 206, 188, 0.16);
  pointer-events: none;
}

@media (max-width: 900px) {
  .workspace-tab-float-wrap {
    left: 24px;
    right: 24px;
    transform: none;
  }

  .workspace-tab-float {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    width: 100%;
    gap: 8px;
    padding: 8px;
  }

  .workspace-tab-chip {
    min-width: 0;
    padding: 12px 10px;
    font-size: 13px;
  }
}
</style>
