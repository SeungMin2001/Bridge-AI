<script setup>
defineProps({
  hoveredSchedule: { type: Object, default: null },
  formatDateLabel: { type: Function, required: true },
  getTypeLabel: { type: Function, required: true },
  getTypeIcon: { type: Function, required: true },
  getStatusLabel: { type: Function, required: true }
})
</script>

<template>
  <transition name="schedule-popover-fade">
    <aside
      v-if="hoveredSchedule"
      class="schedule-hover-popover"
      :style="{ left: `${hoveredSchedule.left}px`, top: `${hoveredSchedule.top}px` }"
    >
      <div class="schedule-hover-popover-top">
        <span class="schedule-type-chip">
          <span class="material-symbols-outlined">{{ getTypeIcon(hoveredSchedule.item.type) }}</span>
          {{ getTypeLabel(hoveredSchedule.item.type) }}
        </span>
        <em>{{ getStatusLabel(hoveredSchedule.item.status) }}</em>
      </div>
      <h3>{{ hoveredSchedule.item.title }}</h3>
      <div class="schedule-hover-popover-time">
        <span class="material-symbols-outlined">schedule</span>
        {{ formatDateLabel(hoveredSchedule.item.dateKey) }} · {{ hoveredSchedule.item.time }}
      </div>
      <p>{{ hoveredSchedule.description }}</p>
    </aside>
  </transition>
</template>

<style scoped>
.schedule-hover-popover {
  position: fixed;
  z-index: 50;
  width: 286px;
  padding: 15px;
  border-radius: 16px;
  color: #1f2937;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid #e2e8f0;
  box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
  pointer-events: none;
}

.schedule-hover-popover-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.schedule-type-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border-radius: 999px;
  padding: 4px 8px;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-size: 11px;
  font-weight: 950;
}

.schedule-type-chip .material-symbols-outlined {
  font-size: 14px;
}

.schedule-hover-popover-top em {
  color: #94a3b8;
  font-size: 11px;
  font-weight: 900;
  font-style: normal;
}

.schedule-hover-popover h3 {
  color: #111827;
  font-size: 16px;
  font-weight: 950;
  line-height: 1.35;
  margin-bottom: 8px;
}

.schedule-hover-popover-time {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #64748b;
  font-size: 12px;
  font-weight: 850;
  margin-bottom: 10px;
}

.schedule-hover-popover-time .material-symbols-outlined {
  font-size: 15px;
}

.schedule-hover-popover p {
  color: #64748b;
  font-size: 12px;
  font-weight: 750;
  line-height: 1.55;
  max-height: 94px;
  overflow: hidden;
}

.schedule-popover-fade-enter-active,
.schedule-popover-fade-leave-active {
  transition: opacity 0.14s ease, transform 0.14s ease;
}

.schedule-popover-fade-enter-from,
.schedule-popover-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
