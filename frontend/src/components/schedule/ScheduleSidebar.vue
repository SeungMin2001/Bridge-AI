<script setup>
defineProps({
  selectedDateKey: { type: String, required: true },
  selectedSchedules: { type: Array, required: true },
  pendingSchedules: { type: Array, required: true },
  upcomingSchedules: { type: Array, required: true },
  formatDateLabel: { type: Function, required: true },
  getTypeLabel: { type: Function, required: true },
  getTypeIcon: { type: Function, required: true },
  getStatusLabel: { type: Function, required: true },
  getConfidenceLabel: { type: Function, required: true }
})

const emit = defineEmits([
  'confirm',
  'ignore',
  'open-workspace',
  'focus-schedule'
])
</script>

<template>
  <aside class="schedule-side-card">
    <section class="schedule-side-section">
      <div class="schedule-side-heading">
        <div>
          <span>선택한 날짜</span>
          <h2>{{ formatDateLabel(selectedDateKey) }}</h2>
        </div>
      </div>

      <div v-if="selectedSchedules.length" class="schedule-list">
        <article
          v-for="item in selectedSchedules"
          :key="item.id"
          class="schedule-list-card"
          :class="item.status"
        >
          <div class="schedule-list-top">
            <span class="schedule-type-chip">
              <span class="material-symbols-outlined">{{ getTypeIcon(item.type) }}</span>
              {{ getTypeLabel(item.type) }}
            </span>
            <span class="schedule-status">{{ getStatusLabel(item.status) }}</span>
          </div>
          <h3>{{ item.title }}</h3>
          <div class="schedule-list-meta">
            <span class="material-symbols-outlined">schedule</span>
            {{ item.time }}
          </div>
          <p v-if="item.note">{{ item.note }}</p>
          <blockquote v-if="item.sourceText">{{ item.sourceText }}</blockquote>
          <div class="schedule-card-actions">
            <template v-if="item.status === 'pending'">
              <button class="schedule-primary-btn" @click="emit('confirm', item)">확정</button>
              <button class="schedule-secondary-btn" @click="emit('ignore', item)">무시</button>
            </template>
            <template v-else>
              <button class="schedule-primary-btn" @click="emit('open-workspace', item)">워크스페이스 열기</button>
            </template>
          </div>
        </article>
      </div>
      <div v-else class="schedule-empty-minimal">
        <span class="material-symbols-outlined">event_busy</span>
        <p>등록된 일정이 없습니다.</p>
      </div>
    </section>

    <div class="schedule-divider"></div>

    <section class="schedule-side-section">
      <div class="schedule-side-heading compact">
        <div>
          <span>AI가 찾은 일정</span>
          <h2>확인 대기</h2>
        </div>
        <strong>{{ pendingSchedules.length }}</strong>
      </div>

      <div v-if="pendingSchedules.length" class="schedule-compact-list">
        <article
          v-for="item in pendingSchedules.slice(0, 4)"
          :key="`pending-${item.id}`"
          class="schedule-compact-card"
          @click="emit('focus-schedule', item)"
        >
          <div class="compact-info">
            <strong>{{ item.title }}</strong>
            <span>{{ formatDateLabel(item.dateKey) }} · {{ item.time }}</span>
          </div>
          <em class="confidence-badge">{{ getConfidenceLabel(item.confidence) }}</em>
        </article>
      </div>
      <div v-else class="schedule-small-empty-minimal">
        <span class="material-symbols-outlined">done_all</span>
        <p>모든 일정을 확인했습니다.</p>
      </div>
    </section>

    <div class="schedule-divider"></div>

    <section class="schedule-side-section">
      <div class="schedule-side-heading compact">
        <div>
          <span>전체 일정</span>
          <h2>다가오는 일정</h2>
        </div>
      </div>

      <div class="schedule-compact-list">
        <article
          v-for="item in upcomingSchedules"
          :key="`upcoming-${item.id}`"
          class="schedule-compact-card"
          @click="emit('focus-schedule', item)"
        >
          <div class="compact-info">
            <strong>{{ item.title }}</strong>
            <span>{{ formatDateLabel(item.dateKey) }} · {{ getTypeLabel(item.type) }}</span>
          </div>
          <span class="material-symbols-outlined arrow-icon">chevron_right</span>
        </article>
      </div>
    </section>
  </aside>
</template>

<style scoped>
.schedule-side-card {
  width: 100%;
  min-width: 0;
  max-width: none;
  min-height: 0;
  align-self: stretch;
  justify-self: stretch;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.95);
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 1);
  backdrop-filter: blur(24px) saturate(140%);
  -webkit-backdrop-filter: blur(24px) saturate(140%);
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 24px 12px;
  overflow-y: auto;
}

.schedule-divider {
  height: 1px;
  background: linear-gradient(90deg, rgba(200, 200, 200, 0) 0%, rgba(200, 200, 200, 0.3) 50%, rgba(200, 200, 200, 0) 100%);
  margin: 0 10px;
}

.schedule-side-section {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.schedule-side-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.schedule-side-heading span {
  display: block;
  font-size: 12px;
  font-weight: 800;
  color: #a1a1aa;
  margin-bottom: 2px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.schedule-side-heading h2 {
  font-size: 20px;
  font-weight: 900;
  color: #18181b;
  letter-spacing: -0.3px;
}

.schedule-side-heading strong {
  min-width: 26px;
  height: 26px;
  border-radius: 8px;
  background: #3b82f6;
  color: white;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 900;
  box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
}

.schedule-list,
.schedule-compact-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.schedule-list-card {
  border-radius: 16px;
  background: #f4ede4;
  border: 1px solid rgba(220, 210, 200, 0.8);
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
  transition: all 0.2s ease;
}

.schedule-list-card:hover {
  transform: translateY(-2px);
  background: #e6dfd6;
  border-color: #cbbfaa;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.schedule-list-card.pending {
  background: #fdf5e6;
  border-color: #fcd34d;
}

.schedule-list-card.pending:hover {
  background: #faedce;
  border-color: #fbbf24;
}

.schedule-list-top,
.schedule-list-meta,
.schedule-card-actions {
  display: flex;
  align-items: center;
}

.schedule-list-top {
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

.schedule-type-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border-radius: 6px;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.7);
  color: #2563eb;
  font-size: 11px;
  font-weight: 800;
}

.schedule-type-chip .material-symbols-outlined {
  font-size: 14px;
}

.schedule-status {
  font-size: 11px;
  font-weight: 800;
  color: #a1a1aa;
}

.schedule-list-card h3 {
  font-size: 16px;
  font-weight: 900;
  color: #27272a;
  margin-bottom: 6px;
  line-height: 1.4;
}

.schedule-list-meta {
  gap: 4px;
  color: #71717a;
  font-size: 12px;
  font-weight: 700;
}

.schedule-list-meta .material-symbols-outlined {
  font-size: 14px;
}

.schedule-list-card p {
  margin-top: 8px;
  color: #71717a;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.5;
}

.schedule-list-card blockquote {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.4);
  color: #52525b;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.5;
  border-left: 3px solid #d4d4d8;
}

.schedule-card-actions {
  gap: 8px;
  margin-top: 14px;
}

.schedule-primary-btn,
.schedule-secondary-btn {
  border: none;
  border-radius: 10px;
  padding: 8px 14px;
  font-size: 12px;
  font-weight: 800;
  cursor: pointer;
  transition: all 0.2s ease;
}

.schedule-primary-btn {
  color: white;
  background: #3b82f6;
  box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
}

.schedule-primary-btn:hover {
  background: #2563eb;
  transform: translateY(-1px);
}

.schedule-secondary-btn {
  color: #52525b;
  background: rgba(255, 255, 255, 0.6);
}

.schedule-secondary-btn:hover {
  background: rgba(255, 255, 255, 0.9);
}

.schedule-compact-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px;
  border-radius: 14px;
  background: #f4ede4;
  border: 1px solid rgba(220, 210, 200, 0.8);
  cursor: pointer;
  transition: all 0.2s ease;
}

.schedule-compact-card:hover {
  background: #e6dfd6;
  border-color: #cbbfaa;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transform: translateX(2px);
}

.compact-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.compact-info strong {
  font-size: 13px;
  font-weight: 800;
  color: #27272a;
}

.compact-info span {
  color: #a1a1aa;
  font-size: 11px;
  font-weight: 600;
}

.confidence-badge {
  font-size: 10px;
  font-weight: 800;
  padding: 3px 6px;
  border-radius: 6px;
  background: #fef3c7;
  color: #d97706;
  font-style: normal;
}

.arrow-icon {
  color: #d4d4d8;
  font-size: 18px;
  transition: color 0.2s ease;
}

.schedule-compact-card:hover .arrow-icon {
  color: #a1a1aa;
}

.schedule-empty-minimal,
.schedule-small-empty-minimal {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(244, 244, 245, 0.6);
  border: 1px dashed rgba(212, 212, 216, 0.8);
  color: #a1a1aa;
  font-size: 13px;
  font-weight: 700;
}

.schedule-empty-minimal .material-symbols-outlined,
.schedule-small-empty-minimal .material-symbols-outlined {
  font-size: 18px;
}

.schedule-empty-minimal p,
.schedule-small-empty-minimal p {
  margin: 0;
}

@media (max-width: 1100px) {
  .schedule-side-card {
    overflow: visible;
  }
}

@media (max-width: 720px) {
  .schedule-side-card {
    padding: 16px;
    border-radius: 20px;
  }
}
</style>
