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
      <div v-else class="schedule-empty">
        <span class="material-symbols-outlined">event_available</span>
        <p>선택한 날짜에 등록된 일정이 없습니다.</p>
      </div>
    </section>

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
          <div>
            <strong>{{ item.title }}</strong>
            <span>{{ formatDateLabel(item.dateKey) }} · {{ item.time }}</span>
          </div>
          <em>{{ getConfidenceLabel(item.confidence) }}</em>
        </article>
      </div>
      <div v-else class="schedule-small-empty">확인할 AI 후보가 없습니다.</div>
    </section>

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
          <div>
            <strong>{{ item.title }}</strong>
            <span>{{ formatDateLabel(item.dateKey) }} · {{ getTypeLabel(item.type) }}</span>
          </div>
        </article>
      </div>
    </section>
  </aside>
</template>

<style scoped>
.schedule-side-card {
  min-height: 0;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.9);
  box-shadow: 0 24px 48px rgba(148, 163, 184, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 22px;
  overflow-y: auto;
}

.schedule-side-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
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
  font-weight: 900;
  color: #8e8e93;
  margin-bottom: 4px;
}

.schedule-side-heading h2 {
  font-size: 21px;
  font-weight: 950;
}

.schedule-side-heading strong {
  min-width: 28px;
  height: 28px;
  border-radius: 999px;
  background: #1d1d1f;
  color: white;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 950;
}

.schedule-list,
.schedule-compact-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.schedule-list-card,
.schedule-compact-card,
.schedule-empty,
.schedule-small-empty {
  border-radius: 18px;
  background: #f4ede4;
  border: 1px solid rgba(255, 255, 255, 0.86);
}

.schedule-list-card {
  padding: 14px;
}

.schedule-list-card.pending {
  border-color: rgba(245, 158, 11, 0.26);
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

.schedule-status {
  font-size: 11px;
  font-weight: 900;
  color: #8e8e93;
}

.schedule-list-card h3 {
  font-size: 16px;
  font-weight: 950;
  margin-bottom: 7px;
}

.schedule-list-meta {
  gap: 5px;
  color: #6b7280;
  font-size: 12px;
  font-weight: 800;
}

.schedule-list-meta .material-symbols-outlined {
  font-size: 15px;
}

.schedule-list-card p {
  margin-top: 8px;
  color: #6b7280;
  font-size: 12px;
  font-weight: 700;
}

.schedule-list-card blockquote {
  margin-top: 10px;
  padding: 10px 11px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.58);
  color: #6b7280;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.5;
}

.schedule-card-actions {
  gap: 8px;
  margin-top: 12px;
}

.schedule-primary-btn,
.schedule-secondary-btn {
  border: none;
  border-radius: 999px;
  padding: 9px 13px;
  font-size: 12px;
  font-weight: 950;
}

.schedule-primary-btn {
  color: white;
  background: #2f64ed;
}

.schedule-secondary-btn {
  color: #6b7280;
  background: rgba(255, 255, 255, 0.72);
}

.schedule-compact-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px;
  cursor: pointer;
}

.schedule-compact-card strong {
  display: block;
  font-size: 13px;
  font-weight: 950;
  margin-bottom: 4px;
}

.schedule-compact-card span,
.schedule-compact-card em {
  color: #6b7280;
  font-size: 11px;
  font-weight: 800;
  font-style: normal;
}

.schedule-empty,
.schedule-small-empty {
  padding: 18px;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
  text-align: center;
}

.schedule-empty .material-symbols-outlined {
  font-size: 28px;
  margin-bottom: 8px;
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
