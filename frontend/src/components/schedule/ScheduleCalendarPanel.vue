<script setup>
defineProps({
  calendarView: { type: String, required: true },
  calendarViewOptions: { type: Array, required: true },
  currentPeriodLabel: { type: String, required: true },
  calendarDays: { type: Array, required: true },
  weekDays: { type: Array, required: true },
  hourSlots: { type: Array, required: true },
  selectedDateKey: { type: String, required: true },
  selectedSchedules: { type: Array, required: true },
  getWeekEventStyle: { type: Function, required: true },
  formatHourSlot: { type: Function, required: true },
  getTypeLabel: { type: Function, required: true }
})

const emit = defineEmits([
  'set-calendar-view',
  'select-date',
  'focus-schedule',
  'show-popover',
  'hide-popover'
])
</script>

<template>
  <section class="schedule-calendar-panel">
    <div class="schedule-calendar-title-row">
      <div v-if="calendarViewOptions.length" class="schedule-view-switch" role="tablist" aria-label="일정 보기">
        <button
          v-for="option in calendarViewOptions"
          :key="option.value"
          type="button"
          class="schedule-view-tab"
          :class="{ active: calendarView === option.value }"
          role="tab"
          :aria-selected="calendarView === option.value"
          @click="emit('set-calendar-view', option.value)"
        >
          {{ option.label }}
        </button>
      </div>
      <h2>{{ currentPeriodLabel }}</h2>
      <div class="schedule-legend">
        <span><i class="confirmed"></i>확정</span>
        <span><i class="pending"></i>AI 후보</span>
      </div>
    </div>

    <template v-if="calendarView === 'month'">
      <div class="schedule-weekdays">
        <span v-for="label in ['일', '월', '화', '수', '목', '금', '토']" :key="label">{{ label }}</span>
      </div>

      <div class="schedule-calendar-grid">
        <div
          v-for="day in calendarDays"
          :key="day.dateKey"
          class="schedule-day-cell"
          :class="{ muted: day.muted, today: day.isToday, selected: day.dateKey === selectedDateKey, pending: day.hasPendingSchedule, confirmed: day.hasConfirmedSchedule }"
          role="button"
          tabindex="0"
          @click="emit('select-date', day)"
          @keydown.enter.prevent="emit('select-date', day)"
          @keydown.space.prevent="emit('select-date', day)"
        >
          <span class="schedule-day-number">{{ day.label }}</span>
          <div class="schedule-day-items">
            <span
              v-for="item in day.schedules.slice(0, 3)"
              :key="`${day.dateKey}-${item.id}`"
              class="schedule-day-pill"
              :class="item.status"
              tabindex="0"
              @click.stop="emit('focus-schedule', item)"
              @mouseenter="emit('show-popover', item, $event)"
              @mouseleave="emit('hide-popover')"
              @focus="emit('show-popover', item, $event)"
              @blur="emit('hide-popover')"
            >
              <span>{{ item.title }}</span>
              <em>{{ item.time }}</em>
            </span>
          </div>
          <span v-if="day.schedules.length > 3" class="schedule-day-more">+{{ day.schedules.length - 3 }}</span>
        </div>
      </div>
    </template>

    <div v-else-if="calendarView === 'week'" class="schedule-week-view">
      <div class="schedule-week-header-grid">
        <div class="schedule-week-time-corner"></div>
        <button
          v-for="day in weekDays"
          :key="`week-head-${day.dateKey}`"
          type="button"
          class="schedule-week-head-cell"
          :class="{ today: day.isToday, selected: day.dateKey === selectedDateKey }"
          @click="emit('select-date', day)"
        >
          <span>{{ day.dayName }}</span>
          <strong>{{ day.label }}</strong>
        </button>
      </div>

      <div class="schedule-week-scroll">
        <div class="schedule-week-time-axis">
          <div
            v-for="hour in hourSlots"
            :key="`week-time-${hour}`"
            class="schedule-week-time-slot"
          >
            {{ formatHourSlot(hour) }}
          </div>
        </div>
        <div class="schedule-week-grid">
          <section
            v-for="day in weekDays"
            :key="`week-body-${day.dateKey}`"
            class="schedule-week-column"
            :class="{ selected: day.dateKey === selectedDateKey }"
            @click="emit('select-date', day)"
          >
            <div
              v-for="hour in hourSlots"
              :key="`week-line-${day.dateKey}-${hour}`"
              class="schedule-week-hour-line"
            ></div>
            <article
              v-for="item in day.schedules"
              :key="`week-event-${item.id}`"
              class="schedule-week-event"
              :class="item.status"
              :style="getWeekEventStyle(item)"
              tabindex="0"
              @click.stop="emit('focus-schedule', item)"
              @mouseenter="emit('show-popover', item, $event)"
              @mouseleave="emit('hide-popover')"
              @focus="emit('show-popover', item, $event)"
              @blur="emit('hide-popover')"
            >
              <strong>{{ item.title }}</strong>
              <span>{{ item.time }}</span>
            </article>
          </section>
        </div>
      </div>
    </div>

    <div v-else class="schedule-day-agenda">
      <div class="schedule-day-agenda-date">
        <span>{{ currentPeriodLabel }}</span>
        <strong>{{ selectedSchedules.length }}</strong>
      </div>
      <div v-if="selectedSchedules.length" class="schedule-day-agenda-list">
        <article
          v-for="item in selectedSchedules"
          :key="`day-agenda-${item.id}`"
          class="schedule-day-agenda-item"
          :class="item.status"
          tabindex="0"
          @mouseenter="emit('show-popover', item, $event)"
          @mouseleave="emit('hide-popover')"
          @focus="emit('show-popover', item, $event)"
          @blur="emit('hide-popover')"
          @click="emit('focus-schedule', item)"
        >
          <div>
            <span>{{ item.time }}</span>
            <strong>{{ item.title }}</strong>
          </div>
          <p>{{ item.note || item.sourceText || getTypeLabel(item.type) }}</p>
        </article>
      </div>
      <div v-else class="schedule-day-agenda-empty">등록된 일정이 없습니다.</div>
    </div>
  </section>
</template>

<style scoped>
.schedule-calendar-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  min-width: 0;
  flex: 1;
  padding: 0;
  border-radius: 22px;
  background: transparent;
  border: 0;
}

.schedule-calendar-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.schedule-calendar-title-row h2 {
  flex: 1;
  text-align: center;
  font-size: 24px;
  font-weight: 950;
  color: var(--copy-text);
}

.schedule-view-switch {
  display: inline-flex;
  align-items: center;
  min-width: 220px;
  height: 38px;
  padding: 3px;
  border-radius: 10px;
  background: var(--copy-surface-soft);
  border: 1px solid var(--copy-line);
}

.schedule-view-tab {
  flex: 1;
  height: 30px;
  border: 0;
  border-radius: 8px;
  color: #64748b;
  background: transparent;
  font-size: 11px;
  font-weight: 900;
  cursor: pointer;
}

.schedule-view-tab.active {
  color: #ffffff;
  background: var(--copy-black);
}

.schedule-view-tab:not(.active):hover {
  color: var(--copy-text);
  background: #fff;
}

.schedule-legend {
  display: flex;
  gap: 12px;
  color: #6b7280;
  font-size: 12px;
  font-weight: 800;
}

.schedule-legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.schedule-legend i {
  width: 8px;
  height: 8px;
  border-radius: 999px;
}

.schedule-legend .confirmed {
  background: var(--copy-green);
}

.schedule-legend .pending {
  background: var(--copy-amber);
}

.schedule-weekdays,
.schedule-calendar-grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.schedule-weekdays {
  gap: 8px;
  margin-bottom: 8px;
}

.schedule-weekdays span {
  padding: 0 10px;
  color: #6b7280;
  font-size: 12px;
  font-weight: 900;
  text-align: center;
}

.schedule-calendar-grid {
  flex: 1;
  min-height: 0;
  gap: 0;
  grid-auto-rows: minmax(84px, 1fr);
  overflow: hidden;
  border: 1px solid var(--copy-line);
  border-radius: 18px;
  background: var(--copy-surface);
}

.schedule-day-cell {
  position: relative;
  min-height: 78px;
  border: 0;
  border-right: 1px solid var(--copy-line);
  border-bottom: 1px solid var(--copy-line);
  border-radius: 0;
  padding: 10px;
  background: var(--copy-surface);
  text-align: left;
  overflow: hidden;
  cursor: pointer;
}

.schedule-day-cell:nth-child(7n) {
  border-right: 0;
}

.schedule-day-cell:nth-last-child(-n + 7) {
  border-bottom: 0;
}

.schedule-day-cell:hover,
.schedule-day-cell.selected {
  background: var(--copy-surface-soft);
}

.schedule-day-cell.muted {
  background:
    repeating-linear-gradient(
      45deg,
      #fbfaff,
      #fbfaff 8px,
      #f0edf6 8px,
      #f0edf6 16px
    );
  color: #9a8d7a;
}

.schedule-day-cell.today .schedule-day-number {
  background: var(--copy-black);
  color: white;
}

.schedule-day-number {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 950;
}

.schedule-day-cell.pending::after,
.schedule-day-cell.confirmed::before {
  content: '';
  position: absolute;
  top: 12px;
  right: 12px;
  width: 7px;
  height: 7px;
  border-radius: 999px;
}

.schedule-day-cell.confirmed::before {
  background: var(--copy-green);
}

.schedule-day-cell.pending::after {
  right: 22px;
  background: var(--copy-amber);
}

.schedule-day-items {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 8px;
}

.schedule-day-pill,
.schedule-day-more {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  border-radius: 5px;
  padding: 5px 7px;
  font-size: 10px;
  font-weight: 900;
}

.schedule-day-pill {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  cursor: pointer;
}

.schedule-day-pill span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.schedule-day-pill em {
  flex: 0 0 auto;
  font-size: 9px;
  font-style: normal;
  opacity: 0.76;
}

.schedule-day-pill.confirmed {
  background: var(--copy-green);
  color: #ffffff;
}

.schedule-day-pill.pending {
  background: #ffca64;
  color: #5f3b00;
}

.schedule-day-more {
  display: inline-block;
  color: #6b7280;
}

.schedule-week-view,
.schedule-day-agenda {
  flex: 1;
  min-height: 0;
}

.schedule-week-view {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--copy-line);
  border-radius: 18px;
  overflow: hidden;
  background: var(--copy-surface);
}

.schedule-week-header-grid {
  display: grid;
  grid-template-columns: 76px repeat(7, minmax(0, 1fr));
  border-bottom: 1px solid var(--copy-line);
  background: var(--copy-surface-soft);
}

.schedule-week-time-corner {
  border-right: 1px solid var(--copy-line);
}

.schedule-week-head-cell {
  min-height: 64px;
  border: 0;
  border-right: 1px solid var(--copy-line);
  background: transparent;
  color: #64748b;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  cursor: pointer;
}

.schedule-week-head-cell:last-child {
  border-right: 0;
}

.schedule-week-head-cell span {
  font-size: 11px;
  font-weight: 900;
}

.schedule-week-head-cell strong {
  width: 30px;
  height: 30px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #334155;
  font-size: 14px;
  font-weight: 950;
}

.schedule-week-head-cell.today strong {
  color: #ffffff;
  background: var(--copy-black);
}

.schedule-week-head-cell.selected {
  background: var(--copy-surface-soft);
}

.schedule-week-scroll {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 76px minmax(0, 1fr);
  overflow-y: auto;
}

.schedule-week-time-axis {
  background: var(--copy-surface);
  border-right: 1px solid var(--copy-line);
}

.schedule-week-time-slot {
  height: 68px;
  padding: 8px 8px 0 0;
  color: #8a7d6d;
  font-size: 11px;
  font-weight: 900;
  text-align: right;
  border-bottom: 1px solid var(--copy-line);
}

.schedule-week-grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  min-height: calc(68px * 15);
}

.schedule-week-column {
  position: relative;
  min-height: calc(68px * 15);
  padding: 0 8px;
  border-right: 1px solid var(--copy-line);
  background: var(--copy-surface);
  cursor: pointer;
}

.schedule-week-column:last-child {
  border-right: 0;
}

.schedule-week-column.selected {
  background: var(--copy-surface-soft);
}

.schedule-week-hour-line {
  height: 68px;
  border-bottom: 1px solid var(--copy-line);
}

.schedule-week-event {
  position: absolute;
  left: 8px;
  right: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-radius: 7px;
  padding: 8px;
  cursor: pointer;
  overflow: hidden;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}

.schedule-week-event.confirmed {
  color: #ffffff;
  background: var(--copy-green);
}

.schedule-week-event.pending {
  color: #5f3b00;
  background: #ffca64;
}

.schedule-week-event strong,
.schedule-week-event span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.schedule-week-event strong {
  font-size: 11px;
  font-weight: 950;
}

.schedule-week-event span {
  font-size: 10px;
  font-weight: 900;
  opacity: 0.78;
}

.schedule-day-agenda {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px;
  border: 1px solid var(--copy-line);
  border-radius: 18px;
  background: var(--copy-surface);
}

.schedule-day-agenda-date {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--copy-text);
  font-weight: 950;
}

.schedule-day-agenda-date span {
  font-size: 18px;
}

.schedule-day-agenda-date strong {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  background: var(--copy-black);
  font-size: 13px;
}

.schedule-day-agenda-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  padding-right: 4px;
}

.schedule-day-agenda-item {
  display: grid;
  grid-template-columns: minmax(160px, 0.42fr) minmax(0, 1fr);
  gap: 18px;
  padding: 14px;
  border-radius: 14px;
  border-left: 6px solid var(--copy-green);
  background: var(--copy-surface-soft);
  cursor: pointer;
}

.schedule-day-agenda-item.pending {
  border-left-color: #ffca64;
}

.schedule-day-agenda-item span {
  display: block;
  color: #64748b;
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 4px;
}

.schedule-day-agenda-item strong {
  display: block;
  color: #111827;
  font-size: 15px;
  font-weight: 950;
}

.schedule-day-agenda-item p {
  min-width: 0;
  color: #64748b;
  font-size: 13px;
  font-weight: 750;
  line-height: 1.5;
}

.schedule-day-agenda-empty {
  flex: 1;
  min-height: 220px;
  border-radius: 14px;
  color: #94a3b8;
  background: var(--copy-surface-soft);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 900;
}

@media (max-width: 720px) {
  .schedule-calendar-title-row {
    align-items: stretch;
    flex-direction: column;
  }

  .schedule-calendar-title-row h2 {
    text-align: left;
    font-size: 19px;
  }

  .schedule-view-switch {
    width: 100%;
    min-width: 0;
  }

  .schedule-day-cell {
    min-height: 72px;
    padding: 7px;
  }

  .schedule-day-pill,
  .schedule-week-event span {
    display: none;
  }

  .schedule-week-view {
    overflow-x: auto;
  }

  .schedule-week-header-grid,
  .schedule-week-scroll {
    min-width: 760px;
  }

  .schedule-day-agenda-item {
    grid-template-columns: 1fr;
    gap: 8px;
  }
}
</style>
