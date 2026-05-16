<!-- AI가 감지한 일정과 확정 일정을 큰 캘린더에서 관리하는 페이지입니다. -->
<script setup>
import { onMounted } from 'vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import ScheduleCalendarPanel from '../../components/schedule/ScheduleCalendarPanel.vue'
import ScheduleHoverPopover from '../../components/schedule/ScheduleHoverPopover.vue'
import ScheduleSidebar from '../../components/schedule/ScheduleSidebar.vue'
import { useScheduleCalendarView } from '../../composables/schedule/useScheduleCalendarView'
import { useScheduleIcsExport } from '../../composables/schedule/useScheduleIcsExport'
import { useScheduleState } from '../../composables/useScheduleState'

const emit = defineEmits(['navigate'])

defineProps({
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const {
  visibleSchedules,
  pendingSchedules,
  hydrateSchedules,
  confirmSchedule,
  ignoreSchedule,
  getSchedulesForDate,
  getScheduleDayFlags,
  formatDateKey,
  formatDateLabel,
  getTypeLabel,
  getTypeIcon,
  getStatusLabel
} = useScheduleState()

const {
  calendarView,
  calendarViewOptions,
  hourSlots,
  hoveredSchedule,
  selectedDateKey,
  selectedSchedules,
  currentPeriodLabel,
  calendarDays,
  weekDays,
  getWeekEventStyle,
  formatHourSlot,
  setCalendarView,
  movePeriod,
  moveToday,
  selectDate,
  focusSchedule,
  showSchedulePopover,
  hideSchedulePopover,
  focusFirstPendingSchedule
} = useScheduleCalendarView({
  visibleSchedules,
  pendingSchedules,
  getSchedulesForDate,
  getScheduleDayFlags,
  formatDateKey,
  formatDateLabel
})

const { downloadGoogleCalendarIcs } = useScheduleIcsExport({
  visibleSchedules,
  formatDateKey,
  getTypeLabel
})

function handleConfirm(item) {
  confirmSchedule(item.id)
  selectedDateKey.value = item.dateKey
}

function handleIgnore(item) {
  ignoreSchedule(item.id)
}

function openWorkspace(item) {
  if (item.status === 'pending') confirmSchedule(item.id)
  emit('navigate', 'workspace')
}

onMounted(async () => {
  await hydrateSchedules()
  focusFirstPendingSchedule()
})
</script>

<template>
  <div class="schedule-page">
    <InfiniteGrid />
    <HomeSidebar
      class="relative z-10"
      :fileTree="fileTree"
      :favorites="favorites"
      @navigate="emit('navigate', $event)"
    />

    <main class="schedule-shell">
      <section class="schedule-main-card">
        <header class="schedule-header">
          <div class="schedule-header-left">
            <button class="schedule-back-btn" @click="emit('navigate', 'home')">
              <span class="material-symbols-outlined">arrow_back</span>
            </button>
          </div>
          <div class="schedule-header-actions">
            <button class="schedule-soft-btn export" @click="downloadGoogleCalendarIcs">
              <span class="material-symbols-outlined">ios_share</span>
              Google 캘린더
            </button>
            <button class="schedule-soft-btn" @click="moveToday">오늘</button>
            <button class="schedule-icon-btn" @click="movePeriod(-1)">
              <span class="material-symbols-outlined">chevron_left</span>
            </button>
            <button class="schedule-icon-btn" @click="movePeriod(1)">
              <span class="material-symbols-outlined">chevron_right</span>
            </button>
          </div>
        </header>

        <ScheduleCalendarPanel
          :calendarView="calendarView"
          :calendarViewOptions="calendarViewOptions"
          :currentPeriodLabel="currentPeriodLabel"
          :calendarDays="calendarDays"
          :weekDays="weekDays"
          :hourSlots="hourSlots"
          :selectedDateKey="selectedDateKey"
          :selectedSchedules="selectedSchedules"
          :getWeekEventStyle="getWeekEventStyle"
          :formatHourSlot="formatHourSlot"
          :getTypeLabel="getTypeLabel"
          @set-calendar-view="setCalendarView"
          @select-date="selectDate"
          @focus-schedule="focusSchedule"
          @show-popover="showSchedulePopover"
          @hide-popover="hideSchedulePopover"
        />
      </section>

      <ScheduleSidebar
        :selectedDateKey="selectedDateKey"
        :selectedSchedules="selectedSchedules"
        :pendingSchedules="pendingSchedules"
        :formatDateLabel="formatDateLabel"
        :getTypeLabel="getTypeLabel"
        :getTypeIcon="getTypeIcon"
        :getStatusLabel="getStatusLabel"
        @confirm="handleConfirm"
        @ignore="handleIgnore"
        @open-workspace="openWorkspace"
        @focus-schedule="focusSchedule"
      />
    </main>

    <ScheduleHoverPopover
      :hoveredSchedule="hoveredSchedule"
      :formatDateLabel="formatDateLabel"
      :getTypeLabel="getTypeLabel"
      :getTypeIcon="getTypeIcon"
      :getStatusLabel="getStatusLabel"
    />
  </div>
</template>

<style scoped>
.schedule-page {
  position: relative;
  display: flex;
  width: 100vw;
  height: 100vh;
  min-width: 0;
  overflow: hidden;
  color: var(--copy-text);
  background: var(--copy-bg);
}

.schedule-page > :deep(.infinite-grid-container) {
  opacity: 0;
}

.schedule-shell {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 8fr) minmax(320px, 2fr);
  gap: 12px;
  flex: 1;
  width: auto;
  height: 100vh;
  padding: 28px 30px;
  box-sizing: border-box;
}

.schedule-main-card {
  min-height: 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: visible;
  border-radius: 0;
  background: transparent;
  border: 0;
  box-shadow: none;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}

.schedule-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 12px;
}

.schedule-header-left {
  display: flex;
  align-items: center;
}

.schedule-back-btn,
.schedule-icon-btn,
.schedule-soft-btn {
  border: 1px solid var(--copy-line);
  background: var(--copy-surface-soft);
  color: var(--copy-text);
  font-weight: 900;
}

.schedule-back-btn,
.schedule-icon-btn {
  width: 42px;
  height: 42px;
  border-radius: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.schedule-back-btn {
  margin-bottom: 0;
}

.schedule-soft-btn {
  height: 42px;
  border-radius: 14px;
  padding: 0 16px;
}

.schedule-soft-btn.export {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.schedule-soft-btn.export .material-symbols-outlined {
  font-size: 17px;
}

.schedule-header-actions {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

@media (max-width: 1100px) {
  .schedule-shell {
    grid-template-columns: 1fr;
    overflow-y: auto;
  }

  .schedule-main-card {
    overflow: visible;
  }
}

@media (max-width: 720px) {
  .schedule-shell {
    padding: 10px;
  }

  .schedule-main-card {
    padding: 16px;
    border-radius: 20px;
  }

  .schedule-header {
    flex-direction: column;
  }
}
</style>
