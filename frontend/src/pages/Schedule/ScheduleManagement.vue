<!-- AI가 감지한 일정과 확정 일정을 큰 캘린더에서 관리하는 페이지입니다. -->
<script setup>
import { onMounted, ref } from 'vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import ScheduleCalendarPanel from '../../components/schedule/ScheduleCalendarPanel.vue'
import ScheduleHoverPopover from '../../components/schedule/ScheduleHoverPopover.vue'
import ScheduleSidebar from '../../components/schedule/ScheduleSidebar.vue'
import { useScheduleCalendarView } from '../../composables/schedule/useScheduleCalendarView'
import { useScheduleIcsExport } from '../../composables/schedule/useScheduleIcsExport'
import { useScheduleState } from '../../composables/useScheduleState'

const emit = defineEmits(['navigate', 'open-workspace-source'])

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
  getStatusLabel,
  syncNotionExport,
  syncNotionImport,
  notionSyncing
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
  emit('open-workspace-source', item)
}

onMounted(async () => {
  await hydrateSchedules()
  focusFirstPendingSchedule()
})

const notionSyncMessage = ref(null)
let notionSyncTimeout = null

function showNotionSyncMessage(text, type = 'success') {
  notionSyncMessage.value = { text, type }
  clearTimeout(notionSyncTimeout)
  notionSyncTimeout = setTimeout(() => {
    notionSyncMessage.value = null
  }, 3500)
}

async function handleNotionExport() {
  try {
    const result = await syncNotionExport()
    showNotionSyncMessage(`노션에 ${result?.synced_count || 0}개 일정을 내보냈습니다.`)
  } catch {
    showNotionSyncMessage('노션 내보내기에 실패했습니다.', 'error')
  }
}

async function handleNotionImport() {
  try {
    const result = await syncNotionImport()
    showNotionSyncMessage(`노션에서 ${result?.imported_count || 0}개 일정을 가져왔습니다.`)
  } catch {
    showNotionSyncMessage('노션 불러오기에 실패했습니다.', 'error')
  }
}
</script>

<template>
  <div class="schedule-page">
    <InfiniteGrid />
    <HomeSidebar
      class="relative z-10"
      activeView="schedule"
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
            <div class="schedule-notion-group">
              <button
                class="schedule-soft-btn notion-btn"
                :disabled="notionSyncing"
                @click="handleNotionImport"
              >
                <span class="material-symbols-outlined">cloud_download</span>
                노션 불러오기
              </button>
              <button
                class="schedule-soft-btn notion-btn"
                :disabled="notionSyncing"
                @click="handleNotionExport"
              >
                <span class="material-symbols-outlined">cloud_upload</span>
                노션 내보내기
              </button>
            </div>

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

    <transition name="notion-toast-fade">
      <div
        v-if="notionSyncMessage"
        :class="['schedule-notion-toast', notionSyncMessage.type]"
        @click="notionSyncMessage = null"
      >
        <span class="material-symbols-outlined">
          {{ notionSyncMessage.type === 'error' ? 'error' : 'check_circle' }}
        </span>
        {{ notionSyncMessage.text }}
      </div>
    </transition>

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
  padding: 18px 20px 20px;
  overflow: hidden;
  border-radius: 32px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(255, 255, 255, 0.9);
  box-shadow:
    0 26px 54px rgba(48, 42, 58, 0.08),
    inset 0 1px 0 rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(18px) saturate(130%);
  -webkit-backdrop-filter: blur(18px) saturate(130%);
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



.schedule-header-actions {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.schedule-notion-group {
  display: flex;
  gap: 4px;
  border-right: 1px solid var(--copy-line, #e2e8f0);
  padding-right: 8px;
}

.schedule-soft-btn.notion-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease, opacity 0.18s ease;
}

.schedule-soft-btn.notion-btn .material-symbols-outlined {
  font-size: 17px;
}

.schedule-soft-btn.notion-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.schedule-soft-btn.notion-btn:not(:disabled):hover {
  transform: translateY(-1px);
  background: #e2e8f0;
}

.schedule-notion-toast {
  position: fixed;
  bottom: 28px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 9999;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 22px;
  border-radius: 14px;
  font-size: 14px;
  font-weight: 800;
  cursor: pointer;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(12px);
}

.schedule-notion-toast.success {
  color: #065f46;
  background: rgba(209, 250, 229, 0.95);
  border: 1px solid #6ee7b7;
}

.schedule-notion-toast.error {
  color: #991b1b;
  background: rgba(254, 226, 226, 0.95);
  border: 1px solid #fca5a5;
}

.schedule-notion-toast .material-symbols-outlined {
  font-size: 18px;
}

.notion-toast-fade-enter-active,
.notion-toast-fade-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}

.notion-toast-fade-enter-from,
.notion-toast-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(12px);
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

  .schedule-notion-group {
    border-right: 0;
    padding-right: 0;
  }
}
</style>
