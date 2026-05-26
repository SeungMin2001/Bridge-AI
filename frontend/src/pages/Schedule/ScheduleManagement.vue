<!-- AI가 감지한 일정과 확정 일정을 큰 캘린더에서 관리하는 페이지입니다. -->
<script setup>
import { onMounted, ref } from 'vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import HomeSidebar from '../../components/home/HomeSidebar.vue'
import ScheduleCalendarPanel from '../../components/schedule/ScheduleCalendarPanel.vue'
import ScheduleHoverPopover from '../../components/schedule/ScheduleHoverPopover.vue'
import ScheduleSidebar from '../../components/schedule/ScheduleSidebar.vue'
import { useScheduleCalendarView } from '../../composables/schedule/useScheduleCalendarView'
import { useScheduleState } from '../../composables/useScheduleState'

const emit = defineEmits(['navigate', 'openWorkspace'])

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
  syncConfirmedSchedulesToNotion,
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
  hourSlots,
  hoveredSchedule,
  selectedDateKey,
  selectedSchedules,
  currentPeriodLabel,
  calendarDays,
  weekDays,
  getWeekEventStyle,
  formatHourSlot,
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

const isNotionSyncing = ref(false)
const notionToast = ref('')
let notionToastTimer = null

function showNotionToast(message) {
  notionToast.value = message
  if (notionToastTimer) window.clearTimeout(notionToastTimer)
  notionToastTimer = window.setTimeout(() => {
    notionToast.value = ''
  }, 2400)
}

function showConfirmSyncToast(result) {
  if (result?.status === 'success') {
    showNotionToast('노션에 추가하였습니다.')
    return
  }
  if (result?.status === 'already_synced') {
    showNotionToast('이미 노션에 추가된 일정입니다.')
    return
  }
  if (result?.status === 'notion_failed') {
    showNotionToast('일정은 확정했지만 노션 추가에 실패했습니다.')
  }
}

async function handleConfirm(item) {
  const result = await confirmSchedule(item.id)
  selectedDateKey.value = item.dateKey
  showConfirmSyncToast(result)
}

function handleIgnore(item) {
  ignoreSchedule(item.id)
}

async function openWorkspace(item) {
  if (item.status === 'pending') {
    const result = await confirmSchedule(item.id)
    showConfirmSyncToast(result)
  }
  emit('openWorkspace', item)
}

async function syncNotionSchedules() {
  if (isNotionSyncing.value) return

  isNotionSyncing.value = true
  try {
    const result = await syncConfirmedSchedulesToNotion()
    if (result.synced_count > 0) {
      showNotionToast(`노션에 ${result.synced_count}개 일정을 추가하였습니다.`)
    } else {
      showNotionToast('노션에 추가할 새 일정이 없습니다.')
    }
  } catch (error) {
    console.error('[schedule] notion bulk sync failed:', error)
    showNotionToast('노션 저장에 실패했습니다.')
  } finally {
    isNotionSyncing.value = false
  }
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
            <button class="schedule-soft-btn export" :disabled="isNotionSyncing" @click="syncNotionSchedules">
              <span class="material-symbols-outlined">database</span>
              {{ isNotionSyncing ? '노션 저장 중' : '노션에 저장' }}
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
          :currentPeriodLabel="currentPeriodLabel"
          :calendarDays="calendarDays"
          :weekDays="weekDays"
          :hourSlots="hourSlots"
          :selectedDateKey="selectedDateKey"
          :selectedSchedules="selectedSchedules"
          :getWeekEventStyle="getWeekEventStyle"
          :formatHourSlot="formatHourSlot"
          :getTypeLabel="getTypeLabel"
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

    <transition name="schedule-toast">
      <div v-if="notionToast" class="schedule-notion-toast">
        <span class="material-symbols-outlined">check_circle</span>
        {{ notionToast }}
      </div>
    </transition>
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

.schedule-soft-btn:disabled {
  cursor: wait;
  opacity: 0.68;
}

.schedule-soft-btn.export {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.schedule-soft-btn.export .material-symbols-outlined {
  font-size: 17px;
}

.schedule-notion-toast {
  position: fixed;
  right: 34px;
  bottom: 30px;
  z-index: 80;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  min-height: 46px;
  max-width: min(360px, calc(100vw - 40px));
  padding: 0 16px;
  border-radius: 14px;
  color: #ffffff;
  background: #18181b;
  box-shadow: 0 18px 36px rgba(24, 24, 27, 0.22);
  font-size: 13px;
  font-weight: 900;
}

.schedule-notion-toast .material-symbols-outlined {
  font-size: 18px;
}

.schedule-toast-enter-active,
.schedule-toast-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.schedule-toast-enter-from,
.schedule-toast-leave-to {
  opacity: 0;
  transform: translateY(8px);
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
