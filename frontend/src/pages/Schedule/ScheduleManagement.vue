<!-- AI가 감지한 일정과 확정 일정을 큰 캘린더에서 관리하는 페이지입니다. -->
<script setup>
import { computed, onMounted, ref } from 'vue'
import InfiniteGrid from '../../components/home/InfiniteGrid.vue'
import { useScheduleState } from '../../composables/useScheduleState'

const emit = defineEmits(['navigate'])

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
  getConfidenceLabel
} = useScheduleState()

const today = new Date()
const activeMonthDate = ref(new Date(today.getFullYear(), today.getMonth(), 1))
const selectedDateKey = ref(formatDateKey(today))

const currentMonthLabel = computed(() => {
  const date = activeMonthDate.value
  return `${date.getFullYear()}년 ${date.getMonth() + 1}월`
})

const selectedSchedules = computed(() => getSchedulesForDate(selectedDateKey.value))

const confirmedSchedules = computed(() => {
  return visibleSchedules.value.filter((item) => item.status === 'confirmed')
})

const currentMonthPrefix = computed(() => {
  const date = activeMonthDate.value
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
})

const upcomingSchedules = computed(() => {
  const todayKey = formatDateKey(today)
  return visibleSchedules.value.filter((item) => item.dateKey >= todayKey).slice(0, 8)
})

const scheduleStats = computed(() => [
  { label: 'AI 후보', value: pendingSchedules.value.length },
  { label: '확정 일정', value: confirmedSchedules.value.length },
  { label: '이번 달', value: visibleSchedules.value.filter((item) => item.dateKey.startsWith(currentMonthPrefix.value)).length }
])

const parseDateKey = (dateKey) => {
  const [year, month, day] = String(dateKey).split('-').map(Number)
  return new Date(year, month - 1, day)
}

const calendarDays = computed(() => {
  const baseDate = activeMonthDate.value
  const year = baseDate.getFullYear()
  const month = baseDate.getMonth()
  const firstDay = new Date(year, month, 1)
  const lastDay = new Date(year, month + 1, 0)
  const prevLastDay = new Date(year, month, 0)
  const leadingCount = firstDay.getDay()
  const trailingCount = 6 - lastDay.getDay()
  const days = []

  for (let i = leadingCount - 1; i >= 0; i -= 1) {
    const date = new Date(year, month - 1, prevLastDay.getDate() - i)
    const dateKey = formatDateKey(date)
    days.push({ label: prevLastDay.getDate() - i, dateKey, muted: true, isToday: dateKey === formatDateKey(today), schedules: getSchedulesForDate(dateKey), ...getScheduleDayFlags(dateKey) })
  }

  for (let date = 1; date <= lastDay.getDate(); date += 1) {
    const currentDate = new Date(year, month, date)
    const dateKey = formatDateKey(currentDate)
    days.push({ label: date, dateKey, muted: false, isToday: dateKey === formatDateKey(today), schedules: getSchedulesForDate(dateKey), ...getScheduleDayFlags(dateKey) })
  }

  for (let date = 1; date <= trailingCount; date += 1) {
    const nextDate = new Date(year, month + 1, date)
    const dateKey = formatDateKey(nextDate)
    days.push({ label: date, dateKey, muted: true, isToday: dateKey === formatDateKey(today), schedules: getSchedulesForDate(dateKey), ...getScheduleDayFlags(dateKey) })
  }

  return days
})

function moveMonth(offset) {
  const nextDate = new Date(activeMonthDate.value)
  nextDate.setMonth(nextDate.getMonth() + offset)
  activeMonthDate.value = nextDate
}

function moveToday() {
  activeMonthDate.value = new Date(today.getFullYear(), today.getMonth(), 1)
  selectedDateKey.value = formatDateKey(today)
}

function selectDate(day) {
  selectedDateKey.value = day.dateKey
  if (day.muted) {
    const date = parseDateKey(day.dateKey)
    activeMonthDate.value = new Date(date.getFullYear(), date.getMonth(), 1)
  }
}

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

function escapeIcsText(value = '') {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/,/g, '\\,')
    .replace(/;/g, '\\;')
    .replace(/\n/g, '\\n')
}

function parseKoreanTime(timeText = '오전 09:00') {
  const match = String(timeText).match(/(오전|오후)\s*(\d{1,2}):(\d{2})/)
  if (!match) return { hour: 9, minute: 0 }

  const [, meridiem, rawHour, rawMinute] = match
  let hour = Number(rawHour)
  const minute = Number(rawMinute)

  if (meridiem === '오전' && hour === 12) hour = 0
  if (meridiem === '오후' && hour !== 12) hour += 12

  return { hour, minute }
}

function formatIcsDateTime(dateKey, timeText) {
  const [year, month, day] = String(dateKey).split('-')
  const { hour, minute } = parseKoreanTime(timeText)
  return `${year}${month}${day}T${String(hour).padStart(2, '0')}${String(minute).padStart(2, '0')}00`
}

function getCalendarExportItems() {
  return visibleSchedules.value.filter((item) => item.status === 'confirmed')
}

function downloadGoogleCalendarIcs() {
  const items = getCalendarExportItems()
  if (!items.length) return

  const nowStamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
  const events = items.map((item) => {
    const dtStart = formatIcsDateTime(item.dateKey, item.startTime || item.time)
    const dtEnd = formatIcsDateTime(item.dateKey, item.endTime || item.startTime || item.time)
    const description = [
      item.note,
      item.sourceText ? `원문: ${item.sourceText}` : '',
      item.sourceSessionTitle ? `출처: ${item.sourceSessionTitle}` : '',
      `LectoAI 타입: ${getTypeLabel(item.type)}`
    ].filter(Boolean).join('\n')

    return [
      'BEGIN:VEVENT',
      `UID:${item.id}@lectoai.local`,
      `DTSTAMP:${nowStamp}`,
      `DTSTART;TZID=Asia/Seoul:${dtStart}`,
      `DTEND;TZID=Asia/Seoul:${dtEnd}`,
      `SUMMARY:${escapeIcsText(item.title)}`,
      `DESCRIPTION:${escapeIcsText(description)}`,
      'END:VEVENT'
    ].join('\r\n')
  })

  const ics = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//LectoAI//Schedule Export//KO',
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    'X-WR-CALNAME:LectoAI 일정',
    ...events,
    'END:VCALENDAR'
  ].join('\r\n')

  const blob = new Blob([ics], { type: 'text/calendar;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `lectoai-schedules-${formatDateKey(today)}.ics`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

onMounted(() => {
  hydrateSchedules()
  if (pendingSchedules.value.length > 0) {
    selectedDateKey.value = pendingSchedules.value[0].dateKey
    const date = parseDateKey(pendingSchedules.value[0].dateKey)
    activeMonthDate.value = new Date(date.getFullYear(), date.getMonth(), 1)
  }
})
</script>

<template>
  <div class="schedule-page">
    <InfiniteGrid />

    <main class="schedule-shell">
      <section class="schedule-main-card">
        <header class="schedule-header">
          <div>
            <button class="schedule-back-btn" @click="emit('navigate', 'home')">
              <span class="material-symbols-outlined">arrow_back</span>
            </button>
            <div class="schedule-kicker">LectoAI Calendar</div>
            <h1>일정관리</h1>
          </div>
          <div class="schedule-header-actions">
            <button class="schedule-soft-btn export" @click="downloadGoogleCalendarIcs">
              <span class="material-symbols-outlined">ios_share</span>
              Google 캘린더
            </button>
            <button class="schedule-soft-btn" @click="moveToday">오늘</button>
            <button class="schedule-icon-btn" @click="moveMonth(-1)">
              <span class="material-symbols-outlined">chevron_left</span>
            </button>
            <button class="schedule-icon-btn" @click="moveMonth(1)">
              <span class="material-symbols-outlined">chevron_right</span>
            </button>
          </div>
        </header>

        <div class="schedule-stats-row">
          <article v-for="stat in scheduleStats" :key="stat.label" class="schedule-stat">
            <span>{{ stat.label }}</span>
            <strong>{{ stat.value }}</strong>
          </article>
        </div>

        <section class="schedule-calendar-panel">
          <div class="schedule-calendar-title-row">
            <h2>{{ currentMonthLabel }}</h2>
            <div class="schedule-legend">
              <span><i class="confirmed"></i>확정</span>
              <span><i class="pending"></i>AI 후보</span>
            </div>
          </div>

          <div class="schedule-weekdays">
            <span v-for="label in ['일', '월', '화', '수', '목', '금', '토']" :key="label">{{ label }}</span>
          </div>

          <div class="schedule-calendar-grid">
            <button
              v-for="day in calendarDays"
              :key="day.dateKey"
              class="schedule-day-cell"
              :class="{ muted: day.muted, today: day.isToday, selected: day.dateKey === selectedDateKey, pending: day.hasPendingSchedule, confirmed: day.hasConfirmedSchedule }"
              @click="selectDate(day)"
            >
              <span class="schedule-day-number">{{ day.label }}</span>
              <div class="schedule-day-items">
                <span
                  v-for="item in day.schedules.slice(0, 2)"
                  :key="`${day.dateKey}-${item.id}`"
                  class="schedule-day-pill"
                  :class="item.status"
                >
                  {{ item.title }}
                </span>
              </div>
              <span v-if="day.schedules.length > 2" class="schedule-day-more">+{{ day.schedules.length - 2 }}</span>
            </button>
          </div>
        </section>
      </section>

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
                  <button class="schedule-primary-btn" @click="handleConfirm(item)">확정</button>
                  <button class="schedule-secondary-btn" @click="handleIgnore(item)">무시</button>
                </template>
                <template v-else>
                  <button class="schedule-primary-btn" @click="openWorkspace(item)">워크스페이스 열기</button>
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
              @click="selectedDateKey = item.dateKey"
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
              @click="selectedDateKey = item.dateKey"
            >
              <div>
                <strong>{{ item.title }}</strong>
                <span>{{ formatDateLabel(item.dateKey) }} · {{ getTypeLabel(item.type) }}</span>
              </div>
            </article>
          </div>
        </section>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.schedule-page {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  color: #1d1d1f;
}

.schedule-shell {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 380px;
  gap: 14px;
  width: 100%;
  height: 100%;
  padding: 14px;
}

.schedule-main-card,
.schedule-side-card {
  min-height: 0;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.9);
  box-shadow: 0 24px 48px rgba(148, 163, 184, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.schedule-main-card {
  display: flex;
  flex-direction: column;
  padding: 28px;
  overflow: hidden;
}

.schedule-side-card {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 22px;
  overflow-y: auto;
}

.schedule-header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 20px;
}

.schedule-back-btn,
.schedule-icon-btn,
.schedule-soft-btn {
  border: 1px solid rgba(226, 213, 195, 0.72);
  background: #f4ede4;
  color: #1d1d1f;
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
  margin-bottom: 16px;
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

.schedule-kicker {
  font-size: 12px;
  font-weight: 900;
  color: #6b7280;
  margin-bottom: 6px;
}

.schedule-header h1 {
  font-size: 42px;
  font-weight: 950;
  letter-spacing: 0;
}

.schedule-stats-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}

.schedule-stat {
  padding: 14px 16px;
  border-radius: 18px;
  background: #f4ede4;
  border: 1px solid rgba(255, 255, 255, 0.86);
}

.schedule-stat span {
  display: block;
  font-size: 12px;
  font-weight: 800;
  color: #6b7280;
}

.schedule-stat strong {
  font-size: 28px;
  font-weight: 950;
}

.schedule-calendar-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  flex: 1;
  padding: 18px;
  border-radius: 24px;
  background: #f4ede4;
  border: 1px solid rgba(255, 255, 255, 0.86);
}

.schedule-calendar-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}

.schedule-calendar-title-row h2 {
  font-size: 26px;
  font-weight: 950;
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
  background: #2fc07a;
}

.schedule-legend .pending {
  background: #f59e0b;
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
  padding: 0 8px;
  color: #6b7280;
  font-size: 12px;
  font-weight: 900;
}

.schedule-calendar-grid {
  flex: 1;
  min-height: 0;
  gap: 8px;
}

.schedule-day-cell {
  position: relative;
  min-height: 96px;
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 16px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.48);
  text-align: left;
  overflow: hidden;
}

.schedule-day-cell:hover,
.schedule-day-cell.selected {
  background: rgba(255, 255, 255, 0.84);
  border-color: rgba(47, 192, 122, 0.46);
}

.schedule-day-cell.muted {
  opacity: 0.45;
}

.schedule-day-cell.today .schedule-day-number {
  background: #1d1d1f;
  color: white;
}

.schedule-day-number {
  width: 26px;
  height: 26px;
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
  background: #2fc07a;
}

.schedule-day-cell.pending::after {
  right: 22px;
  background: #f59e0b;
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
  border-radius: 999px;
  padding: 4px 7px;
  font-size: 10px;
  font-weight: 900;
}

.schedule-day-pill.confirmed {
  background: rgba(47, 192, 122, 0.12);
  color: #15803d;
}

.schedule-day-pill.pending {
  background: rgba(245, 158, 11, 0.14);
  color: #92400e;
}

.schedule-day-more {
  display: inline-block;
  color: #6b7280;
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
  .schedule-shell {
    grid-template-columns: 1fr;
    overflow-y: auto;
  }

  .schedule-main-card,
  .schedule-side-card {
    overflow: visible;
  }
}

@media (max-width: 720px) {
  .schedule-shell {
    padding: 10px;
  }

  .schedule-main-card,
  .schedule-side-card {
    padding: 16px;
    border-radius: 20px;
  }

  .schedule-header {
    flex-direction: column;
  }

  .schedule-header h1 {
    font-size: 32px;
  }

  .schedule-stats-row {
    grid-template-columns: 1fr;
  }

  .schedule-day-cell {
    min-height: 72px;
    padding: 7px;
  }

  .schedule-day-pill {
    display: none;
  }
}
</style>
