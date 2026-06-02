import { computed, ref } from 'vue'

const CALENDAR_VIEW_OPTIONS = [
  { value: 'week', label: 'Week' },
  { value: 'day', label: 'Day' }
]

const WEEK_START_HOUR = 8
const WEEK_END_HOUR = 22
const WEEK_HOUR_HEIGHT = 68
const HOUR_SLOTS = Array.from({ length: WEEK_END_HOUR - WEEK_START_HOUR + 1 }, (_, index) => WEEK_START_HOUR + index)

const parseDateKey = (dateKey) => {
  const [year, month, day] = String(dateKey).split('-').map(Number)
  return new Date(year, month - 1, day)
}

const addDays = (date, days) => {
  const nextDate = new Date(date)
  nextDate.setDate(nextDate.getDate() + days)
  return nextDate
}

const getWeekStartDate = (date) => {
  const nextDate = new Date(date)
  const day = nextDate.getDay()
  const diff = day === 0 ? -6 : 1 - day
  nextDate.setDate(nextDate.getDate() + diff)
  nextDate.setHours(0, 0, 0, 0)
  return nextDate
}

export function useScheduleCalendarView({
  visibleSchedules,
  pendingSchedules,
  getSchedulesForDate,
  getScheduleDayFlags,
  formatDateKey,
  formatDateLabel
}) {
  const today = new Date()
  const activeMonthDate = ref(new Date(today.getFullYear(), today.getMonth(), 1))
  const selectedDateKey = ref(formatDateKey(today))
  const calendarView = ref('week')
  const hoveredSchedule = ref(null)

  const currentMonthLabel = computed(() => {
    const date = activeMonthDate.value
    return `${date.getFullYear()}년 ${date.getMonth() + 1}월`
  })

  const selectedDate = computed(() => parseDateKey(selectedDateKey.value))
  const selectedSchedules = computed(() => getSchedulesForDate(selectedDateKey.value))

  const upcomingSchedules = computed(() => {
    const todayKey = formatDateKey(today)
    return visibleSchedules.value.filter((item) => item.dateKey >= todayKey).slice(0, 8)
  })

  const weekDays = computed(() => {
    const weekStartDate = getWeekStartDate(selectedDate.value)
    return Array.from({ length: 7 }, (_, index) => {
      const date = addDays(weekStartDate, index)
      const dateKey = formatDateKey(date)
      return {
        label: date.getDate(),
        dayName: ['월', '화', '수', '목', '금', '토', '일'][index],
        dateKey,
        muted: false,
        isToday: dateKey === formatDateKey(today),
        schedules: getSchedulesForDate(dateKey),
        ...getScheduleDayFlags(dateKey)
      }
    })
  })

  const currentPeriodLabel = computed(() => {
    if (calendarView.value === 'month') return currentMonthLabel.value

    if (calendarView.value === 'week') {
      const startDate = weekDays.value[0]
      const endDate = weekDays.value[6]
      return `${formatDateLabel(startDate.dateKey)} - ${formatDateLabel(endDate.dateKey)}`
    }

    return formatDateLabel(selectedDateKey.value)
  })

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

  function parseScheduleTimeToMinutes(timeText = '') {
    const match = String(timeText).match(/(오전|오후)\s*(\d{1,2}):(\d{2})/)
    if (!match) return WEEK_START_HOUR * 60

    const [, meridiem, rawHour, rawMinute] = match
    let hour = Number(rawHour)
    const minute = Number(rawMinute)
    if (meridiem === '오전' && hour === 12) hour = 0
    if (meridiem === '오후' && hour !== 12) hour += 12
    return hour * 60 + minute
  }

  function getWeekEventStyle(item) {
    const startMinutes = parseScheduleTimeToMinutes(item.startTime || item.time)
    const endMinutes = parseScheduleTimeToMinutes(item.endTime || item.startTime || item.time)
    const clampedStart = Math.max(startMinutes, WEEK_START_HOUR * 60)
    const clampedEnd = Math.min(Math.max(endMinutes, clampedStart + 60), (WEEK_END_HOUR + 1) * 60)
    const top = ((clampedStart - WEEK_START_HOUR * 60) / 60) * WEEK_HOUR_HEIGHT
    const height = Math.max(((clampedEnd - clampedStart) / 60) * WEEK_HOUR_HEIGHT - 8, 44)

    return {
      top: `${top + 6}px`,
      height: `${height}px`
    }
  }

  function formatHourSlot(hour) {
    const meridiem = hour < 12 ? '오전' : '오후'
    let displayHour = hour % 12
    if (displayHour === 0) displayHour = 12
    return `${meridiem} ${displayHour}시`
  }

  function syncActiveMonth(date) {
    activeMonthDate.value = new Date(date.getFullYear(), date.getMonth(), 1)
  }

  function moveMonth(offset) {
    const nextDate = new Date(activeMonthDate.value)
    nextDate.setMonth(nextDate.getMonth() + offset)
    activeMonthDate.value = nextDate
  }

  function setCalendarView(view) {
    calendarView.value = view
    if (view !== 'month') {
      syncActiveMonth(selectedDate.value)
    }
  }

  function movePeriod(offset) {
    if (calendarView.value === 'month') {
      moveMonth(offset)
      return
    }

    const dayOffset = calendarView.value === 'week' ? offset * 7 : offset
    const nextDate = addDays(selectedDate.value, dayOffset)
    selectedDateKey.value = formatDateKey(nextDate)
    syncActiveMonth(nextDate)
  }

  function moveToday() {
    activeMonthDate.value = new Date(today.getFullYear(), today.getMonth(), 1)
    selectedDateKey.value = formatDateKey(today)
  }

  function selectDate(day) {
    selectedDateKey.value = day.dateKey
    if (day.muted) {
      syncActiveMonth(parseDateKey(day.dateKey))
    }
  }

  function focusSchedule(item) {
    selectedDateKey.value = item.dateKey
    syncActiveMonth(parseDateKey(item.dateKey))
  }

  function showSchedulePopover(item, event) {
    const rect = event.currentTarget.getBoundingClientRect()
    const width = 286
    const margin = 14
    const left = Math.min(
      Math.max(rect.left + rect.width / 2 - width / 2, margin),
      window.innerWidth - width - margin
    )
    const top = rect.bottom + 10 > window.innerHeight - 180
      ? Math.max(rect.top - 178, margin)
      : rect.bottom + 10

    hoveredSchedule.value = {
      item,
      left,
      top,
      description: item.note || item.sourceText || '설명이 없습니다.'
    }
  }

  function hideSchedulePopover() {
    hoveredSchedule.value = null
  }

  function focusFirstPendingSchedule() {
    if (pendingSchedules.value.length === 0) return
    selectedDateKey.value = pendingSchedules.value[0].dateKey
    syncActiveMonth(parseDateKey(pendingSchedules.value[0].dateKey))
  }

  return {
    calendarView,
    calendarViewOptions: CALENDAR_VIEW_OPTIONS,
    hourSlots: HOUR_SLOTS,
    hoveredSchedule,
    selectedDateKey,
    selectedSchedules,
    upcomingSchedules,
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
  }
}
