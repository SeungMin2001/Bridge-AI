import { computed, ref } from 'vue'

const STORAGE_KEY = 'lecto_home_calendar_schedules'

const scheduleItems = ref([])
const hasLoadedSchedules = ref(false)

const TYPE_LABELS = {
  lecture: '수업',
  meeting: '회의',
  assignment: '과제',
  exam: '시험',
  etc: '기타'
}

const TYPE_ICONS = {
  lecture: 'menu_book',
  meeting: 'groups_2',
  assignment: 'assignment',
  exam: 'quiz',
  etc: 'event'
}

const STATUS_LABELS = {
  pending: '확인 필요',
  confirmed: '예정',
  ignored: '무시됨'
}

const formatDateKey = (date = new Date()) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

const parseDateKey = (dateKey) => {
  const [year, month, day] = String(dateKey).split('-').map(Number)
  return new Date(year, month - 1, day)
}

const addDays = (date, days) => {
  const nextDate = new Date(date)
  nextDate.setDate(nextDate.getDate() + days)
  return nextDate
}

const getWeekStartDate = (date = new Date()) => {
  const nextDate = new Date(date)
  const day = nextDate.getDay()
  const diff = day === 0 ? -6 : 1 - day
  nextDate.setDate(nextDate.getDate() + diff)
  nextDate.setHours(0, 0, 0, 0)
  return nextDate
}

const getWeekKeyFromDateKey = (dateKey) => formatDateKey(getWeekStartDate(parseDateKey(dateKey)))

const formatDateLabel = (dateKey) => {
  const [year, month, day] = String(dateKey).split('-').map(Number)
  return `${year}년 ${month}월 ${day}일`
}

const createLocalId = (prefix) => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

const normalizeStatus = (status, origin = 'manual') => {
  if (status === 'pending' || status === 'confirmed' || status === 'ignored') return status
  if (status === '예정') return 'confirmed'
  return origin === 'ai' ? 'pending' : 'confirmed'
}

const normalizeScheduleItem = (item) => {
  const origin = item.origin || (item.sourceText ? 'ai' : 'manual')
  const dateKey = item.dateKey || formatDateKey()
  const startTime = item.startTime || ''
  const endTime = item.endTime || ''
  const time = item.time || [startTime, endTime].filter(Boolean).join(' - ')

  return {
    ...item,
    id: item.id || createLocalId(origin === 'ai' ? 'ai-schedule' : 'schedule'),
    origin,
    type: item.type || item.fileKind || (origin === 'ai' ? 'meeting' : 'etc'),
    dateKey,
    weekKey: item.weekKey || getWeekKeyFromDateKey(dateKey),
    startTime,
    endTime,
    time,
    note: item.note || '',
    sourceText: item.sourceText || '',
    sourceSessionTitle: item.sourceSessionTitle || '',
    workspaceFileId: item.workspaceFileId || '',
    confidence: typeof item.confidence === 'number' ? item.confidence : null,
    status: normalizeStatus(item.status, origin)
  }
}

const createSampleAiSchedules = () => {
  const today = new Date()
  const lectureDateKey = formatDateKey(addDays(today, 1))
  const meetingDateKey = formatDateKey(addDays(today, 2))
  const assignmentDateKey = formatDateKey(addDays(today, 3))

  return [
    {
      id: `sample-ai-lecture-${lectureDateKey}`,
      origin: 'ai',
      type: 'lecture',
      dateKey: lectureDateKey,
      title: '운영체제 보강 수업',
      startTime: '오전 10:00',
      endTime: '오전 11:30',
      time: '오전 10:00 - 오전 11:30',
      status: 'pending',
      note: '전사 중 보강 수업 일정으로 감지됨',
      sourceText: '내일 오전 10시에 운영체제 보강 수업을 진행하겠습니다.',
      sourceSessionTitle: '운영체제론 4주차 실시간 전사',
      workspaceFileId: '',
      confidence: 0.91
    },
    {
      id: `sample-ai-meeting-${meetingDateKey}`,
      origin: 'ai',
      type: 'meeting',
      dateKey: meetingDateKey,
      title: '팀 프로젝트 회의',
      startTime: '오후 03:00',
      endTime: '오후 04:00',
      time: '오후 03:00 - 오후 04:00',
      status: 'pending',
      note: '회의 발화에서 자동 추출된 일정 후보',
      sourceText: '이번 주 수요일 오후 3시에 팀 프로젝트 회의 잡을게요.',
      sourceSessionTitle: '캡스톤 회의 녹음',
      workspaceFileId: '',
      confidence: 0.88
    },
    {
      id: `sample-ai-assignment-${assignmentDateKey}`,
      origin: 'ai',
      type: 'assignment',
      dateKey: assignmentDateKey,
      title: '발표자료 초안 제출',
      startTime: '오후 11:55',
      endTime: '오후 11:55',
      time: '오후 11:55',
      status: 'pending',
      note: '마감 일정으로 감지됨',
      sourceText: '발표자료 초안은 이번 주 목요일까지 올려주세요.',
      sourceSessionTitle: '프로젝트 회의 녹음',
      workspaceFileId: '',
      confidence: 0.84
    }
  ].map(normalizeScheduleItem)
}

const persistSchedules = (items) => {
  scheduleItems.value = items.map(normalizeScheduleItem)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(scheduleItems.value))
}

const mergeSampleSchedules = (items) => {
  const existingIds = new Set(items.map((item) => item.id))
  const samples = createSampleAiSchedules().filter((item) => !existingIds.has(item.id))
  return [...items, ...samples]
}

const sortSchedules = (items) => {
  return [...items].sort((a, b) => {
    const dateCompare = a.dateKey.localeCompare(b.dateKey)
    if (dateCompare !== 0) return dateCompare
    return String(a.startTime || '').localeCompare(String(b.startTime || ''))
  })
}

export function useScheduleState() {
  const hydrateSchedules = () => {
    if (hasLoadedSchedules.value) return

    const raw = localStorage.getItem(STORAGE_KEY)
    let loadedItems = []

    if (raw) {
      try {
        loadedItems = JSON.parse(raw).map(normalizeScheduleItem)
      } catch {
        loadedItems = []
      }
    }

    scheduleItems.value = mergeSampleSchedules(loadedItems)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(scheduleItems.value))
    hasLoadedSchedules.value = true
  }

  const visibleSchedules = computed(() => {
    return sortSchedules(scheduleItems.value.filter((item) => item.status !== 'ignored'))
  })

  const pendingSchedules = computed(() => {
    return sortSchedules(visibleSchedules.value.filter((item) => item.status === 'pending'))
  })

  const getSchedulesForDate = (dateKey) => {
    return visibleSchedules.value.filter((item) => item.dateKey === dateKey)
  }

  const getScheduleDayFlags = (dateKey) => {
    const items = getSchedulesForDate(dateKey)
    return {
      hasSchedule: items.length > 0,
      hasPendingSchedule: items.some((item) => item.status === 'pending'),
      hasConfirmedSchedule: items.some((item) => item.status === 'confirmed')
    }
  }

  const addManualSchedule = (schedule) => {
    const item = normalizeScheduleItem({
      ...schedule,
      id: createLocalId('schedule'),
      origin: 'manual',
      status: 'confirmed'
    })
    persistSchedules([...scheduleItems.value, item])
    return item
  }

  const updateScheduleStatus = (scheduleId, status) => {
    persistSchedules(scheduleItems.value.map((item) => (
      item.id === scheduleId
        ? normalizeScheduleItem({ ...item, status })
        : item
    )))
  }

  const confirmSchedule = (scheduleId) => updateScheduleStatus(scheduleId, 'confirmed')
  const ignoreSchedule = (scheduleId) => updateScheduleStatus(scheduleId, 'ignored')
  const updateScheduleSync = (scheduleId, updates) => {
    persistSchedules(scheduleItems.value.map((item) => (
      item.id === scheduleId
        ? normalizeScheduleItem({ ...item, ...updates })
        : item
    )))
  }

  const getTypeLabel = (type) => TYPE_LABELS[type] || TYPE_LABELS.etc
  const getTypeIcon = (type) => TYPE_ICONS[type] || TYPE_ICONS.etc
  const getStatusLabel = (status) => STATUS_LABELS[status] || STATUS_LABELS.confirmed
  const getConfidenceLabel = (confidence) => (
    typeof confidence === 'number' ? `${Math.round(confidence * 100)}%` : ''
  )

  return {
    scheduleItems,
    visibleSchedules,
    pendingSchedules,
    hydrateSchedules,
    addManualSchedule,
    confirmSchedule,
    ignoreSchedule,
    updateScheduleSync,
    getSchedulesForDate,
    getScheduleDayFlags,
    formatDateKey,
    formatDateLabel,
    getWeekKeyFromDateKey,
    getTypeLabel,
    getTypeIcon,
    getStatusLabel,
    getConfidenceLabel
  }
}
