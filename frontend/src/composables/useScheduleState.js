import { computed, ref, readonly } from 'vue'

const STORAGE_KEY = 'lecto_home_calendar_schedules'
const API_BASE = '/schedule'
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

const scheduleItems = ref([])
const hasLoadedSchedules = ref(false)
const notionSyncing = ref(false)

const TYPE_LABELS = {
  lecture: '수업',
  meeting: '회의',
  assignment: '과제',
  exam: '시험',
  presentation: '발표',
  project: '프로젝트',
  etc: '기타'
}

const TYPE_ICONS = {
  lecture: 'menu_book',
  meeting: 'groups_2',
  assignment: 'assignment',
  exam: 'quiz',
  presentation: 'campaign',
  project: 'workspaces',
  etc: 'event'
}

const STATUS_LABELS = {
  pending: '확인 필요',
  confirmed: '예정',
  ignored: '무시됨'
}

const EVENT_TYPE_TO_FRONTEND = {
  수업: 'lecture',
  강의: 'lecture',
  회의: 'meeting',
  과제: 'assignment',
  제출: 'assignment',
  시험: 'exam',
  발표: 'presentation',
  프로젝트: 'project',
  기타: 'etc'
}

const FRONTEND_TYPE_TO_EVENT = {
  lecture: '수업',
  meeting: '회의',
  assignment: '과제',
  exam: '시험',
  presentation: '발표',
  project: '프로젝트',
  etc: '기타'
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

const normalizeEventType = (type, origin = 'manual') => {
  if (TYPE_LABELS[type]) return type
  if (type && EVENT_TYPE_TO_FRONTEND[type]) return EVENT_TYPE_TO_FRONTEND[type]
  return origin === 'ai' ? 'meeting' : 'etc'
}

const formatKoreanTime = (date) => {
  let hour = date.getHours()
  const minute = String(date.getMinutes()).padStart(2, '0')
  const meridiem = hour < 12 ? '오전' : '오후'
  hour %= 12
  if (hour === 0) hour = 12
  return `${meridiem} ${String(hour).padStart(2, '0')}:${minute}`
}

const parseDueDate = (dueDate) => {
  if (!dueDate) return null
  const parsed = new Date(dueDate)
  if (Number.isNaN(parsed.getTime())) return null
  return parsed
}

const parseKoreanTime = (timeText = '오전 09:00') => {
  const match = String(timeText).match(/(오전|오후)\s*(\d{1,2}):(\d{2})/)
  if (!match) return { hour: 9, minute: 0 }

  const [, meridiem, rawHour, rawMinute] = match
  let hour = Number(rawHour)
  const minute = Number(rawMinute)

  if (meridiem === '오전' && hour === 12) hour = 0
  if (meridiem === '오후' && hour !== 12) hour += 12

  return { hour, minute }
}

const toApiDueDate = (item) => {
  const { hour, minute } = parseKoreanTime(item.startTime || item.time)
  return `${item.dateKey}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00`
}

const mapScheduleFromApi = (item) => {
  const dueDate = parseDueDate(item.due_date)
  const dateKey = dueDate ? formatDateKey(dueDate) : formatDateKey()
  const startTime = dueDate ? formatKoreanTime(dueDate) : ''

  return normalizeScheduleItem({
    id: item.schedule_id,
    apiId: item.schedule_id,
    origin: item.source_text ? 'ai' : 'manual',
    type: normalizeEventType(item.event_type, item.source_text ? 'ai' : 'manual'),
    dateKey,
    title: item.title,
    startTime,
    endTime: startTime,
    time: startTime,
    note: item.description || '',
    sourceText: item.source_text || '',
    sourceSessionTitle: item.session_title || item.course_title || '',
    workspaceFileId: item.session_id || '',
    confidence: null,
    status: item.status,
    transcriptId: item.transcript_id || '',
    sourceStartTime: item.source_start_time,
    sourceEndTime: item.source_end_time
  })
}

const requestJson = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  })

  if (!response.ok) {
    throw new Error(`Schedule API request failed: ${response.status}`)
  }

  return response.json()
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
    apiId: item.apiId || (UUID_PATTERN.test(item.id || '') ? item.id : ''),
    origin,
    type: normalizeEventType(item.type || item.fileKind, origin),
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

const persistSchedules = (items) => {
  scheduleItems.value = items.map(normalizeScheduleItem)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(scheduleItems.value))
}

const removeDemoSchedules = (items) => {
  return items.filter((item) => !String(item.id || '').startsWith('sample-ai-'))
}

const loadCachedSchedules = () => {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return []

  try {
    return removeDemoSchedules(JSON.parse(raw).map(normalizeScheduleItem))
  } catch {
    return []
  }
}

const sortSchedules = (items) => {
  return [...items].sort((a, b) => {
    const dateCompare = a.dateKey.localeCompare(b.dateKey)
    if (dateCompare !== 0) return dateCompare
    return String(a.startTime || '').localeCompare(String(b.startTime || ''))
  })
}

const syncScheduleStatus = (scheduleId, status) => {
  const endpoint = status === 'confirmed' ? 'confirm' : 'ignore'
  if (!UUID_PATTERN.test(scheduleId) || (status !== 'confirmed' && status !== 'ignored')) return
  requestJson(`${API_BASE}/${scheduleId}/${endpoint}`, { method: 'PUT' }).catch((error) => {
    console.warn('[Schedule] status sync failed', error)
  })
}

const syncToNotion = async (scheduleId) => {
  if (!UUID_PATTERN.test(scheduleId)) return null
  try {
    const result = await requestJson(`${API_BASE}/${scheduleId}/sync-notion`, { method: 'POST' })
    return result?.notion_page_id || null
  } catch (error) {
    console.warn('[Schedule] notion sync failed for', scheduleId, error)
    return null
  }
}

const syncManualSchedule = (localItem) => {
  requestJson(`${API_BASE}/manual`, {
    method: 'POST',
    body: JSON.stringify({
      title: localItem.title,
      description: localItem.note,
      event_type: FRONTEND_TYPE_TO_EVENT[localItem.type] || '기타',
      due_date: toApiDueDate(localItem),
      status: 'confirmed'
    })
  })
    .then((saved) => {
      if (!saved?.schedule_id) return
      const apiItem = mapScheduleFromApi(saved)
      persistSchedules(scheduleItems.value.map((item) => (
        item.id === localItem.id ? apiItem : item
      )))
    })
    .catch((error) => {
      console.warn('[Schedule] manual schedule sync failed', error)
    })
}

export function useScheduleState() {
  const hydrateSchedules = async ({ force = false } = {}) => {
    if (hasLoadedSchedules.value && !force) return scheduleItems.value

    try {
      const data = await requestJson(`${API_BASE}/`)
      const apiItems = Array.isArray(data.schedules)
        ? data.schedules.map(mapScheduleFromApi)
        : []
      persistSchedules(apiItems)
    } catch (error) {
      console.warn('[Schedule] API load failed, using local fallback', error)
      const cachedItems = loadCachedSchedules()
      persistSchedules(cachedItems)
    }

    hasLoadedSchedules.value = true
    return scheduleItems.value
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
    syncManualSchedule(item)
    return item
  }

  const updateScheduleStatus = (scheduleId, status) => {
    persistSchedules(scheduleItems.value.map((item) => (
      item.id === scheduleId
        ? normalizeScheduleItem({ ...item, status })
        : item
    )))
    syncScheduleStatus(scheduleId, status)
  }

  const confirmSchedule = (scheduleId) => updateScheduleStatus(scheduleId, 'confirmed')
  const ignoreSchedule = (scheduleId) => updateScheduleStatus(scheduleId, 'ignored')

  const confirmAndSyncToNotion = async (scheduleId) => {
    updateScheduleStatus(scheduleId, 'confirmed')
    const notionPageId = await syncToNotion(scheduleId)
    if (notionPageId) {
      persistSchedules(scheduleItems.value.map((item) => (
        item.id === scheduleId
          ? normalizeScheduleItem({ ...item, notionPageId })
          : item
      )))
    }
    return notionPageId
  }

  const syncNotionExport = async () => {
    notionSyncing.value = true
    try {
      const result = await requestJson(`${API_BASE}/sync-notion-confirmed`, { method: 'POST' })
      await hydrateSchedules({ force: true })
      return result
    } catch (error) {
      console.error('[Schedule] notion export failed', error)
      throw error
    } finally {
      notionSyncing.value = false
    }
  }

  const syncNotionImport = async () => {
    notionSyncing.value = true
    try {
      const result = await requestJson(`${API_BASE}/sync-notion-import`, { method: 'POST' })
      await hydrateSchedules({ force: true })
      return result
    } catch (error) {
      console.error('[Schedule] notion import failed', error)
      throw error
    } finally {
      notionSyncing.value = false
    }
  }
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
    notionSyncing: readonly(notionSyncing),
    hydrateSchedules,
    addManualSchedule,
    confirmSchedule,
    ignoreSchedule,
    confirmAndSyncToNotion,
    syncNotionExport,
    syncNotionImport,
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
