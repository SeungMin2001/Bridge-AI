<!-- 홈 화면의 왼쪽 사이드바 컴포넌트로, 앱 메뉴와 캘린더 기능을 포함합니다. -->
<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useScheduleState } from '../../composables/useScheduleState'

const props = defineProps({
  isCollapsed: Boolean,
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['toggle', 'navigate'])

const expandedSidebarWidth = 370
const collapsedSidebarWidth = 56

const weekLabels = ['일', '월', '화', '수', '목', '금', '토']
const meridiemOptions = ['오전', '오후']
const hourOptions = Array.from({ length: 12 }, (_, index) => `${index + 1}`.padStart(2, '0'))
const minuteOptions = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
const scheduleTypeOptions = [
  { value: 'lecture', label: '수업' },
  { value: 'meeting', label: '회의' },
  { value: 'assignment', label: '과제' },
  { value: 'exam', label: '시험' },
  { value: 'etc', label: '기타' }
]

const {
  pendingSchedules,
  hydrateSchedules,
  addManualSchedule,
  confirmSchedule,
  ignoreSchedule,
  getSchedulesForDate,
  getScheduleDayFlags,
  formatDateKey,
  formatDateLabel,
  getWeekKeyFromDateKey,
  getTypeLabel,
  getTypeIcon,
  getStatusLabel,
  getConfidenceLabel
} = useScheduleState()

const today = new Date()
const currentMonthLabel = computed(() => `${today.getMonth() + 1}월`)
const calendarCardRef = ref(null)
const isScheduleModalOpen = ref(false)
const scheduleModalPos = ref({ x: 0, y: 0 })
const selectedDateKey = ref(formatDateKey(today))
const scheduleForm = ref({
  type: 'meeting',
  title: '',
  startMeridiem: '오전',
  startHour: '09',
  startMinute: '00',
  endMeridiem: '오전',
  endHour: '10',
  endMinute: '00',
  note: '',
})

function createEmptyScheduleForm() {
  return {
    type: 'meeting',
    title: '',
    startMeridiem: '오전',
    startHour: '09',
    startMinute: '00',
    endMeridiem: '오전',
    endHour: '10',
    endMinute: '00',
    note: '',
  }
}

function openScheduleModal(day) {
  if (day.muted) return
  selectedDateKey.value = day.dateKey
  scheduleForm.value = createEmptyScheduleForm()
  isScheduleModalOpen.value = true

  nextTick(() => {
    const rect = calendarCardRef.value?.getBoundingClientRect()
    if (!rect) return
    scheduleModalPos.value = {
      x: rect.right + 14,
      y: rect.top + 12,
    }
  })
}

function closeScheduleModal() {
  isScheduleModalOpen.value = false
}

function formatSelectedTime(meridiem, hour, minute) {
  return `${meridiem} ${hour}:${minute}`
}

function submitSchedule() {
  if (!scheduleForm.value.title.trim()) return

  const startTime = formatSelectedTime(
    scheduleForm.value.startMeridiem,
    scheduleForm.value.startHour,
    scheduleForm.value.startMinute
  )
  const endTime = formatSelectedTime(
    scheduleForm.value.endMeridiem,
    scheduleForm.value.endHour,
    scheduleForm.value.endMinute
  )
  const timeRange = `${startTime} - ${endTime}`

  addManualSchedule({
    dateKey: selectedDateKey.value,
    weekKey: getWeekKeyFromDateKey(selectedDateKey.value),
    type: scheduleForm.value.type,
    title: scheduleForm.value.title.trim(),
    startTime,
    endTime,
    time: timeRange,
    note: scheduleForm.value.note.trim()
  })

  closeScheduleModal()
}

function handleConfirmSchedule(item) {
  confirmSchedule(item.id)
  selectedDateKey.value = item.dateKey
}

function handleIgnoreSchedule(item) {
  ignoreSchedule(item.id)
}

function openScheduleManagement(item) {
  if (item?.dateKey) selectedDateKey.value = item.dateKey
  emit('navigate', 'schedule')
}

function handleWindowResize() {
  if (!isScheduleModalOpen.value) return
  const rect = calendarCardRef.value?.getBoundingClientRect()
  if (!rect) return
  scheduleModalPos.value = {
    x: rect.right + 14,
    y: rect.top + 12,
  }
}

const calendarDays = computed(() => {
  const year = today.getFullYear()
  const month = today.getMonth()
  const firstDay = new Date(year, month, 1)
  const lastDay = new Date(year, month + 1, 0)
  const prevLastDay = new Date(year, month, 0)

  const days = []
  const leadingCount = firstDay.getDay()
  const trailingCount = 6 - lastDay.getDay()

  for (let i = leadingCount - 1; i >= 0; i -= 1) {
    const date = new Date(year, month - 1, prevLastDay.getDate() - i)
    const dateKey = formatDateKey(date)
    days.push({ label: prevLastDay.getDate() - i, muted: true, isToday: false, dateKey, ...getScheduleDayFlags(dateKey) })
  }

  for (let date = 1; date <= lastDay.getDate(); date += 1) {
    const currentDate = new Date(year, month, date)
    const dateKey = formatDateKey(currentDate)
    days.push({ label: date, muted: false, isToday: date === today.getDate(), dateKey, ...getScheduleDayFlags(dateKey) })
  }

  for (let date = 1; date <= trailingCount; date += 1) {
    const nextDate = new Date(year, month + 1, date)
    const dateKey = formatDateKey(nextDate)
    days.push({ label: date, muted: true, isToday: false, dateKey, ...getScheduleDayFlags(dateKey) })
  }

  return days
})

const selectedDateLabel = computed(() => formatDateLabel(selectedDateKey.value))

const selectedSchedules = computed(() => {
  return getSchedulesForDate(selectedDateKey.value)
})

function getFavoriteIcon(item) {
  if (item.type === 'folder') return 'folder'
  if (item.fileIcon) return item.fileIcon
  if (item.tag === '회의' || item.fileKind === 'meeting') return 'groups_2'
  if (item.tag === '프로젝트') return 'workspaces'
  if (item.tag === '개인') return 'person'
  if (item.tag === '중요') return 'priority_high'
  return 'description'
}

function getFavoriteIconStyle(item) {
  if (item.type === 'folder') {
    return { color: item.color, fontVariationSettings: "'FILL' 1" }
  }

  if (item.fileIcon || item.fileKind === 'meeting') {
    return { color: item.color || '#ec4899', fontVariationSettings: "'FILL' 1" }
  }

  return { color: item.color, fontVariationSettings: "'FILL' 0" }
}

function colorWithAlpha(color = '#6366f1', alpha = 0.12) {
  const hex = String(color).trim()
  const fullHex = /^#[0-9a-fA-F]{6}$/.test(hex)
    ? hex
    : (/^#[0-9a-fA-F]{3}$/.test(hex)
        ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
        : '#6366f1')
  const value = fullHex.slice(1)
  const red = parseInt(value.slice(0, 2), 16)
  const green = parseInt(value.slice(2, 4), 16)
  const blue = parseInt(value.slice(4, 6), 16)
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

function getFavoriteTag(item) {
  if (item.type === 'folder') return ''
  return item.tag || (item.fileKind === 'meeting' ? '회의' : '수업')
}

function getFavoriteTagStyle(item) {
  const color = item.color || '#6366f1'
  return {
    color,
    background: colorWithAlpha(color, 0.12)
  }
}

watch(() => props.isCollapsed, (collapsed) => {
  if (collapsed) closeScheduleModal()
})

onMounted(() => {
  hydrateSchedules()
  if (pendingSchedules.value.length > 0) {
    selectedDateKey.value = pendingSchedules.value[0].dateKey
  }
  window.addEventListener('resize', handleWindowResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleWindowResize)
})
</script>

<template>
  <aside
    id="sidebar"
    :class="[
      'home-sidebar flex flex-col h-full shrink-0 overflow-hidden transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] relative z-10 rounded-[24px]',
      { 'sidebar-collapsed': isCollapsed }
    ]"
    :style="{
      width: `${isCollapsed ? collapsedSidebarWidth : expandedSidebarWidth}px`,
      minWidth: `${isCollapsed ? collapsedSidebarWidth : expandedSidebarWidth}px`,
      maxWidth: `${isCollapsed ? collapsedSidebarWidth : expandedSidebarWidth}px`
    }"
  >
    <div class="sidebar-main-card card flex flex-col h-full p-5 overflow-hidden min-w-[280px] home-left-sidebar-card">
      <div class="sidebar-header transition-all">
        <div class="sidebar-logo-section">
          <div class="sidebar-logo-box">
            <span class="material-symbols-outlined text-white text-[20px]">menu_book</span>
          </div>
          <span class="collapsible-content sidebar-logo-text font-extrabold">LectoAI</span>
        </div>
        <div class="sidebar-btn-group">
          <button class="sidebar-icon-btn home-sidebar-icon-btn" @click="emit('toggle')">
            <span class="material-symbols-outlined">
              {{ isCollapsed ? 'menu' : 'side_navigation' }}
            </span>
          </button>
          <button
            class="sidebar-icon-btn home-sidebar-icon-btn"
            @click="emit('navigate', 'workspace')"
          >
            <span class="material-symbols-outlined">edit_note</span>
          </button>
        </div>
      </div>
      
      <div class="sidebar-search-container rounded-[24px]">
        <span class="material-symbols-outlined shrink-0">search</span>
        <span class="sidebar-search-text collapsible-content">제목으로 검색</span>
      </div>

      <div class="sidebar-content collapsible-content-container custom-scrollbar">
        <section ref="calendarCardRef" class="home-calendar-card collapsible-content">
          <div class="home-calendar-header">
            <span class="home-calendar-month">{{ currentMonthLabel }}</span>
            <button class="home-calendar-icon-btn" @click="emit('navigate', 'schedule')">
              <span class="material-symbols-outlined text-[18px]">menu</span>
            </button>
          </div>

          <div class="home-calendar-weekdays">
            <span v-for="label in weekLabels" :key="label">{{ label }}</span>
          </div>

          <div class="home-calendar-grid">
            <div
              v-for="(day, index) in calendarDays"
              :key="`${day.label}-${index}`"
              class="home-calendar-day"
              :class="{ 'is-muted': day.muted, 'is-today': day.isToday, 'is-selected': day.dateKey === selectedDateKey, 'has-schedule': day.hasSchedule, 'has-pending': day.hasPendingSchedule, 'has-confirmed': day.hasConfirmedSchedule }"
              @click="openScheduleModal(day)"
            >
              {{ day.label }}
              <span v-if="day.hasSchedule" class="home-calendar-day-dot"></span>
            </div>
          </div>

          <div class="home-calendar-divider"></div>

          <div v-if="pendingSchedules.length" class="home-calendar-ai-panel">
            <div class="home-calendar-ai-heading">
              <span class="material-symbols-outlined text-[15px]">auto_awesome</span>
              <span>AI 감지 일정</span>
              <span class="home-calendar-ai-count">{{ pendingSchedules.length }}</span>
            </div>
            <article
              v-for="item in pendingSchedules.slice(0, 3)"
              :key="`pending-${item.id}`"
              class="home-calendar-ai-card"
              @click="selectedDateKey = item.dateKey"
            >
              <div class="home-calendar-ai-card-top">
                <span class="home-calendar-type-chip">
                  <span class="material-symbols-outlined text-[13px]">{{ getTypeIcon(item.type) }}</span>
                  {{ getTypeLabel(item.type) }}
                </span>
                <span class="home-calendar-confidence">{{ getConfidenceLabel(item.confidence) }}</span>
              </div>
              <div class="home-calendar-ai-title">{{ item.title }}</div>
              <div class="home-calendar-ai-meta">{{ formatDateLabel(item.dateKey) }} · {{ item.time }}</div>
              <div class="home-calendar-source-text">{{ item.sourceText }}</div>
              <div class="home-calendar-schedule-actions">
                <button class="home-calendar-action-btn confirm" @click.stop="handleConfirmSchedule(item)">확정</button>
                <button class="home-calendar-action-btn ghost" @click.stop="handleIgnoreSchedule(item)">무시</button>
              </div>
            </article>
          </div>

          <div v-if="pendingSchedules.length" class="home-calendar-divider"></div>

          <div class="home-calendar-schedule-list">
            <div class="home-calendar-schedule-heading">{{ selectedDateLabel }}</div>
            <template v-if="selectedSchedules.length > 0">
              <article
                v-for="item in selectedSchedules"
                :key="item.id"
                class="home-calendar-schedule-item"
                :class="{ 'is-pending': item.status === 'pending', 'is-ai': item.origin === 'ai' }"
              >
                <div class="home-calendar-schedule-rail"></div>
                <div class="home-calendar-schedule-content">
                  <div class="home-calendar-schedule-top">
                    <span class="home-calendar-type-chip">
                      <span class="material-symbols-outlined text-[13px]">{{ getTypeIcon(item.type) }}</span>
                      {{ getTypeLabel(item.type) }}
                    </span>
                    <span class="home-calendar-schedule-status">{{ getStatusLabel(item.status) }}</span>
                  </div>
                  <div class="home-calendar-schedule-title">{{ item.title }}</div>
                  <div class="home-calendar-schedule-meta">
                    <span class="material-symbols-outlined text-[13px]">schedule</span>
                    <span>{{ item.time }}</span>
                  </div>
                  <div v-if="item.note" class="home-calendar-schedule-note">{{ item.note }}</div>
                  <div v-if="item.sourceText" class="home-calendar-source-text">{{ item.sourceText }}</div>
                  <div class="home-calendar-schedule-actions">
                    <template v-if="item.status === 'pending'">
                      <button class="home-calendar-action-btn confirm" @click="handleConfirmSchedule(item)">확정</button>
                      <button class="home-calendar-action-btn ghost" @click="handleIgnoreSchedule(item)">무시</button>
                    </template>
                    <button
                      v-else
                      class="home-calendar-action-btn open"
                      @click="openScheduleManagement(item)"
                    >
                      일정관리
                    </button>
                  </div>
                </div>
              </article>
            </template>
            <div v-else class="home-calendar-empty-state">
              날짜를 눌러 일정을 추가해보세요.
            </div>
          </div>
        </section>

        <div class="sidebar-section-title">즐겨찾기</div>
        <div id="favorites-list" class="flex flex-col gap-1">
          <div 
            v-for="fav in props.fileTree.filter(item => props.favorites.has(item.id))" 
            :key="`fav-${fav.id}`" 
            class="sidebar-nav-item home-sidebar-nav-item" 
            @click="emit('navigate', 'workspace')"
          >
            <span class="material-symbols-outlined nav-icon" :style="getFavoriteIconStyle(fav)">
              {{ getFavoriteIcon(fav) }}
            </span>
            <span class="nav-text truncate collapsible-content">{{ fav.name }}</span>
            <span
              v-if="getFavoriteTag(fav)"
              class="home-sidebar-kind-badge collapsible-content"
              :style="getFavoriteTagStyle(fav)"
            >
              {{ getFavoriteTag(fav) }}
            </span>
            <span v-if="fav.type === 'folder'" class="material-symbols-outlined text-[#8e8e93] text-[18px] collapsible-content">expand_more</span>
          </div>
        </div>
      </div>

      <div class="mt-auto pt-5 flex justify-center w-full">
        <button class="sidebar-icon-btn home-sidebar-icon-btn !p-3 rounded-full text-[#1d1d1f]" @click="emit('navigate', 'home')">
          <span class="material-symbols-outlined !text-[24px]" style="font-variation-settings: 'FILL' 1">home</span>
        </button>
      </div>
    </div>
  </aside>

  <Teleport to="body">
    <transition name="calendar-modal">
      <div v-if="isScheduleModalOpen" class="home-calendar-modal-overlay" @click.self="closeScheduleModal">
        <div class="home-calendar-modal" :style="{ left: `${scheduleModalPos.x}px`, top: `${scheduleModalPos.y}px` }">
          <div class="home-calendar-modal-header">
            <div>
              <div class="home-calendar-modal-title">일정 추가</div>
              <div class="home-calendar-modal-date">{{ selectedDateLabel }}</div>
            </div>
            <button class="home-calendar-modal-close" @click="closeScheduleModal">
              <span class="material-symbols-outlined text-[18px]">close</span>
            </button>
          </div>

          <label class="home-calendar-field">
            <span>종류</span>
            <select v-model="scheduleForm.type" class="home-calendar-type-select">
              <option v-for="option in scheduleTypeOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
            </select>
          </label>

          <label class="home-calendar-field">
            <span>제목</span>
            <input v-model="scheduleForm.title" type="text" placeholder="예: 팀 미팅" />
          </label>

          <label class="home-calendar-field">
            <span>시간</span>
            <div class="home-calendar-time-grid">
              <div class="home-calendar-time-block">
                <span class="home-calendar-time-label">시작</span>
                <div class="home-calendar-time-picker">
                  <select v-model="scheduleForm.startMeridiem" class="home-calendar-time-select home-calendar-time-meridiem">
                    <option v-for="option in meridiemOptions" :key="`start-${option}`" :value="option">{{ option }}</option>
                  </select>
                  <select v-model="scheduleForm.startHour" class="home-calendar-time-select">
                    <option v-for="option in hourOptions" :key="`start-hour-${option}`" :value="option">{{ option }}</option>
                  </select>
                  <span class="home-calendar-time-separator">:</span>
                  <select v-model="scheduleForm.startMinute" class="home-calendar-time-select">
                    <option v-for="option in minuteOptions" :key="`start-minute-${option}`" :value="option">{{ option }}</option>
                  </select>
                </div>
              </div>
              <div class="home-calendar-time-block">
                <span class="home-calendar-time-label">종료</span>
                <div class="home-calendar-time-picker">
                  <select v-model="scheduleForm.endMeridiem" class="home-calendar-time-select home-calendar-time-meridiem">
                    <option v-for="option in meridiemOptions" :key="`end-${option}`" :value="option">{{ option }}</option>
                  </select>
                  <select v-model="scheduleForm.endHour" class="home-calendar-time-select">
                    <option v-for="option in hourOptions" :key="`end-hour-${option}`" :value="option">{{ option }}</option>
                  </select>
                  <span class="home-calendar-time-separator">:</span>
                  <select v-model="scheduleForm.endMinute" class="home-calendar-time-select">
                    <option v-for="option in minuteOptions" :key="`end-minute-${option}`" :value="option">{{ option }}</option>
                  </select>
                </div>
              </div>
            </div>
          </label>

          <label class="home-calendar-field">
            <span>메모</span>
            <textarea v-model="scheduleForm.note" rows="3" placeholder="필요한 내용을 적어보세요"></textarea>
          </label>

          <button class="home-calendar-submit-btn" @click="submitSchedule">일정 저장</button>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<style scoped>
.home-sidebar-kind-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(236, 72, 153, 0.12);
  color: #db2777;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: -0.01em;
}
</style>

<style scoped>
.home-left-sidebar-card {
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.86);
  box-shadow: 0 24px 48px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.home-left-sidebar-card::before {
  background: rgba(255, 255, 255, 0.22);
}

.home-left-sidebar-card::after {
  border-color: rgba(255, 255, 255, 0.42);
}

.home-sidebar-icon-btn {
  border: 1px solid rgba(255, 255, 255, 0.58);
  background: rgba(255, 255, 255, 0.4);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.84);
}

.home-sidebar-nav-item {
  border: 1px solid rgba(255, 255, 255, 0.44);
  background: #f4ede4;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
}
</style>
