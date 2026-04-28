<!-- 홈 화면의 왼쪽 사이드바 컴포넌트로, 앱 메뉴와 캘린더 기능을 포함합니다. -->
<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  isCollapsed: Boolean,
  fileTree: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() }
})

const emit = defineEmits(['toggle', 'navigate'])

const weekLabels = ['일', '월', '화', '수', '목', '금', '토']
const meridiemOptions = ['오전', '오후']
const hourOptions = Array.from({ length: 12 }, (_, index) => `${index + 1}`.padStart(2, '0'))
const minuteOptions = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
const today = new Date()
const currentMonthLabel = computed(() => `${today.getMonth() + 1}월`)
const calendarCardRef = ref(null)
const isScheduleModalOpen = ref(false)
const scheduleModalPos = ref({ x: 0, y: 0 })
const selectedDateKey = ref(formatDateKey(today))
const scheduleForm = ref({
  title: '',
  startMeridiem: '오전',
  startHour: '09',
  startMinute: '00',
  endMeridiem: '오전',
  endHour: '10',
  endMinute: '00',
  note: '',
})
const scheduleItems = ref([])

function formatDateKey(date) {
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  return `${year}-${month}-${day}`
}

function formatDateLabel(dateKey) {
  const [year, month, day] = dateKey.split('-').map(Number)
  return `${year}년 ${month}월 ${day}일`
}

function loadSchedules() {
  const raw = localStorage.getItem('lecto_home_calendar_schedules')
  if (!raw) return
  try {
    scheduleItems.value = JSON.parse(raw)
  } catch {
    scheduleItems.value = []
  }
}

function saveSchedules(items) {
  scheduleItems.value = items
  localStorage.setItem('lecto_home_calendar_schedules', JSON.stringify(items))
}

function openScheduleModal(day) {
  if (day.muted) return
  selectedDateKey.value = day.dateKey
  scheduleForm.value = {
    title: '',
    startMeridiem: '오전',
    startHour: '09',
    startMinute: '00',
    endMeridiem: '오전',
    endHour: '10',
    endMinute: '00',
    note: '',
  }
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

  const nextItems = [
    ...scheduleItems.value,
    {
      id: `${selectedDateKey.value}-${Date.now()}`,
      dateKey: selectedDateKey.value,
      title: scheduleForm.value.title.trim(),
      startTime,
      endTime,
      time: timeRange,
      note: scheduleForm.value.note.trim(),
      status: '예정',
    },
  ]

  saveSchedules(nextItems)
  closeScheduleModal()
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
    days.push({ label: prevLastDay.getDate() - i, muted: true, isToday: false, dateKey, hasSchedule: scheduleItems.value.some((item) => item.dateKey === dateKey) })
  }

  for (let date = 1; date <= lastDay.getDate(); date += 1) {
    const currentDate = new Date(year, month, date)
    const dateKey = formatDateKey(currentDate)
    days.push({ label: date, muted: false, isToday: date === today.getDate(), dateKey, hasSchedule: scheduleItems.value.some((item) => item.dateKey === dateKey) })
  }

  for (let date = 1; date <= trailingCount; date += 1) {
    const nextDate = new Date(year, month + 1, date)
    const dateKey = formatDateKey(nextDate)
    days.push({ label: date, muted: true, isToday: false, dateKey, hasSchedule: scheduleItems.value.some((item) => item.dateKey === dateKey) })
  }

  return days
})

const selectedDateLabel = computed(() => formatDateLabel(selectedDateKey.value))

const selectedSchedules = computed(() => {
  return scheduleItems.value.filter((item) => item.dateKey === selectedDateKey.value)
})

function getFavoriteIcon(item) {
  if (item.type === 'folder') return 'folder'
  return item.fileKind === 'meeting' ? 'groups_2' : 'description'
}

function getFavoriteIconStyle(item) {
  if (item.type === 'folder') {
    return { color: item.color, fontVariationSettings: "'FILL' 1" }
  }

  if (item.fileKind === 'meeting') {
    return { color: item.color || '#ec4899', fontVariationSettings: "'FILL' 1" }
  }

  return { color: item.color, fontVariationSettings: "'FILL' 0" }
}

watch(() => props.isCollapsed, (collapsed) => {
  if (collapsed) closeScheduleModal()
})

onMounted(() => {
  loadSchedules()
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
      isCollapsed ? 'w-16' : 'w-[340px]',
      'home-sidebar flex flex-col h-full shrink-0 overflow-hidden transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)] relative z-10 rounded-[24px]',
      { 'sidebar-collapsed': isCollapsed }
    ]"
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
            <button class="home-calendar-icon-btn">
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
              :class="{ 'is-muted': day.muted, 'is-today': day.isToday, 'is-selected': day.dateKey === selectedDateKey, 'has-schedule': day.hasSchedule }"
              @click="openScheduleModal(day)"
            >
              {{ day.label }}
              <span v-if="day.hasSchedule" class="home-calendar-day-dot"></span>
            </div>
          </div>

          <div class="home-calendar-divider"></div>

          <div class="home-calendar-schedule-list">
            <div class="home-calendar-schedule-heading">{{ selectedDateLabel }}</div>
            <template v-if="selectedSchedules.length > 0">
              <article v-for="item in selectedSchedules" :key="item.id" class="home-calendar-schedule-item">
                <div class="home-calendar-schedule-rail"></div>
                <div class="home-calendar-schedule-content">
                  <div class="home-calendar-schedule-top">
                    <span class="home-calendar-schedule-day">{{ selectedDateLabel }}</span>
                    <span class="home-calendar-schedule-status">{{ item.status }}</span>
                  </div>
                  <div class="home-calendar-schedule-title">{{ item.title }}</div>
                  <div class="home-calendar-schedule-meta">
                    <span class="material-symbols-outlined text-[13px]">schedule</span>
                    <span>{{ item.time }}</span>
                  </div>
                  <div v-if="item.note" class="home-calendar-schedule-note">{{ item.note }}</div>
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
            <span v-if="fav.fileKind === 'meeting'" class="home-sidebar-kind-badge collapsible-content">회의</span>
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
