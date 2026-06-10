<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  initialData: { type: Object, default: () => null }
})

const emit = defineEmits(['close', 'save'])

const title = ref('')
const description = ref('')
const selectedDate = ref(new Date())
const calendarYear = ref(new Date().getFullYear())
const calendarMonth = ref(new Date().getMonth())
const isCalendarExpanded = ref(false)
const eventType = ref('etc')
const status = ref('confirmed')

const EVENT_TYPES = [
  { label: '수업', value: 'lecture', icon: 'menu_book' },
  { label: '회의', value: 'meeting', icon: 'groups' },
  { label: '과제', value: 'assignment', icon: 'assignment' },
  { label: '시험', value: 'exam', icon: 'quiz' },
  { label: '발표', value: 'presentation', icon: 'campaign' },
  { label: '프로젝트', value: 'project', icon: 'workspaces' },
  { label: '기타', value: 'etc', icon: 'event' }
]

const STATUS_TYPES = [
  { label: '예정', value: 'confirmed' },
  { label: '확인 대기', value: 'pending' }
]

watch(() => props.visible, (newVal) => {
  if (newVal) {
    title.value = ''
    description.value = ''
    const now = new Date()
    selectedDate.value = now
    calendarYear.value = now.getFullYear()
    calendarMonth.value = now.getMonth()
    eventType.value = 'etc'
    status.value = 'confirmed'
    isCalendarExpanded.value = false
  }
})

const formatDateToString = (date) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`
}

const handleSave = () => {
  if (!title.value.trim()) {
    alert('일정 제목을 입력해주세요.')
    return
  }

  emit('save', {
    title: title.value,
    description: description.value,
    event_type: eventType.value,
    due_date: formatDateToString(selectedDate.value),
    status: status.value,
    session_id: props.initialData?.sessionId,
    recording_id: props.initialData?.recordingId,
    source_start_time: props.initialData?.sourceStartTime,
    source_end_time: props.initialData?.sourceEndTime,
    source_text: props.initialData?.sourceText
  })
}

// 달력 날짜 목록 계산
const getDaysInMonth = (year, month) => new Date(year, month + 1, 0).getDate()
const getFirstDayOfMonth = (year, month) => new Date(year, month, 1).getDay()

const daysArray = computed(() => {
  const daysInMonth = getDaysInMonth(calendarYear.value, calendarMonth.value)
  const firstDayIndex = getFirstDayOfMonth(calendarYear.value, calendarMonth.value)
  const arr = []
  for (let i = 0; i < firstDayIndex; i++) arr.push(null)
  for (let i = 1; i <= daysInMonth; i++) arr.push(i)
  return arr
})

const handlePrevMonth = () => {
  if (calendarMonth.value === 0) {
    calendarMonth.value = 11
    calendarYear.value -= 1
  } else {
    calendarMonth.value -= 1
  }
}

const handleNextMonth = () => {
  if (calendarMonth.value === 11) {
    calendarMonth.value = 0
    calendarYear.value += 1
  } else {
    calendarMonth.value += 1
  }
}

const changeHour = (amount) => {
  const newDate = new Date(selectedDate.value)
  newDate.setHours((newDate.getHours() + amount + 24) % 24)
  selectedDate.value = newDate
}

const changeMinute = (amount) => {
  const newDate = new Date(selectedDate.value)
  newDate.setMinutes((newDate.getMinutes() + amount + 60) % 60)
  selectedDate.value = newDate
}

const selectDay = (day) => {
  if (day !== null) {
    const newDate = new Date(selectedDate.value)
    newDate.setFullYear(calendarYear.value)
    newDate.setMonth(calendarMonth.value)
    newDate.setDate(day)
    selectedDate.value = newDate
  }
}

const isSelectedDay = (day) => {
  return day !== null &&
    selectedDate.value.getFullYear() === calendarYear.value &&
    selectedDate.value.getMonth() === calendarMonth.value &&
    selectedDate.value.getDate() === day
}

const isToday = (day) => {
  const now = new Date()
  return day !== null &&
    now.getFullYear() === calendarYear.value &&
    now.getMonth() === calendarMonth.value &&
    now.getDate() === day
}

const getDayName = (date) => ['일', '월', '화', '수', '목', '금', '토'][date.getDay()]
const displayDateStr = computed(() => `${selectedDate.value.getFullYear()}년 ${selectedDate.value.getMonth() + 1}월 ${selectedDate.value.getDate()}일 (${getDayName(selectedDate.value)})`)
const displayTimeStr = computed(() => `${String(selectedDate.value.getHours()).padStart(2, '0')}:${String(selectedDate.value.getMinutes()).padStart(2, '0')}`)

const weekDays = ['일', '월', '화', '수', '목', '금', '토']
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="fixed inset-0 z-[9999]">
      <div class="absolute inset-0 bg-transparent" @click="emit('close')"></div>
      
      <!-- 오른쪽 탭 하단에 나타나도록 고정 (left는 --workspace-script-pane-width 변수 활용) -->
      <div class="absolute top-[82px] left-[calc(var(--workspace-script-pane-width,50%)+16px)] w-[330px] max-h-[85%] bg-white rounded-2xl overflow-hidden shadow-[0_8px_24px_rgba(15,23,42,0.12)] border border-[#E2E8F0] flex flex-col z-[10000]">
        
        <div class="flex items-center justify-between px-4 py-3 border-b border-[#F1F5F9]">
          <span class="font-black text-[16px] text-[#0F172A]">일정 추가</span>
          <button type="button" @click="emit('close')" class="p-1 hover:bg-gray-100 rounded">
            <span class="material-symbols-outlined text-[20px] text-[#64748B]">close</span>
          </button>
        </div>

        <div class="flex-shrink flex-grow overflow-y-auto custom-scrollbar p-4 flex flex-col gap-4">
          <!-- 제목 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">제목 *</label>
            <input v-model="title" type="text" placeholder="일정 제목 입력" class="bg-[#F8FAFC] border border-[#E2E8F0] rounded-[10px] px-3 py-2.5 font-semibold text-[13px] text-[#0F172A] outline-none focus:border-[#355CFF]" />
          </div>

          <!-- 날짜 및 시간 선택 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">날짜 및 시간</label>
            <button type="button" @click="isCalendarExpanded = !isCalendarExpanded" :class="['flex items-center gap-2 bg-[#F8FAFC] border rounded-[10px] px-3 py-2.5', isCalendarExpanded ? 'border-[#355CFF] bg-[#EEF2FF]' : 'border-[#E2E8F0]']">
              <span class="material-symbols-outlined text-[18px] text-[#355CFF]">event</span>
              <span class="flex-1 text-left font-semibold text-[13px] text-[#0F172A]">{{ displayDateStr }} {{ displayTimeStr }}</span>
              <span class="material-symbols-outlined text-[20px] text-[#71717A]">{{ isCalendarExpanded ? 'expand_less' : 'expand_more' }}</span>
            </button>

            <div v-if="isCalendarExpanded" class="bg-white border border-[#E2E8F0] rounded-[10px] p-3 mt-1 flex flex-col gap-3">
              <div class="flex justify-between items-center">
                <button type="button" @click="handlePrevMonth" class="p-1 rounded-md bg-[#F1F5F9] hover:bg-[#E2E8F0]">
                  <span class="material-symbols-outlined text-[20px] text-[#4B5563]">chevron_left</span>
                </button>
                <span class="font-extrabold text-[13px] text-[#1E293B]">{{ calendarYear }}년 {{ calendarMonth + 1 }}월</span>
                <button type="button" @click="handleNextMonth" class="p-1 rounded-md bg-[#F1F5F9] hover:bg-[#E2E8F0]">
                  <span class="material-symbols-outlined text-[20px] text-[#4B5563]">chevron_right</span>
                </button>
              </div>

              <div class="flex flex-col gap-1">
                <div class="flex justify-around">
                  <div v-for="(wd, index) in weekDays" :key="index" class="w-8 h-8 flex items-center justify-center">
                    <span :class="['font-extrabold text-[10px] text-center', index === 0 ? 'text-[#EF4444]' : (index === 6 ? 'text-[#3B82F6]' : 'text-[#94A3B8]')]">{{ wd }}</span>
                  </div>
                </div>
                <div class="flex flex-wrap justify-around">
                  <button v-for="(day, idx) in daysArray" :key="idx" type="button" :disabled="day === null" @click="selectDay(day)" :class="[
                    'w-8 h-8 flex items-center justify-center rounded-full',
                    isSelectedDay(day) ? 'bg-[#355CFF]' : (isToday(day) ? 'border border-[#355CFF]' : 'hover:bg-gray-100')
                  ]">
                    <span v-if="day !== null" :class="[
                      'font-semibold text-[11px]',
                      isSelectedDay(day) ? 'text-white font-extrabold' : (isToday(day) ? 'text-[#355CFF]' : ((idx % 7 === 0) ? 'text-[#EF4444]' : ((idx % 7 === 6) ? 'text-[#3B82F6]' : 'text-[#334155]')))
                    ]">{{ day }}</span>
                  </button>
                </div>
              </div>

              <div class="border-t border-[#F1F5F9] pt-2.5 flex flex-col gap-2">
                <span class="font-extrabold text-[11px] text-[#475569]">시간 설정</span>
                <div class="flex justify-between gap-3">
                  <div class="flex-1 flex items-center justify-between bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg px-2 py-1.5">
                    <button type="button" @click="changeHour(-1)" class="p-1 rounded bg-[#E2E8F0] hover:bg-[#CBD5E1]">
                      <span class="material-symbols-outlined text-[16px] text-[#4B5563]">remove</span>
                    </button>
                    <span class="font-extrabold text-[12px] text-[#1E293B]">{{ String(selectedDate.getHours()).padStart(2, '0') }}시</span>
                    <button type="button" @click="changeHour(1)" class="p-1 rounded bg-[#E2E8F0] hover:bg-[#CBD5E1]">
                      <span class="material-symbols-outlined text-[16px] text-[#4B5563]">add</span>
                    </button>
                  </div>
                  <div class="flex-1 flex items-center justify-between bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg px-2 py-1.5">
                    <button type="button" @click="changeMinute(-5)" class="p-1 rounded bg-[#E2E8F0] hover:bg-[#CBD5E1]">
                      <span class="material-symbols-outlined text-[16px] text-[#4B5563]">remove</span>
                    </button>
                    <span class="font-extrabold text-[12px] text-[#1E293B]">{{ String(selectedDate.getMinutes()).padStart(2, '0') }}분</span>
                    <button type="button" @click="changeMinute(5)" class="p-1 rounded bg-[#E2E8F0] hover:bg-[#CBD5E1]">
                      <span class="material-symbols-outlined text-[16px] text-[#4B5563]">add</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 일정 유형 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">유형</label>
            <div class="flex flex-wrap gap-1.5">
              <button v-for="type in EVENT_TYPES" :key="type.value" type="button" @click="eventType = type.value" :class="['flex items-center gap-1 px-2.5 py-1.5 rounded-xl', eventType === type.value ? 'bg-[#EEF2FF] border border-[#C7D2FE]' : 'bg-[#F1F5F9]']">
                <span :class="['material-symbols-outlined text-[12px]', eventType === type.value ? 'text-[#355CFF]' : 'text-[#71717A]']">{{ type.icon }}</span>
                <span :class="['font-extrabold text-[11px]', eventType === type.value ? 'text-[#355CFF]' : 'text-[#64748B]']">{{ type.label }}</span>
              </button>
            </div>
          </div>

          <!-- 상태 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">상태</label>
            <div class="flex flex-wrap gap-1.5">
              <button v-for="st in STATUS_TYPES" :key="st.value" type="button" @click="status = st.value" :class="['px-2.5 py-1.5 rounded-xl', status === st.value ? 'bg-[#EEF2FF] border border-[#C7D2FE]' : 'bg-[#F1F5F9]']">
                <span :class="['font-extrabold text-[11px]', status === st.value ? 'text-[#355CFF]' : 'text-[#64748B]']">{{ st.label }}</span>
              </button>
            </div>
          </div>

          <!-- 본문 메모 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">메모</label>
            <textarea v-model="description" placeholder="세부 일정 메모" class="bg-[#F8FAFC] border border-[#E2E8F0] rounded-[10px] px-3 py-2.5 font-semibold text-[13px] text-[#0F172A] outline-none focus:border-[#355CFF] h-[70px] resize-none"></textarea>
          </div>

          <!-- 근거 텍스트 -->
          <div class="flex flex-col gap-1.5">
            <label class="font-extrabold text-[12px] text-[#475569]">근거 텍스트</label>
            <div class="bg-[#F8FAFC] p-2.5 rounded-lg border-l-4 border-l-[#94A3B8]">
              <p class="font-semibold text-[12px] text-[#475569] leading-4 line-clamp-3">
                {{ initialData?.sourceText || '선택된 스크립트가 없습니다.' }}
              </p>
            </div>
          </div>
        </div>

        <div class="flex gap-2 p-3 border-t border-[#F1F5F9]">
          <button type="button" @click="emit('close')" class="flex-1 h-[38px] flex items-center justify-center bg-white border border-[#E2E8F0] rounded-lg font-extrabold text-[13px] text-[#475569] hover:bg-gray-50">
            취소
          </button>
          <button type="button" @click="handleSave" class="flex-1 h-[38px] flex items-center justify-center bg-[#0F172A] rounded-lg font-extrabold text-[13px] text-white hover:bg-black">
            저장
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
