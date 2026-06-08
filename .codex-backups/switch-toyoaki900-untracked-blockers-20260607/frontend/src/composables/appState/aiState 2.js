import { ref } from 'vue'

// AI 사이드바 입력값과 정리 노트 목록을 관리합니다.
export function useAiState({ isRightSidebarVisible }) {
  const isRightSidebarVisibleRef = isRightSidebarVisible
  const summaryNotes = ref([
    { id: 1, text: '데모 데이터: AI가 전사한 내용을 여기에 정리할 수 있습니다.', source: 'AI 분석 결과', time: '12:00 PM' }
  ])
  const aiInput = ref('')

  // 우측 AI 채팅 패널을 열고 닫습니다.
  const handleRightSidebarToggle = () => {
    isRightSidebarVisibleRef.value = !isRightSidebarVisibleRef.value
  }

  // 선택한 전사/AI 결과를 정리 노트 탭에 추가합니다.
  const handleAddToNote = (text, source) => {
    const now = new Date()
    summaryNotes.value.push({
      id: Date.now(),
      text,
      source: source || 'AI 분석 결과',
      time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
    })
  }

  // 특정 단어를 AI 질문 입력창에 넣고 채팅 패널을 엽니다.
  const handleAskAi = (word) => {
    aiInput.value = word
    isRightSidebarVisibleRef.value = true
  }

  // AI 입력창의 v-model 값을 루트 상태와 동기화합니다.
  const handleAiInputUpdate = (value) => {
    aiInput.value = value
  }

  return {
    aiInput,
    summaryNotes,
    handleRightSidebarToggle,
    handleAddToNote,
    handleAskAi,
    handleAiInputUpdate
  }
}
