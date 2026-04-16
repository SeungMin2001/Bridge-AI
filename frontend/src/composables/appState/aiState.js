import { ref } from 'vue'

export function useAiState({ isRightSidebarVisible }) {
  const isRightSidebarVisibleRef = isRightSidebarVisible
  const summaryNotes = ref([
    { id: 1, text: '데모 데이터: AI가 전사한 내용을 여기에 정리할 수 있습니다.', source: 'AI 분석 결과', time: '12:00 PM' }
  ])
  const aiInput = ref('')

  const handleRightSidebarToggle = () => {
    isRightSidebarVisibleRef.value = !isRightSidebarVisibleRef.value
  }

  const handleAddToNote = (text, source) => {
    const now = new Date()
    summaryNotes.value.push({
      id: Date.now(),
      text,
      source: source || 'AI 분석 결과',
      time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
    })
  }

  const handleAskAi = (word) => {
    aiInput.value = word
    isRightSidebarVisibleRef.value = true
  }

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
