import { ref } from 'vue'

const messages = ref([])

// ═══ 근거 확인 팝오버 상태 (전역) ═══
const showCitePopover = ref(false)
const currentCite = ref(null)
const citePopoverPos = ref({ x: 0, y: 0 })

export function useChat() {
  const addMessage = (message) => {
    messages.value.push(message)
  }

  const updateLastAiMessage = (updates) => {
    if (messages.value.length > 0) {
      const lastIndex = messages.value.length - 1
      if (messages.value[lastIndex].role === 'ai') {
        messages.value[lastIndex] = { ...messages.value[lastIndex], ...updates }
      }
    }
  }

  const clearHistory = () => {
    messages.value = []
  }

  // 팝오버 열기
  const openCitePopover = (cite, x = 0, y = 0) => {
    if (!cite) return
    currentCite.value = cite
    citePopoverPos.value = { x, y }
    showCitePopover.value = true
  }

  // 팝오버 닫기
  const closeCitePopover = () => {
    showCitePopover.value = false
    currentCite.value = null
  }

  return {
    messages,
    addMessage,
    updateLastAiMessage,
    clearHistory,
    showCitePopover,
    currentCite,
    citePopoverPos,
    openCitePopover,
    closeCitePopover
  }
}
