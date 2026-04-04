import { ref } from 'vue'

const messages = ref([])

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

  return {
    messages,
    addMessage,
    updateLastAiMessage,
    clearHistory
  }
}
