import { ref, computed } from 'vue'

const sessions = ref([
  {
    id: Date.now().toString(),
    title: '새로운 대화',
    messages: [],
    createdAt: new Date().toISOString()
  }
])

const currentSessionId = ref(sessions.value[0].id)

export function useChat() {
  const currentSession = computed(() => {
    return sessions.value.find(s => s.id === currentSessionId.value) || sessions.value[0]
  })

  const messages = computed(() => currentSession.value.messages)

  const createNewSession = () => {
    const newId = Date.now().toString()
    sessions.value.unshift({
      id: newId,
      title: '새로운 대화',
      messages: [],
      createdAt: new Date().toISOString()
    })
    currentSessionId.value = newId
    return newId
  }

  const switchToSession = (id) => {
    if (sessions.value.some(s => s.id === id)) {
      currentSessionId.value = id
    }
  }

  const deleteSession = (id) => {
    const index = sessions.value.findIndex(s => s.id === id)
    if (index !== -1) {
      sessions.value.splice(index, 1)
      if (sessions.value.length === 0) {
        createNewSession()
      } else if (currentSessionId.value === id) {
        currentSessionId.value = sessions.value[0].id
      }
    }
  }

  const addMessage = (message) => {
    currentSession.value.messages.push(message)
    // Auto-update title if it's the first user message
    if (currentSession.value.messages.length === 1 && message.role === 'user') {
      const title = message.text.length > 20 
        ? message.text.substring(0, 20) + '...' 
        : message.text
      currentSession.value.title = title
    }
  }

  const updateLastAiMessage = (updates) => {
    const msgs = currentSession.value.messages
    if (msgs.length > 0) {
      const lastIndex = msgs.length - 1
      if (msgs[lastIndex].role === 'ai') {
        msgs[lastIndex] = { ...msgs[lastIndex], ...updates }
      }
    }
  }

  const clearHistory = () => {
    sessions.value = []
    createNewSession()
  }

  return {
    sessions,
    currentSessionId,
    currentSession,
    messages,
    createNewSession,
    switchToSession,
    deleteSession,
    addMessage,
    updateLastAiMessage,
    clearHistory
  }
}
