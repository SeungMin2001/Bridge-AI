import { computed, onMounted, onUnmounted, ref } from 'vue'

const VIEW_TO_HASH = {
  home: '#/',
  workspace: '#/workspace',
  workfolder: '#/workfolder',
  'ai-history': '#/ai-history'
}

const HASH_TO_VIEW = {
  '#/': 'home',
  '#/workspace': 'workspace',
  '#/workfolder': 'workfolder',
  '#/ai-history': 'ai-history'
}

const normalizeHash = (hash) => {
  if (!hash || hash === '#') return '#/'
  return hash.startsWith('#/') ? hash : '#/'
}

const resolveView = (hash) => {
  return HASH_TO_VIEW[normalizeHash(hash)] || 'home'
}

export function usePageNavigation() {
  const currentView = ref(resolveView(window.location.hash))

  const syncFromLocation = () => {
    const normalized = normalizeHash(window.location.hash)
    if (normalized !== window.location.hash) {
      window.location.replace(normalized)
      return
    }
    currentView.value = resolveView(normalized)
  }

  const navigateTo = (view) => {
    const nextHash = VIEW_TO_HASH[view] || VIEW_TO_HASH.home
    if (window.location.hash === nextHash) {
      currentView.value = resolveView(nextHash)
      return
    }
    window.location.hash = nextHash
  }

  const currentPath = computed(() => normalizeHash(window.location.hash))

  onMounted(() => {
    syncFromLocation()
    window.addEventListener('hashchange', syncFromLocation)
  })

  onUnmounted(() => {
    window.removeEventListener('hashchange', syncFromLocation)
  })

  return {
    currentView,
    currentPath,
    navigateTo
  }
}
