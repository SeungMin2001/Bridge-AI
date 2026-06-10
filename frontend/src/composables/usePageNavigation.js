import { computed, onMounted, onUnmounted, ref } from 'vue'

// 브라우저 hash 경로와 앱 내부 화면 이름을 매핑합니다.
const VIEW_TO_HASH = {
  home: '#/',
  workspace: '#/workspace',
  workfolder: '#/workfolder',
  schedule: '#/schedule',
  'ai-history': '#/ai-history'
}

const HASH_TO_VIEW = {
  '#/': 'home',
  '#/workspace': 'workspace',
  '#/workfolder': 'workfolder',
  '#/schedule': 'schedule',
  '#/ai-history': 'ai-history'
}

// 비어 있거나 지원하지 않는 hash를 홈 경로로 보정합니다.
const normalizeHash = (hash) => {
  if (!hash || hash === '#') return '#/'
  return hash.startsWith('#/') ? hash : '#/'
}

// 현재 hash에 해당하는 화면 이름을 반환합니다.
const resolveView = (hash) => {
  return HASH_TO_VIEW[normalizeHash(hash)] || 'home'
}

// hash 기반 페이지 전환 상태를 관리합니다.
export function usePageNavigation() {
  const currentView = ref(resolveView(window.location.hash))

  // 브라우저 주소창 hash가 바뀌면 현재 화면 상태를 동기화합니다.
  const syncFromLocation = () => {
    const normalized = normalizeHash(window.location.hash)
    if (normalized !== window.location.hash) {
      window.location.replace(normalized)
      return
    }
    currentView.value = resolveView(normalized)
  }

  // 앱 내부 화면 이름을 받아 hash 경로를 변경합니다.
  const navigateTo = (view) => {
    const nextHash = VIEW_TO_HASH[view] || VIEW_TO_HASH.home
    const shouldReload = ['home', 'workfolder', 'schedule'].includes(view)

    if (window.location.hash === nextHash) {
      currentView.value = resolveView(nextHash)
      if (shouldReload) {
        window.location.reload()
      }
      return
    }
    window.location.hash = nextHash
    if (shouldReload) {
      setTimeout(() => {
        window.location.reload()
      }, 50)
    }
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
