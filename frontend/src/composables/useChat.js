import { ref } from 'vue'

// 홈/워크스페이스 AI 채팅에서 공유하는 메시지 목록입니다.
const messages = ref([])

// ═══ 근거 확인 팝오버 상태 (전역) ═══
const showCitePopover = ref(false)
const currentCite = ref(null)
const citePopoverPos = ref({ x: 0, y: 0 })

// ═══ 단어 팝오버 / 정보 상태 (전역) ═══
const selectedWordData = ref(null)

// 전사 단어를 클릭했을 때 보여줄 임시 설명 사전입니다.
const WORD_EXPLANATIONS = {
  "기초": { desc: "어떤 지식이나 기술 따위의 바탕이 되는 토대입니다.", source: "강의 교안 Chapter 1" },
  "네트워크": { desc: "여러 대의 컴퓨터나 통신기기를 통신망으로 연결하여 데이터를 주고받는 가상의 연결 체계입니다.", source: "IT 용어 대사전" },
  "OSI": { desc: "Open Systems Interconnection의 약자로, 국제표준화기구(ISO)에서 제정한 네트워크 통신 계층 모델입니다.", source: "네트워크 개론 p.42" },
  "전사": { desc: "음성이나 말소리를 텍스트 형태의 글자로 옮겨 적는 작업을 의미합니다.", source: "언어학 입문" },
  "백엔드": { desc: "사용자의 눈에 보이지 않는 서버 측의 로직, 데이터베이스 관리, API 등을 처리하는 영역입니다.", source: "풀스택 개발 가이드" },
  "데이터": { desc: "컴퓨터가 처리할 수 있는 문자, 숫자, 소리, 그림 따위의 가공되지 않은 정보의 단위입니다.", source: "데이터 정보학" },
  "테스트": { desc: "어떤 사물이나 기능이 정해진 목적에 잘 맞는지 확인하고 검사하는 과정입니다.", source: "소프트웨어 공학" },
  "샘플": { desc: "실제 제품이나 서비스의 상태를 미리 보여주기 위해 예본으로 만든 표본입니다.", source: "UI/UX 디자인 시스템" },
  "실시간": { desc: "데이터가 발생하는 즉시 또는 아주 짧은 지연 시간 내에 처리되는 방식을 의미합니다.", source: "운영체제론" },
}

// 채팅 메시지, 출처 팝오버, 단어 설명 팝오버 상태를 관리합니다.
export function useChat() {
  // 사용자/AI 메시지를 채팅 기록에 추가합니다.
  const addMessage = (message) => {
    messages.value.push(message)
  }

  // 스트리밍 중인 마지막 AI 메시지에 토큰/출처/상태를 덧씌웁니다.
  const updateLastAiMessage = (updates) => {
    if (messages.value.length > 0) {
      const lastIndex = messages.value.length - 1
      if (messages.value[lastIndex].role === 'ai') {
        messages.value[lastIndex] = { ...messages.value[lastIndex], ...updates }
      }
    }
  }

  // 현재 세션의 채팅 메시지를 모두 비웁니다.
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

  // 클릭한 단어를 정리해 등록된 설명이 있으면 표시하고, 없으면 기본 안내를 보여줍니다.
  const selectWord = (word) => {
    const cleanWord = word.replace(/[.,]/g, '')
    const data = WORD_EXPLANATIONS[cleanWord] || {
      desc: "해당 단어에 대한 상세 설명 정보가 아직 등록되지 않았습니다. AI를 사용하여 자동으로 검색하거나 노트를 추가할 수 있습니다.",
      source: "AI 분석 결과"
    }
    selectedWordData.value = {
      word: cleanWord,
      desc: data.desc,
      source: data.source
    }
  }

  // 단어 설명 팝오버를 닫습니다.
  const clearSelectedWord = () => {
    selectedWordData.value = null
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
    closeCitePopover,
    selectedWordData,
    selectWord,
    clearSelectedWord
  }
}
