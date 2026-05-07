<script setup>
import { computed, ref, watch } from 'vue'
import { isWorkspaceUuid } from '../../../api/workspaceApi.js'

const QUIZ_API_BASE = '/quiz'

const props = defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  activeFileName: { type: String, default: '' },
  activeFileId: { type: String, default: '' },
  quizSource: { type: Object, default: null }
})

const quizStatus = ref('idle')
const quizError = ref('')
const quizMode = ref('create')
const quizList = ref([])
const activeQuiz = ref(null)
const quizAnswers = ref({})
const quizResult = ref(null)
const quizTypeCounts = ref({
  MULTIPLE_CHOICE: 3,
  OX: 1,
  SHORT_ANSWER: 1
})

const quizModes = [
  { key: 'create', label: '만들기', icon: 'tune' },
  { key: 'list', label: '퀴즈 목록', icon: 'format_list_bulleted' },
  { key: 'solve', label: '문제 풀기', icon: 'edit_note' }
]

const quizTypeOptions = [
  {
    key: 'MULTIPLE_CHOICE',
    label: '객관식',
    icon: 'checklist',
    description: '보기 중 하나를 선택'
  },
  {
    key: 'OX',
    label: 'O/X',
    icon: 'rule',
    description: '참과 거짓을 판단'
  },
  {
    key: 'SHORT_ANSWER',
    label: '단답형',
    icon: 'short_text',
    description: '핵심 답안을 직접 입력'
  }
]

const formatQuizApiDetail = (detail, fallback = '') => {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.message || JSON.stringify(item))
      .join('\n')
  }
  if (typeof detail === 'object') {
    return detail.message || detail.error || JSON.stringify(detail)
  }
  return String(detail)
}

const requestQuizJson = async (endpoint, options = {}) => {
  const response = await fetch(`${QUIZ_API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  })

  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const parsed = raw ? JSON.parse(raw) : {}
      detail = formatQuizApiDetail(parsed.detail || parsed.error, raw)
    } catch {
      detail = raw
    }
    throw new Error(detail || `Quiz API request failed: ${response.status}`)
  }

  return response.json()
}

const postQuizJson = (endpoint, payload) => requestQuizJson(endpoint, {
  method: 'POST',
  body: JSON.stringify(payload)
})

const quizSessionId = computed(() => props.quizSource?.sessionId || props.activeFileId)
const canUseQuiz = computed(() => isWorkspaceUuid(props.activeFileId))
const sourceTranscriptIds = computed(() => {
  if (!Array.isArray(props.quizSource?.transcriptIds)) return []
  const ids = new Set()
  props.quizSource.transcriptIds.forEach((id) => {
    const transcriptId = String(id || '').trim()
    if (isWorkspaceUuid(transcriptId)) ids.add(transcriptId)
  })
  return Array.from(ids)
})
const hasSelectedQuizSource = computed(() => !!props.quizSource?.title)
const hasTranscriptScope = computed(() => sourceTranscriptIds.value.length > 0)
const selectedSourceItems = computed(() => (
  Array.isArray(props.quizSource?.sources)
    ? props.quizSource.sources
    : (hasSelectedQuizSource.value ? [{
        id: props.quizSource?.materialId || props.quizSource?.recordingId || props.quizSource?.title,
        type: props.quizSource?.type || 'source',
        title: props.quizSource?.title,
        transcriptIds: sourceTranscriptIds.value
      }] : [])
))
const quizQuestionCount = computed(() => Object.values(quizTypeCounts.value).reduce((sum, count) => sum + Number(count || 0), 0))
const hasActiveQuiz = computed(() => Array.isArray(activeQuiz.value?.quiz_data) && activeQuiz.value.quiz_data.length > 0)
const isQuizBusy = computed(() => ['loading', 'generating', 'submitting'].includes(quizStatus.value))
const canGenerateQuiz = computed(() => (
  isWorkspaceUuid(quizSessionId.value) &&
  hasSelectedQuizSource.value &&
  hasTranscriptScope.value &&
  quizQuestionCount.value >= 1 &&
  quizQuestionCount.value <= 20 &&
  !isQuizBusy.value
))
const selectedSourceMeta = computed(() => {
  if (!hasSelectedQuizSource.value) return '좌측 사이드바에서 강의자료 또는 녹음본을 선택하세요.'
  if (!hasTranscriptScope.value) return '선택한 소스에 연결된 transcript_id가 없습니다.'
  const sourceCount = props.quizSource?.sourceCount || selectedSourceItems.value.length || 1
  return `선택 ${sourceCount}개 · 전사 ${sourceTranscriptIds.value.length}개 연결됨`
})
const quizScopeText = computed(() => {
  if (hasSelectedQuizSource.value) return `${props.quizSource.title} 기준`
  return props.activeFileName ? `${props.activeFileName}에서 소스를 선택하세요.` : '파일을 선택하면 퀴즈를 만들 수 있습니다.'
})
const quizProgressText = computed(() => {
  if (quizStatus.value === 'loading') return '퀴즈를 불러오고 있습니다.'
  if (quizStatus.value === 'generating') return '선택한 파일의 전사문으로 퀴즈를 만들고 있습니다.'
  if (quizStatus.value === 'submitting') return '답안을 채점하고 있습니다.'
  return ''
})

const getQuizQuestions = (quiz = activeQuiz.value) => (
  Array.isArray(quiz?.quiz_data) ? quiz.quiz_data : []
)

const formatQuizDate = (value = '') => {
  if (!value) return '방금 생성됨'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return '저장된 퀴즈'
  return parsed.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getQuizTypeLabel = (type = '') => {
  if (type === 'MULTIPLE_CHOICE') return '객관식'
  if (type === 'OX') return 'O/X'
  if (type === 'SHORT_ANSWER') return '단답형'
  return '문항'
}

const getQuizSummaryLabel = (quiz) => {
  if (quiz?.correct_count === null || quiz?.correct_count === undefined) return '미응시'
  return `${quiz.correct_count}/${quiz.total_questions || getQuizQuestions(quiz).length} 정답`
}

const buildQuizListItem = (quiz) => ({
  quiz_id: quiz.quiz_id,
  session_id: quiz.session_id || props.activeFileId,
  total_questions: quiz.total_questions || getQuizQuestions(quiz).length,
  correct_count: quiz.correct_count ?? null,
  created_at: quiz.created_at || new Date().toISOString()
})

const resetQuizState = () => {
  quizStatus.value = 'idle'
  quizError.value = ''
  quizList.value = []
  activeQuiz.value = null
  quizAnswers.value = {}
  quizResult.value = null
  quizMode.value = 'create'
}

const resetActiveQuiz = () => {
  activeQuiz.value = null
  quizAnswers.value = {}
  quizResult.value = null
}

const hydrateQuizAnswers = (questions = []) => {
  quizAnswers.value = questions.reduce((answers, question) => {
    if (question?.user_answer) {
      answers[String(question.question_index)] = question.user_answer
    }
    return answers
  }, {})
}

const setActiveQuiz = (quiz) => {
  activeQuiz.value = quiz
  const questions = getQuizQuestions(quiz)
  hydrateQuizAnswers(questions)
  quizResult.value = quiz?.correct_count === null || quiz?.correct_count === undefined
    ? null
    : {
        correct_count: quiz.correct_count,
        total_questions: quiz.total_questions || questions.length,
        score: questions.length ? Math.round((quiz.correct_count / questions.length) * 1000) / 10 : 0
      }
}

const loadQuizDetail = async (quizId, nextMode = 'solve') => {
  if (!quizId) return
  quizStatus.value = 'loading'
  quizError.value = ''
  try {
    const quiz = await requestQuizJson(`/${quizId}`)
    setActiveQuiz(quiz)
    quizMode.value = nextMode
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈를 불러오지 못했습니다.'
  }
}

const loadQuizzesForSession = async () => {
  if (!canUseQuiz.value) {
    resetQuizState()
    return
  }

  quizStatus.value = 'loading'
  quizError.value = ''
  try {
    const result = await requestQuizJson(`/session/${props.activeFileId}`)
    quizList.value = Array.isArray(result.quizzes) ? result.quizzes : []
    resetActiveQuiz()
    quizMode.value = 'create'
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 목록을 불러오지 못했습니다.'
  }
}

const applyPreset = (total) => {
  const mc = Math.max(1, Math.floor(total * 0.5))
  const ox = total >= 3 ? Math.max(1, Math.floor(total * 0.3)) : 0
  const shortAnswer = Math.max(0, total - mc - ox)
  quizTypeCounts.value = {
    MULTIPLE_CHOICE: mc,
    OX: ox,
    SHORT_ANSWER: shortAnswer
  }
}

const adjustTypeCount = (type, delta) => {
  const current = Number(quizTypeCounts.value[type] || 0)
  const next = Math.max(0, current + delta)
  const nextTotal = quizQuestionCount.value - current + next
  if (nextTotal > 20) return
  quizTypeCounts.value = {
    ...quizTypeCounts.value,
    [type]: next
  }
}

const generateQuizForSource = async () => {
  if (!canGenerateQuiz.value) return

  quizStatus.value = 'generating'
  quizError.value = ''
  quizResult.value = null
  try {
    const quiz = await postQuizJson('/generate/transcripts', {
      session_id: quizSessionId.value,
      transcript_ids: sourceTranscriptIds.value,
      num_questions: quizQuestionCount.value,
      type_counts: quizTypeCounts.value
    })
    setActiveQuiz({ ...quiz, session_id: quizSessionId.value, correct_count: null })
    quizList.value = [
      buildQuizListItem({ ...quiz, session_id: quizSessionId.value, correct_count: null }),
      ...quizList.value.filter((item) => item.quiz_id !== quiz.quiz_id)
    ]
    quizMode.value = 'solve'
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 생성에 실패했습니다.'
  }
}

const setQuizAnswer = (question, answer) => {
  quizAnswers.value = {
    ...quizAnswers.value,
    [String(question.question_index)]: answer
  }
}

const isQuizOptionSelected = (question, option) => (
  quizAnswers.value[String(question.question_index)] === option
)

const submitQuizAnswers = async () => {
  if (!activeQuiz.value?.quiz_id || isQuizBusy.value || quizResult.value) return

  quizStatus.value = 'submitting'
  quizError.value = ''
  try {
    const result = await postQuizJson(`/${activeQuiz.value.quiz_id}/submit`, {
      answers: quizAnswers.value
    })
    setActiveQuiz({
      ...activeQuiz.value,
      ...result,
      correct_count: result.correct_count
    })
    quizResult.value = {
      correct_count: result.correct_count,
      total_questions: result.total_questions,
      score: result.score
    }
    quizList.value = quizList.value.map((item) => (
      item.quiz_id === result.quiz_id
        ? { ...item, correct_count: result.correct_count, total_questions: result.total_questions }
        : item
    ))
    quizStatus.value = 'done'
  } catch (error) {
    quizStatus.value = 'error'
    quizError.value = error?.message || '퀴즈 채점에 실패했습니다.'
  }
}

watch(
  () => props.activeFileId,
  () => {
    resetQuizState()
    loadQuizzesForSession()
  },
  { immediate: true }
)

watch(
  () => props.quizSource,
  () => {
    quizError.value = ''
    resetActiveQuiz()
    quizMode.value = 'create'
  },
  { deep: true }
)
</script>

<template>
  <section :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
    <div class="quiz-panel max-w-5xl mx-auto w-full h-full min-h-0">
      <div class="quiz-header">
        <div class="min-w-0">
          <h1>퀴즈</h1>
          <p>{{ quizScopeText }}</p>
        </div>
      </div>

      <div class="quiz-mode-tabs" aria-label="퀴즈 화면 선택">
        <button
          v-for="mode in quizModes"
          :key="mode.key"
          type="button"
          :class="{ active: quizMode === mode.key }"
          @click="quizMode = mode.key"
        >
          <span class="material-symbols-outlined">{{ mode.icon }}</span>
          <span>{{ mode.label }}</span>
        </button>
      </div>

      <div v-if="quizError" class="quiz-error">
        <span class="material-symbols-outlined">error</span>
        <span>{{ quizError }}</span>
      </div>

      <div v-if="!canUseQuiz" class="quiz-empty">
        <span class="material-symbols-outlined">quiz</span>
        <p>워크스페이스 파일을 선택하면 퀴즈를 만들 수 있습니다.</p>
      </div>

      <div v-else-if="quizProgressText" class="quiz-empty">
        <span class="material-symbols-outlined">hourglass_top</span>
        <p>{{ quizProgressText }}</p>
      </div>

      <div v-else-if="quizMode === 'create'" class="quiz-create-view">
        <section :class="['quiz-source-card', { empty: !hasSelectedQuizSource, blocked: hasSelectedQuizSource && !hasTranscriptScope }]">
          <div class="quiz-source-icon">
            <span class="material-symbols-outlined">{{ hasSelectedQuizSource ? 'draft' : 'touch_app' }}</span>
          </div>
          <div class="min-w-0">
            <span class="quiz-source-label">선택된 파일</span>
            <strong>{{ hasSelectedQuizSource ? quizSource.title : '선택된 파일 없음' }}</strong>
            <p>{{ selectedSourceMeta }}</p>
            <div v-if="selectedSourceItems.length" class="quiz-source-list">
              <span
                v-for="source in selectedSourceItems.slice(0, 4)"
                :key="source.id || source.title"
                class="quiz-source-chip"
              >
                <span class="material-symbols-outlined">{{ source.type === 'recording' ? 'graphic_eq' : 'draft' }}</span>
                {{ source.title }}
              </span>
              <span v-if="selectedSourceItems.length > 4" class="quiz-source-more">
                +{{ selectedSourceItems.length - 4 }}
              </span>
            </div>
          </div>
        </section>

        <section class="quiz-builder">
          <div class="quiz-builder-top">
            <div>
              <h2>퀴즈 구성</h2>
              <p>{{ quizQuestionCount }}문항 · 객관식 {{ quizTypeCounts.MULTIPLE_CHOICE }} · O/X {{ quizTypeCounts.OX }} · 단답형 {{ quizTypeCounts.SHORT_ANSWER }}</p>
            </div>
            <div class="quiz-count-presets" aria-label="문항 수 빠른 선택">
              <button
                v-for="count in [5, 10, 15]"
                :key="count"
                type="button"
                :class="{ active: quizQuestionCount === count }"
                :disabled="isQuizBusy"
                @click="applyPreset(count)"
              >
                {{ count }}
              </button>
            </div>
          </div>

          <div class="quiz-type-grid">
            <article v-for="option in quizTypeOptions" :key="option.key" class="quiz-type-card">
              <div class="quiz-type-main">
                <span class="material-symbols-outlined">{{ option.icon }}</span>
                <div>
                  <strong>{{ option.label }}</strong>
                  <p>{{ option.description }}</p>
                </div>
              </div>
              <div class="quiz-stepper" :aria-label="`${option.label} 문항 수`">
                <button
                  type="button"
                  :disabled="quizTypeCounts[option.key] <= 0 || isQuizBusy"
                  @click="adjustTypeCount(option.key, -1)"
                >
                  <span class="material-symbols-outlined">remove</span>
                </button>
                <span>{{ quizTypeCounts[option.key] }}</span>
                <button
                  type="button"
                  :disabled="quizQuestionCount >= 20 || isQuizBusy"
                  @click="adjustTypeCount(option.key, 1)"
                >
                  <span class="material-symbols-outlined">add</span>
                </button>
              </div>
            </article>
          </div>

          <div class="quiz-generate-row">
            <span v-if="quizQuestionCount < 1">문항을 1개 이상 선택하세요.</span>
            <span v-else-if="quizQuestionCount > 20">문항은 최대 20개까지 만들 수 있습니다.</span>
            <span v-else>{{ quizQuestionCount }}문항 생성 준비됨</span>
            <button
              type="button"
              class="quiz-primary-button"
              :disabled="!canGenerateQuiz"
              @click="generateQuizForSource"
            >
              <span class="material-symbols-outlined">add_task</span>
              <span>퀴즈 생성</span>
            </button>
          </div>
        </section>
      </div>

      <div v-else-if="quizMode === 'list'" class="quiz-list-view">
        <div class="quiz-list-header">
          <div>
            <h2>퀴즈 목록</h2>
            <p>{{ quizList.length }}개 저장됨</p>
          </div>
          <button type="button" class="quiz-secondary-button" @click="quizMode = 'create'">
            <span class="material-symbols-outlined">add</span>
            <span>새로 만들기</span>
          </button>
        </div>

        <div v-if="quizList.length === 0" class="quiz-empty compact">
          <span class="material-symbols-outlined">assignment</span>
          <p>아직 생성된 퀴즈가 없습니다.</p>
        </div>

        <div v-else class="quiz-session-list custom-scrollbar">
          <article
            v-for="quiz in quizList"
            :key="quiz.quiz_id"
            :class="['quiz-session-card', { active: activeQuiz?.quiz_id === quiz.quiz_id }]"
          >
            <span class="material-symbols-outlined">assignment</span>
            <div class="quiz-session-main">
              <strong>{{ quiz.total_questions || 0 }}문항 퀴즈</strong>
              <p>{{ formatQuizDate(quiz.created_at) }} · {{ getQuizSummaryLabel(quiz) }}</p>
            </div>
            <button type="button" class="quiz-secondary-button" @click="loadQuizDetail(quiz.quiz_id)">
              <span class="material-symbols-outlined">play_arrow</span>
              <span>{{ quiz.correct_count === null || quiz.correct_count === undefined ? '풀기' : '다시 보기' }}</span>
            </button>
          </article>
        </div>
      </div>

      <div v-else class="quiz-solve-view">
        <div v-if="!hasActiveQuiz" class="quiz-empty compact">
          <span class="material-symbols-outlined">edit_note</span>
          <p>퀴즈 목록에서 풀 퀴즈를 선택하거나 새 퀴즈를 생성하세요.</p>
        </div>

        <template v-else>
          <div class="quiz-solve-top">
            <div>
              <h2>{{ getQuizQuestions().length }}문항 퀴즈</h2>
              <p>{{ Object.keys(quizAnswers).length }} / {{ getQuizQuestions().length }} 응답</p>
            </div>
            <button type="button" class="quiz-secondary-button" @click="quizMode = 'list'">
              <span class="material-symbols-outlined">format_list_bulleted</span>
              <span>목록</span>
            </button>
          </div>

          <div v-if="quizResult" class="quiz-result-bar">
            <div>
              <strong>{{ quizResult.score }}점</strong>
              <span>{{ quizResult.correct_count }} / {{ quizResult.total_questions }} 정답</span>
            </div>
            <button type="button" class="quiz-secondary-button" @click="quizResult = null">
              다시 보기
            </button>
          </div>

          <div class="quiz-question-list custom-scrollbar">
            <article
              v-for="question in getQuizQuestions()"
              :key="question.question_index"
              :class="['quiz-question-card', { graded: question.is_correct !== null && question.is_correct !== undefined }]"
            >
              <div class="quiz-question-top">
                <span class="quiz-type-chip">{{ getQuizTypeLabel(question.type) }}</span>
                <span class="quiz-question-index">{{ question.question_index }}</span>
              </div>

              <h2>{{ question.question }}</h2>

              <div v-if="question.type === 'SHORT_ANSWER'" class="quiz-short-answer">
                <input
                  type="text"
                  :value="quizAnswers[String(question.question_index)] || ''"
                  :disabled="!!quizResult"
                  placeholder="답안을 입력하세요"
                  @input="setQuizAnswer(question, $event.target.value)"
                />
              </div>

              <div v-else class="quiz-options">
                <button
                  v-for="option in question.options"
                  :key="option"
                  type="button"
                  :class="['quiz-option', { selected: isQuizOptionSelected(question, option) }]"
                  :disabled="!!quizResult"
                  @click="setQuizAnswer(question, option)"
                >
                  {{ option }}
                </button>
              </div>

              <div v-if="question.is_correct !== null && question.is_correct !== undefined" class="quiz-explanation">
                <div :class="['quiz-grade', question.is_correct ? 'correct' : 'wrong']">
                  <span class="material-symbols-outlined">{{ question.is_correct ? 'check_circle' : 'cancel' }}</span>
                  <span>{{ question.is_correct ? '정답' : '오답' }}</span>
                </div>
                <p>내 답: {{ question.user_answer || '미응답' }}</p>
                <p>정답: {{ question.correct_answer }}</p>
                <p v-if="question.explanation">해설: {{ question.explanation }}</p>
              </div>
            </article>
          </div>

          <div class="quiz-submit-row">
            <span>{{ Object.keys(quizAnswers).length }} / {{ getQuizQuestions().length }} 응답</span>
            <button
              type="button"
              class="quiz-primary-button"
              :disabled="isQuizBusy || !!quizResult"
              @click="submitQuizAnswers"
            >
              <span class="material-symbols-outlined">fact_check</span>
              <span>{{ quizStatus === 'submitting' ? '채점 중' : '제출하기' }}</span>
            </button>
          </div>
        </template>
      </div>
    </div>
  </section>
</template>

<style scoped>
.quiz-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow: hidden;
}

.quiz-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid #e5e5ea;
}

.quiz-header h1 {
  margin: 0;
  color: #1d1d1f;
  font-size: 28px;
  font-weight: 900;
  line-height: 1.15;
}

.quiz-header p,
.quiz-builder-top p,
.quiz-list-header p,
.quiz-solve-top p {
  margin: 6px 0 0;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.4;
}

.quiz-mode-tabs {
  display: inline-flex;
  width: fit-content;
  gap: 4px;
  padding: 4px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f8fafc;
}

.quiz-mode-tabs button {
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 12px;
  border: 0;
  border-radius: 6px;
  color: #64748b;
  background: transparent;
  font-size: 12px;
  font-weight: 900;
  white-space: nowrap;
}

.quiz-mode-tabs button.active {
  color: #1d1d1f;
  background: #ffffff;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.12);
}

.quiz-mode-tabs .material-symbols-outlined,
.quiz-primary-button .material-symbols-outlined,
.quiz-secondary-button .material-symbols-outlined {
  font-size: 18px;
}

.quiz-create-view,
.quiz-list-view,
.quiz-solve-view {
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow: hidden;
}

.quiz-source-card,
.quiz-builder,
.quiz-session-card,
.quiz-question-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 14px 28px rgba(148, 163, 184, 0.08);
}

.quiz-source-card {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px;
}

.quiz-source-card.empty {
  border-style: dashed;
  background: #f8fafc;
}

.quiz-source-card.blocked {
  border-color: #fed7aa;
  background: #fff7ed;
}

.quiz-source-icon {
  flex: 0 0 auto;
  width: 42px;
  height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #1d1d1f;
  background: #f1f5f9;
}

.quiz-source-label {
  color: #64748b;
  font-size: 11px;
  font-weight: 900;
}

.quiz-source-card strong {
  display: block;
  overflow: hidden;
  margin-top: 3px;
  color: #1d1d1f;
  font-size: 15px;
  font-weight: 900;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-source-card p {
  margin: 4px 0 0;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 800;
}

.quiz-source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.quiz-source-chip,
.quiz-source-more {
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 210px;
  padding: 5px 8px;
  border-radius: 8px;
  background: #eef2ff;
  color: #334155;
  font-size: 11px;
  font-weight: 900;
  line-height: 1.1;
}

.quiz-source-chip {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-source-chip .material-symbols-outlined {
  flex: 0 0 auto;
  color: #2563eb;
  font-size: 14px;
}

.quiz-source-more {
  background: #f1f5f9;
  color: #64748b;
}

.quiz-builder {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
}

.quiz-builder-top,
.quiz-list-header,
.quiz-solve-top,
.quiz-generate-row,
.quiz-submit-row,
.quiz-result-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.quiz-builder-top h2,
.quiz-list-header h2,
.quiz-solve-top h2 {
  margin: 0;
  color: #1d1d1f;
  font-size: 18px;
  font-weight: 900;
  line-height: 1.2;
}

.quiz-count-presets {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f8fafc;
}

.quiz-count-presets button {
  width: 38px;
  height: 30px;
  border: 0;
  border-radius: 6px;
  color: #64748b;
  background: transparent;
  font-size: 12px;
  font-weight: 900;
}

.quiz-count-presets button.active {
  color: #1d1d1f;
  background: #ffffff;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.12);
}

.quiz-type-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.quiz-type-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  padding: 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
}

.quiz-type-main {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.quiz-type-main > .material-symbols-outlined {
  flex: 0 0 auto;
  color: #2563eb;
  font-size: 20px;
}

.quiz-type-main strong {
  display: block;
  color: #1d1d1f;
  font-size: 13px;
  font-weight: 900;
}

.quiz-type-main p {
  margin: 3px 0 0;
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.35;
}

.quiz-stepper {
  display: grid;
  grid-template-columns: 32px minmax(34px, 1fr) 32px;
  align-items: center;
  gap: 5px;
}

.quiz-stepper button {
  width: 32px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  color: #1d1d1f;
  background: #f8fafc;
}

.quiz-stepper button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}

.quiz-stepper .material-symbols-outlined {
  font-size: 17px;
}

.quiz-stepper > span {
  color: #1d1d1f;
  font-size: 18px;
  font-weight: 900;
  text-align: center;
}

.quiz-generate-row {
  padding-top: 2px;
}

.quiz-generate-row > span,
.quiz-submit-row > span {
  color: #64748b;
  font-size: 12px;
  font-weight: 900;
}

.quiz-primary-button,
.quiz-secondary-button {
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  border: 0;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 900;
  line-height: 1;
  white-space: nowrap;
}

.quiz-primary-button {
  padding: 0 14px;
  color: #ffffff;
  background: #1d1d1f;
}

.quiz-secondary-button {
  padding: 0 12px;
  color: #1d1d1f;
  background: #f2f4f7;
}

.quiz-primary-button:disabled,
.quiz-secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
}

.quiz-session-list {
  min-height: 0;
  flex: 1;
  display: grid;
  align-content: start;
  gap: 10px;
  overflow-y: auto;
  padding: 2px 2px 12px;
}

.quiz-session-card {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  padding: 12px;
}

.quiz-session-card.active {
  border-color: #1d1d1f;
  box-shadow: inset 0 0 0 1px #1d1d1f;
}

.quiz-session-card > .material-symbols-outlined {
  color: #64748b;
  font-size: 21px;
}

.quiz-session-main {
  min-width: 0;
}

.quiz-session-main strong {
  display: block;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 14px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-session-main p {
  margin: 4px 0 0;
  overflow: hidden;
  color: #8e8e93;
  font-size: 12px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.quiz-error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid #fecdd3;
  border-radius: 8px;
  color: #be123c;
  background: #fff1f2;
  font-size: 12px;
  font-weight: 900;
}

.quiz-error .material-symbols-outlined {
  font-size: 18px;
}

.quiz-empty {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #8e8e93;
  font-size: 14px;
  font-weight: 800;
  text-align: center;
}

.quiz-empty.compact {
  min-height: 220px;
}

.quiz-empty .material-symbols-outlined {
  color: #c7c7cc;
  font-size: 42px;
}

.quiz-result-bar {
  padding: 12px 14px;
  border: 1px solid #bbf7d0;
  border-radius: 8px;
  background: #f0fdf4;
}

.quiz-result-bar div {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.quiz-result-bar strong {
  color: #166534;
  font-size: 18px;
  font-weight: 900;
}

.quiz-result-bar span {
  color: #15803d;
  font-size: 12px;
  font-weight: 900;
}

.quiz-question-list {
  min-height: 0;
  flex: 1;
  display: grid;
  align-content: start;
  gap: 12px;
  overflow-y: auto;
  padding: 2px 2px 90px;
}

.quiz-question-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
}

.quiz-question-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.quiz-type-chip {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  color: #334155;
  background: #e2e8f0;
  font-size: 11px;
  font-weight: 900;
}

.quiz-question-index {
  width: 26px;
  height: 26px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #1d1d1f;
  background: #f8fafc;
  font-size: 12px;
  font-weight: 900;
}

.quiz-question-card h2 {
  margin: 0;
  color: #1f2937;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.55;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.quiz-options {
  display: grid;
  gap: 8px;
}

.quiz-option {
  min-height: 42px;
  padding: 10px 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  color: #334155;
  background: #ffffff;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.45;
  text-align: left;
}

.quiz-option.selected {
  border-color: #2563eb;
  color: #1d4ed8;
  background: #eff6ff;
}

.quiz-option:disabled {
  cursor: default;
}

.quiz-short-answer input {
  width: 100%;
  min-height: 42px;
  padding: 10px 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  color: #1f2937;
  background: #ffffff;
  font-size: 14px;
  font-weight: 800;
  outline: none;
}

.quiz-short-answer input:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.quiz-explanation {
  display: grid;
  gap: 5px;
  padding-top: 12px;
  border-top: 1px solid #e5e7eb;
  color: #475569;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.55;
}

.quiz-explanation p {
  margin: 0;
}

.quiz-grade {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  width: fit-content;
  font-size: 12px;
  font-weight: 900;
}

.quiz-grade .material-symbols-outlined {
  font-size: 17px;
}

.quiz-grade.correct {
  color: #15803d;
}

.quiz-grade.wrong {
  color: #be123c;
}

@media (max-width: 920px) {
  .quiz-builder-top,
  .quiz-list-header,
  .quiz-solve-top,
  .quiz-generate-row,
  .quiz-submit-row,
  .quiz-result-bar {
    align-items: stretch;
    flex-direction: column;
  }

  .quiz-type-grid {
    grid-template-columns: 1fr;
  }

  .quiz-mode-tabs {
    width: 100%;
  }

  .quiz-mode-tabs button {
    flex: 1;
  }

  .quiz-session-card {
    grid-template-columns: 30px minmax(0, 1fr);
  }

  .quiz-session-card .quiz-secondary-button {
    grid-column: 1 / -1;
  }
}
</style>
