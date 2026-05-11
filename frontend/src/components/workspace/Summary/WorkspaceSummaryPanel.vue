<script setup>
import { computed, ref } from 'vue'
import { marked } from 'marked'
import { isPdfMaterial } from '../../../utils/pdfMaterial.js'

marked.setOptions({
  breaks: true,
  gfm: true,
})

const props = defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  activeSummaryTab: { type: String, default: 'summary' },
  activeFileId: { type: String, default: '' },
  currentRecordings: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  quizSource: { type: Object, default: null },
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingMode: { type: String, default: 'lecture' },
  diarizationEnabled: { type: Boolean, default: true },
  transcriptions: { type: Array, default: () => [] },
  summaryState: { type: Object, default: () => ({}) }
})

const emit = defineEmits([
  'update:activeSummaryTab',
  'generateMaterialSummary',
  'deleteSummary',
  'askAi',
  'addToNote'
])

const setSummaryTab = (tab) => {
  emit('update:activeSummaryTab', tab)
}

const getTranscriptText = (transcription) => {
  if (transcription?.segments?.length) {
    return transcription.segments
      .map((segment) => segment.text)
      .filter(Boolean)
      .join(' ')
      .trim()
  }
  return (transcription?.text || '').trim()
}

const getSpeakerSummaryKey = (transcription, index) => {
  if (transcription?.speakerId) return transcription.speakerId
  if (transcription?.speaker) return transcription.speaker
  return props.recordingMode === 'meeting' ? `unknown-speaker-${index}` : 'me'
}

const getTranscriptionSpeakerId = (transcription = {}) => (
  String(transcription?.speakerId || transcription?.speaker || '').trim()
)

const formatSpeakerLabel = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return `화자 ${Number.parseInt(match[1], 10) + 1}`
  if (!normalized || normalized === 'UNKNOWN') return '화자 미상'
  return normalized
}

const isUnknownSpeaker = (speakerId = '') => {
  const normalized = String(speakerId || '').trim()
  return !normalized || normalized === 'UNKNOWN' || normalized === '화자 미상'
}

const getSpeakerAccent = (speakerLabel) => {
  if (speakerLabel === '나' || speakerLabel === '화자 1') return 'blue'
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return 'rose'
  if (speakerLabel === '화자 C' || speakerLabel === '화자 3') return 'green'
  return 'amber'
}

const getSpeakerAvatarSrc = (speakerLabel) => {
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return '/images/man1.png'
  if (speakerLabel === '화자 3') return '/images/woman2.png'
  if (speakerLabel === '화자 4') return '/images/man2.png'
  return '/images/woman1.png'
}

const buildMockSpeakerSummary = (utterances) => {
  const texts = utterances.map((utterance) => utterance.text).filter(Boolean)
  if (texts.length <= 1) return texts[0] || ''
  return texts.slice(-2).join(' ')
}

// 화면의 화자 번호는 SPEAKER_00 숫자가 아니라 녹음에서 처음 등장한 순서로 부여합니다.
const speakerOrderTranscriptions = computed(() => {
  const liveTranscriptions = Array.isArray(props.transcriptions)
    ? props.transcriptions.filter((transcription) => getTranscriptText(transcription))
    : []
  if (liveTranscriptions.length) return liveTranscriptions

  const recordingId = props.summaryState?.recordingId || ''
  const recording = props.currentRecordings.find((item) => (
    recordingId && (item?.id === recordingId || item?.recordingId === recordingId)
  ))
  return Array.isArray(recording?.transcriptions) ? recording.transcriptions : []
})

const speakerDisplayMap = computed(() => {
  const speakerMap = new Map()

  speakerOrderTranscriptions.value.forEach((transcription) => {
    const speakerId = getTranscriptionSpeakerId(transcription)
    if (isUnknownSpeaker(speakerId) || speakerMap.has(speakerId)) return

    const order = speakerMap.size
    speakerMap.set(speakerId, {
      label: `화자 ${order + 1}`,
      order
    })
  })

  return speakerMap
})

const getDisplaySpeakerLabel = (speakerId = '', fallbackLabel = '') => {
  const normalized = String(speakerId || '').trim()
  return speakerDisplayMap.value.get(normalized)?.label || fallbackLabel || formatSpeakerLabel(normalized)
}

const getDisplaySpeakerOrder = (speakerId = '', fallbackIndex = 0) => {
  const normalized = String(speakerId || '').trim()
  const mappedOrder = speakerDisplayMap.value.get(normalized)?.order
  if (Number.isFinite(mappedOrder)) return mappedOrder

  const match = normalized.match(/^SPEAKER_(\d+)$/i)
  if (match) return Number.parseInt(match[1], 10)

  return fallbackIndex
}

const formatSummaryTime = (value = '') => {
  if (!value) return ''
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return ''
  return parsed.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
}

const materialFromSource = (source = {}) => (
  source?.material || (
    source?.type === 'material'
      ? {
          id: source.id,
          name: source.title,
          title: source.title,
          type: source.fileType || source.mimeType || source.contentType || '',
          url: source.url || '',
          storedName: source.storedName || ''
        }
      : null
  )
)

const selectedPdfMaterials = computed(() => {
  const materials = []
  const seen = new Set()

  const addMaterial = (material) => {
    if (!material || !isPdfMaterial(material)) return
    const key = material.id || material.storedName || material.url || material.name
    if (!key || seen.has(key)) return
    seen.add(key)
    materials.push(material)
  }

  if (Array.isArray(props.quizSource?.sources)) {
    props.quizSource.sources.forEach((source) => addMaterial(materialFromSource(source)))
  }

  addMaterial(props.quizSource?.material)

  if (props.quizSource?.type !== 'recording' && isPdfMaterial(props.currentPreviewMaterial)) {
    addMaterial(props.currentPreviewMaterial)
  }

  return materials
})

const selectedMaterialLabel = computed(() => {
  if (!selectedPdfMaterials.value.length) return 'PDF 선택 없음'
  const first = selectedPdfMaterials.value[0]
  return selectedPdfMaterials.value.length === 1
    ? (first.name || first.title || 'PDF 강의자료')
    : `${first.name || first.title || 'PDF 강의자료'} 외 ${selectedPdfMaterials.value.length - 1}개`
})

const selectedMaterialMeta = computed(() => (
  selectedPdfMaterials.value.length
    ? `선택 ${selectedPdfMaterials.value.length}개 · PDF ${selectedPdfMaterials.value.length}개 연결됨`
    : '좌측 폴더에서 PDF 강의자료를 선택하세요.'
))

const materialSummaryLevels = [
  { value: 'brief', label: '간단', icon: 'short_text', topicCount: 4 },
  { value: 'standard', label: '표준', icon: 'format_list_bulleted', topicCount: 8 },
  { value: 'detailed', label: '상세', icon: 'subject', topicCount: 14 },
  { value: 'page', label: '페이지별', icon: 'view_agenda', topicCount: 20 }
]

const selectedMaterialSummaryLevel = ref('standard')

const selectedMaterialSummaryOption = computed(() => (
  materialSummaryLevels.find((level) => level.value === selectedMaterialSummaryLevel.value)
  || materialSummaryLevels[1]
))

const getMaterialSummaryLevelLabel = (value = 'standard') => (
  value === 'textrank'
    ? 'TextRank'
    : materialSummaryLevels.find((level) => level.value === value)?.label || '표준'
)

const materialSummaryItems = computed(() => (
  Array.isArray(props.summaryState?.materialSummaries)
    ? props.summaryState.materialSummaries
    : []
))

const renderMaterialSummary = (summary = '') => marked.parse(String(summary || ''))

const isMaterialSummaryGenerating = computed(() => props.summaryState?.materialStatus === 'generating')
const materialSummaryError = computed(() => props.summaryState?.materialError || '')

const canGenerateMaterialSummary = computed(() => (
  !!props.activeFileId &&
  selectedPdfMaterials.value.length > 0 &&
  !isMaterialSummaryGenerating.value
))

const handleGenerateMaterialSummary = () => {
  if (!canGenerateMaterialSummary.value) return
  emit('generateMaterialSummary', {
    sessionId: props.quizSource?.sessionId || props.activeFileId,
    materials: selectedPdfMaterials.value,
    summaryLevel: selectedMaterialSummaryLevel.value,
    summarySentences: selectedMaterialSummaryOption.value.topicCount
  })
}

const handleDeleteSummary = (summaryId) => {
  if (!summaryId) return
  if (!confirm('이 요약을 삭제할까요?')) return
  emit('deleteSummary', summaryId)
}

// 백엔드 요약이 아직 없을 때만 현재 전사 상태에서 임시 화자 요약 카드를 만듭니다.
const speakerSummaryItems = computed(() => {
  const speakerMap = new Map()

  props.transcriptions.forEach((transcription, index) => {
    const text = getTranscriptText(transcription)
    if (!text) return

    // UNKNOWN은 화자별 요약 화면에서 제외해 "화자 미상" 카드가 생기지 않게 합니다.
    const rawSpeakerId = transcription.speakerId || transcription.speaker || ''
    if (isUnknownSpeaker(rawSpeakerId)) return

    const speakerLabel = getDisplaySpeakerLabel(rawSpeakerId, transcription.speaker || formatSpeakerLabel(transcription.speakerId))
    const speakerKey = rawSpeakerId || getSpeakerSummaryKey(transcription, index)

    if (!speakerMap.has(speakerKey)) {
      speakerMap.set(speakerKey, {
        key: speakerKey,
        label: speakerLabel,
        firstIndex: index,
        utterances: []
      })
    }

    speakerMap.get(speakerKey).utterances.push({
      text,
      time: transcription.time || '',
      order: index
    })
  })

  return Array.from(speakerMap.values()).map((speaker) => {
    const latestUtterance = speaker.utterances[speaker.utterances.length - 1]
    const accent = getSpeakerAccent(speaker.label)

    return {
      ...speaker,
      accent: {
        avatar: `speaker-summary-avatar-${accent}`,
        dot: `speaker-summary-dot-${accent}`
      },
      avatarSrc: getSpeakerAvatarSrc(speaker.label),
      utteranceCount: speaker.utterances.length,
      summary: buildMockSpeakerSummary(speaker.utterances),
      latestText: latestUtterance?.text || '',
      lastUpdatedAt: latestUtterance?.time || ''
    }
  }).sort((left, right) => left.firstIndex - right.firstIndex)
})

// 백엔드가 저장한 화자별 최신 요약을 화면 모델로 바꿉니다.
const backendSpeakerSummaryItems = computed(() => {
  const items = Array.isArray(props.summaryState?.speakerSummaries)
    ? props.summaryState.speakerSummaries
    : []

  return items.map((item, index) => {
    const speakerId = item.speakerId || item.label || ''
    const label = getDisplaySpeakerLabel(speakerId, item.label || '화자')
    const accent = getSpeakerAccent(label)
    return {
      id: item.id,
      key: item.key || item.id || `backend-speaker-${index}`,
      label,
      speakerId,
      firstIndex: getDisplaySpeakerOrder(speakerId, index),
      accent: {
        avatar: `speaker-summary-avatar-${accent}`,
        dot: `speaker-summary-dot-${accent}`
      },
      avatarSrc: getSpeakerAvatarSrc(label),
      utteranceCount: null,
      summary: item.summary || '',
      latestText: item.latestText || '',
      lastUpdatedAt: formatSummaryTime(item.createdAt)
    }
  }).sort((left, right) => left.firstIndex - right.firstIndex)
})

const isSpeakerSummaryEnabled = computed(() => (
  props.summaryState?.diarizationEnabled ?? props.diarizationEnabled
) !== false)
const displayedSpeakerSummaryItems = computed(() => {
  if (!isSpeakerSummaryEnabled.value) return []
  return backendSpeakerSummaryItems.value.length
    ? backendSpeakerSummaryItems.value
    : speakerSummaryItems.value
})
const sessionSummary = computed(() => props.summaryState?.sessionSummary || null)
const sessionSummaryTitle = computed(() => {
  const recordingId = sessionSummary.value?.recordingId || props.summaryState?.recordingId || ''
  const recording = props.currentRecordings.find((item) => (
    recordingId && (item?.id === recordingId || item?.recordingId === recordingId)
  ))
  const recordingTitle = recording?.title || recording?.name || ''
  return recordingTitle ? `${recordingTitle} 요약` : '녹음본 요약'
})
const isSummaryGenerating = computed(() => ['loading', 'generating'].includes(props.summaryState?.status))
const isLiveSummary = computed(() => props.summaryState?.status === 'live')
const hasSpeakerSummaries = computed(() => displayedSpeakerSummaryItems.value.length > 0)
const hasMaterialSummaries = computed(() => materialSummaryItems.value.length > 0)
</script>

<template>
  <section :class="['tab-content flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-4', tabAnim]">
    <div class="max-w-5xl mx-auto w-full h-full flex flex-col min-h-0">
      <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-5 pb-0">
        <nav class="flex gap-8">
          <div class="relative cursor-pointer summary-subtab-btn group" @click="setSummaryTab('summary')">
            <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'summary' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">요약&nbsp;&nbsp;</button>
            <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'summary' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
          </div>
          <div class="relative cursor-pointer summary-subtab-btn group" @click="setSummaryTab('material')">
            <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'material' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">자료요약&nbsp;&nbsp;</button>
            <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'material' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
          </div>
        </nav>
      </div>

      <div v-show="activeSummaryTab === 'summary'" class="summary-subcontent ai-summary-panel flex-1 min-h-0">
        <div v-if="isSummaryGenerating" class="ai-summary-empty">
          <span class="material-symbols-outlined text-[42px] text-[#c7c7cc]">hourglass_top</span>
          <p>{{ summaryState?.status === 'generating' ? 'AI 요약을 생성하고 있습니다.' : '저장된 요약을 불러오고 있습니다.' }}</p>
        </div>

        <div v-else-if="!hasSpeakerSummaries && !sessionSummary" class="ai-summary-empty">
          <span class="material-symbols-outlined text-[42px] text-[#c7c7cc]">summarize</span>
          <p>{{ summaryState?.error || (isLiveSummary ? '실시간 요약을 준비하고 있습니다.' : '아직 요약된 발화가 없습니다.') }}</p>
        </div>

        <div v-else class="ai-summary-list">
          <article
            v-if="sessionSummary"
            class="speaker-summary-card session-summary-card transcription-item-enter"
          >
            <div class="speaker-summary-top">
              <div class="speaker-summary-identity">
                <div class="speaker-summary-avatar speaker-summary-avatar-blue">
                  <span class="material-symbols-outlined">summarize</span>
                </div>
                <div class="min-w-0">
                  <h3>{{ sessionSummaryTitle }}</h3>
                  <p>{{ formatSummaryTime(sessionSummary.createdAt) || '저장된 요약' }}</p>
                </div>
              </div>

              <div class="speaker-summary-actions">
                <div class="speaker-summary-status">
                  <span class="speaker-summary-dot speaker-summary-dot-blue"></span>
                  <span>전체</span>
                </div>
                <button
                  v-if="sessionSummary.id"
                  type="button"
                  class="summary-delete-button"
                  title="요약 삭제"
                  @click="handleDeleteSummary(sessionSummary.id)"
                >
                  <span class="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>

            <p class="speaker-summary-text">{{ sessionSummary.summary }}</p>
          </article>

          <article
            v-for="speaker in displayedSpeakerSummaryItems"
            :key="speaker.key"
            class="speaker-summary-card transcription-item-enter"
            :style="{ animationDelay: `${speaker.firstIndex * 0.05}s` }"
          >
            <div class="speaker-summary-top">
              <div class="speaker-summary-identity">
                <div class="speaker-summary-avatar" :class="speaker.accent.avatar">
                  <img :src="speaker.avatarSrc" :alt="speaker.label" />
                </div>
                <div class="min-w-0">
                  <h3>{{ speaker.label }}</h3>
                  <p>
                    <span v-if="speaker.utteranceCount">발화 {{ speaker.utteranceCount }}개</span>
                    <span v-else>저장된 요약</span>
                    <span v-if="speaker.lastUpdatedAt">· {{ speaker.lastUpdatedAt }}</span>
                  </p>
                </div>
              </div>

              <div class="speaker-summary-actions">
                <div class="speaker-summary-status">
                  <span :class="['speaker-summary-dot', speaker.accent.dot]"></span>
                  <span>{{ isRecording && !isRecordingPaused ? '실시간' : '요약' }}</span>
                </div>
                <button
                  v-if="speaker.id"
                  type="button"
                  class="summary-delete-button"
                  title="요약 삭제"
                  @click="handleDeleteSummary(speaker.id)"
                >
                  <span class="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>

            <p class="speaker-summary-text">{{ speaker.summary }}</p>

            <div v-if="speaker.latestText" class="speaker-summary-latest">
              <span class="material-symbols-outlined">graphic_eq</span>
              <span>{{ speaker.latestText }}</span>
            </div>
          </article>
        </div>
      </div>

      <div v-show="activeSummaryTab === 'material'" class="summary-subcontent ai-summary-panel flex-1 min-h-0">
        <section :class="['material-summary-source-card', { empty: !selectedPdfMaterials.length }]">
          <div class="material-source-icon">
            <span class="material-symbols-outlined">draft</span>
          </div>
          <div class="material-source-main">
            <span class="material-source-label">선택된 파일</span>
            <strong>{{ selectedMaterialLabel }}</strong>
            <p>{{ selectedMaterialMeta }}</p>
            <div v-if="selectedPdfMaterials.length" class="material-source-list">
              <span
                v-for="material in selectedPdfMaterials.slice(0, 4)"
                :key="material.id || material.storedName || material.url || material.name"
                class="material-source-chip"
              >
                <span class="material-symbols-outlined">picture_as_pdf</span>
                {{ material.name || material.title || material.storedName || 'PDF 강의자료' }}
              </span>
              <span v-if="selectedPdfMaterials.length > 4" class="material-source-more">
                +{{ selectedPdfMaterials.length - 4 }}
              </span>
            </div>
            <div class="material-summary-levels" role="radiogroup" aria-label="자료 요약 단계">
              <button
                v-for="level in materialSummaryLevels"
                :key="level.value"
                type="button"
                :class="['material-summary-level-button', { active: selectedMaterialSummaryLevel === level.value }]"
                :aria-checked="selectedMaterialSummaryLevel === level.value"
                role="radio"
                :title="`${level.label} 요약`"
                @click="selectedMaterialSummaryLevel = level.value"
              >
                <span class="material-symbols-outlined">{{ level.icon }}</span>
                <span>{{ level.label }}</span>
              </button>
            </div>
          </div>
          <button
            type="button"
            class="material-summary-button"
            :disabled="!canGenerateMaterialSummary"
            @click="handleGenerateMaterialSummary"
          >
            <span class="material-symbols-outlined">{{ isMaterialSummaryGenerating ? 'hourglass_top' : 'summarize' }}</span>
            <span>{{ isMaterialSummaryGenerating ? '요약 중' : '요약 생성' }}</span>
          </button>
        </section>

        <div v-if="materialSummaryError" class="material-summary-error">
          <span class="material-symbols-outlined">error</span>
          <span>{{ materialSummaryError }}</span>
        </div>

        <div v-if="isMaterialSummaryGenerating" class="ai-summary-empty compact">
          <span class="material-symbols-outlined text-[42px] text-[#c7c7cc]">hourglass_top</span>
          <p>선택한 PDF 강의자료를 요약하고 있습니다.</p>
        </div>

        <div v-else-if="!hasMaterialSummaries" class="ai-summary-empty compact">
          <span class="material-symbols-outlined text-[42px] text-[#c7c7cc]">picture_as_pdf</span>
          <p>아직 생성된 자료 요약이 없습니다.</p>
        </div>

        <div v-else class="ai-summary-list">
          <article
            v-for="material in materialSummaryItems"
            :key="material.key"
            class="speaker-summary-card material-summary-card transcription-item-enter"
          >
            <div class="speaker-summary-top">
              <div class="speaker-summary-identity">
                <div class="speaker-summary-avatar material-summary-avatar">
                  <span class="material-symbols-outlined">picture_as_pdf</span>
                </div>
                <div class="min-w-0">
                  <h3>{{ material.title || 'PDF 요약' }}</h3>
                  <p>{{ formatSummaryTime(material.createdAt) || '저장된 파일 요약' }}</p>
                </div>
              </div>

              <div class="speaker-summary-actions">
                <div class="speaker-summary-status material-summary-status">
                  <span class="speaker-summary-dot speaker-summary-dot-rose"></span>
                  <span>PDF · {{ getMaterialSummaryLevelLabel(material.summaryLevel) }}</span>
                </div>
                <button
                  v-if="material.id"
                  type="button"
                  class="summary-delete-button"
                  title="요약 삭제"
                  @click="handleDeleteSummary(material.id)"
                >
                  <span class="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>

            <div class="material-summary-markdown-wrap">
              <div
                class="material-summary-markdown"
                v-html="renderMaterialSummary(material.summary)"
              ></div>
            </div>

            <div v-if="material.sourceMaterials?.length" class="material-summary-sources">
              <span
                v-for="source in material.sourceMaterials.slice(0, 3)"
                :key="source.id || source.storedName || source.name"
              >
                {{ source.name || source.storedName || 'PDF 강의자료' }}
              </span>
              <span v-if="material.sourceMaterials.length > 3">
                +{{ material.sourceMaterials.length - 3 }}
              </span>
            </div>
          </article>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.ai-summary-panel {
  overflow-y: auto;
  padding: 2px 2px 120px;
}

.material-summary-source-card {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 16px;
  padding: 18px;
  border: 1px solid rgba(229, 229, 234, 0.9);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 16px 34px rgba(148, 163, 184, 0.08);
}

.material-summary-source-card.empty {
  background: rgba(248, 250, 252, 0.92);
}

.material-source-icon {
  flex: 0 0 auto;
  width: 58px;
  height: 58px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #1f2937;
  background: #eef2f7;
}

.material-source-icon .material-symbols-outlined {
  font-size: 30px;
}

.material-source-main {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.material-source-label {
  color: #64748b;
  font-size: 12px;
  font-weight: 900;
}

.material-source-main strong {
  overflow: hidden;
  color: #1f2937;
  font-size: 18px;
  font-weight: 900;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.material-source-main p {
  margin: 0;
  color: #8e8e93;
  font-size: 13px;
  font-weight: 850;
}

.material-source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 4px;
}

.material-summary-levels {
  width: fit-content;
  max-width: 100%;
  display: inline-grid;
  grid-template-columns: repeat(4, minmax(74px, 1fr));
  gap: 3px;
  margin-top: 2px;
  padding: 3px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 8px;
  background: #f8fafc;
}

.material-summary-level-button {
  min-width: 0;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 0 9px;
  border: 0;
  border-radius: 6px;
  color: #64748b;
  background: transparent;
  font-size: 12px;
  font-weight: 900;
  white-space: nowrap;
  transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
}

.material-summary-level-button.active {
  color: #111827;
  background: #ffffff;
  box-shadow: 0 1px 5px rgba(15, 23, 42, 0.12);
}

.material-summary-level-button .material-symbols-outlined {
  flex: 0 0 auto;
  font-size: 15px;
}

.material-source-chip,
.material-source-more {
  max-width: 220px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  overflow: hidden;
  padding: 6px 9px;
  border-radius: 8px;
  color: #1f2937;
  background: #eef4ff;
  font-size: 12px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.material-source-chip .material-symbols-outlined {
  flex: 0 0 auto;
  color: #2563eb;
  font-size: 16px;
}

.material-summary-button {
  flex: 0 0 auto;
  height: 46px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 0 16px;
  border-radius: 8px;
  color: #ffffff;
  background: #1d1d1f;
  font-size: 13px;
  font-weight: 900;
}

.material-summary-button:disabled {
  color: #94a3b8;
  background: #e5e7eb;
  cursor: not-allowed;
}

.material-summary-button .material-symbols-outlined {
  font-size: 17px;
}

.material-summary-error {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: 8px;
  color: #b91c1c;
  background: #fef2f2;
  font-size: 12px;
  font-weight: 800;
}

.material-summary-error .material-symbols-outlined {
  font-size: 17px;
}

.ai-summary-empty {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #8e8e93;
  font-size: 14px;
  font-weight: 700;
}

.ai-summary-empty.compact {
  min-height: 180px;
}

.ai-summary-list {
  display: grid;
  gap: 14px;
}

.speaker-summary-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid rgba(229, 229, 234, 0.9);
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 16px 34px rgba(148, 163, 184, 0.08);
}

.speaker-summary-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.speaker-summary-identity {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.speaker-summary-avatar {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  padding: 4px;
  overflow: hidden;
  border: 1.5px solid #1d1d1f;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.84),
    0 1px 0 rgba(255, 255, 255, 0.9);
}

.speaker-summary-avatar img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center bottom;
  border-radius: 999px;
  display: block;
}

.speaker-summary-avatar .material-symbols-outlined {
  color: #1d4ed8;
  font-size: 22px;
}

.speaker-summary-avatar-blue {
  background: #dbeafe;
}

.speaker-summary-avatar-amber {
  background: #fff1d6;
}

.speaker-summary-avatar-rose {
  background: #ffe4ea;
}

.speaker-summary-avatar-green {
  background: #dcfce7;
}

.material-summary-avatar {
  background: #fee2e2;
}

.material-summary-avatar .material-symbols-outlined {
  color: #dc2626;
}

.speaker-summary-identity h3 {
  margin: 0;
  overflow: hidden;
  color: #1d1d1f;
  font-size: 14px;
  font-weight: 900;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.speaker-summary-identity p {
  margin: 4px 0 0;
  color: #8e8e93;
  font-size: 11px;
  font-weight: 800;
  line-height: 1.2;
}

.speaker-summary-status {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #6b7280;
  font-size: 11px;
  font-weight: 900;
}

.speaker-summary-actions {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.summary-delete-button {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  color: #c7c7cc;
  background: transparent;
  transition: background 0.18s ease, color 0.18s ease;
}

.summary-delete-button:hover {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.summary-delete-button .material-symbols-outlined {
  font-size: 18px;
}

.speaker-summary-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
}

.speaker-summary-dot-blue {
  background: #2563eb;
}

.speaker-summary-dot-amber {
  background: #d97706;
}

.speaker-summary-dot-rose {
  background: #e11d48;
}

.speaker-summary-dot-green {
  background: #059669;
}

.speaker-summary-text {
  margin: 0;
  color: #1f2937;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.7;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.material-summary-markdown-wrap {
  position: relative;
  color: #1f2937;
}

.material-summary-markdown {
  font-size: 14px;
  font-weight: 700;
  line-height: 1.75;
  word-break: keep-all;
  overflow-wrap: anywhere;
}

.material-summary-markdown :deep(p) {
  margin: 0 0 10px;
}

.material-summary-markdown :deep(ol) {
  display: grid;
  gap: 18px;
  margin: 0;
  padding-left: 22px;
}

.material-summary-markdown :deep(ul) {
  display: grid;
  gap: 8px;
  margin: 10px 0 0;
  padding-left: 20px;
}

.material-summary-markdown :deep(li) {
  padding-left: 2px;
}

.material-summary-markdown :deep(ol > li) {
  color: #111827;
  font-size: 16px;
  font-weight: 900;
  line-height: 1.55;
}

.material-summary-markdown :deep(ol > li > ul) {
  color: #1f2937;
  font-size: 14px;
  font-weight: 750;
}

.material-summary-markdown :deep(ol > li > ul > li) {
  line-height: 1.75;
}

.material-summary-markdown :deep(strong) {
  color: #111827;
  font-weight: 950;
}

.speaker-summary-latest {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  padding-top: 10px;
  border-top: 1px solid rgba(229, 229, 234, 0.8);
  color: #6b7280;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.5;
}

.speaker-summary-latest .material-symbols-outlined {
  flex: 0 0 auto;
  margin-top: 1px;
  color: #9ca3af;
  font-size: 15px;
}

.material-summary-card {
  border-color: rgba(254, 202, 202, 0.9);
  background: rgba(255, 255, 255, 0.9);
}

.material-summary-status {
  color: #991b1b;
}

.material-summary-sources {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding-top: 10px;
  border-top: 1px solid rgba(254, 226, 226, 0.9);
}

.material-summary-sources span {
  max-width: 220px;
  overflow: hidden;
  padding: 4px 8px;
  border-radius: 999px;
  color: #991b1b;
  background: #fee2e2;
  font-size: 11px;
  font-weight: 850;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
