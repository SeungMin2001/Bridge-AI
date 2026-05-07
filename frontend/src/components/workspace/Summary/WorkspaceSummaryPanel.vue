<script setup>
import { computed } from 'vue'
import VoiceTransferSideTab from '../VoiceTransferSideTab.vue'

const props = defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  activeSummaryTab: { type: String, default: 'ai-summary' },
  isRecording: Boolean,
  isRecordingPaused: Boolean,
  recordingMode: { type: String, default: 'lecture' },
  transcriptions: { type: Array, default: () => [] },
  summaryState: { type: Object, default: () => ({}) }
})

const emit = defineEmits([
  'update:activeSummaryTab',
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

const getSpeakerAccent = (speakerLabel) => {
  if (speakerLabel === '나') return 'blue'
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return 'rose'
  if (speakerLabel === '화자 C' || speakerLabel === '화자 3') return 'green'
  return 'amber'
}

const getSpeakerAvatarSrc = (speakerLabel) => {
  if (speakerLabel === '화자 B' || speakerLabel === '화자 2') return '/images/man1.png'
  return '/images/woman1.png'
}

const buildMockSpeakerSummary = (utterances) => {
  const texts = utterances.map((utterance) => utterance.text).filter(Boolean)
  if (texts.length <= 1) return texts[0] || ''
  return texts.slice(-2).join(' ')
}

const formatSummaryTime = (value = '') => {
  if (!value) return ''
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return ''
  return parsed.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
}

const speakerSummaryItems = computed(() => {
  const speakerMap = new Map()

  props.transcriptions.forEach((transcription, index) => {
    const text = getTranscriptText(transcription)
    if (!text) return

    const speakerLabel = transcription.speaker || (props.recordingMode === 'meeting' ? '화자 미상' : '나')
    const speakerKey = getSpeakerSummaryKey(transcription, index)

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
  })
})

const backendSpeakerSummaryItems = computed(() => {
  const items = Array.isArray(props.summaryState?.speakerSummaries)
    ? props.summaryState.speakerSummaries
    : []

  return items.map((item, index) => {
    const label = item.label || '화자'
    const accent = getSpeakerAccent(label)
    return {
      key: item.key || item.id || `backend-speaker-${index}`,
      label,
      firstIndex: index,
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
  })
})

const displayedSpeakerSummaryItems = computed(() => (
  backendSpeakerSummaryItems.value.length
    ? backendSpeakerSummaryItems.value
    : speakerSummaryItems.value
))
const sessionSummary = computed(() => props.summaryState?.sessionSummary || null)
const isSummaryGenerating = computed(() => ['loading', 'generating'].includes(props.summaryState?.status))
const isLiveSummary = computed(() => props.summaryState?.status === 'live')
const hasSpeakerSummaries = computed(() => displayedSpeakerSummaryItems.value.length > 0)
</script>

<template>
  <section :class="['tab-content flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-4', tabAnim]">
    <div class="max-w-5xl mx-auto w-full h-full flex flex-col min-h-0">
      <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-5 pb-0">
        <nav class="flex gap-8">
          <div class="relative cursor-pointer summary-subtab-btn group" @click="setSummaryTab('transcript')">
            <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'transcript' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">실시간 전사&nbsp;&nbsp;</button>
            <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'transcript' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
          </div>
          <div class="relative cursor-pointer summary-subtab-btn group" @click="setSummaryTab('ai-summary')">
            <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'ai-summary' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">AI 요약&nbsp;&nbsp;</button>
            <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'ai-summary' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
          </div>
          <div class="relative cursor-pointer summary-subtab-btn group" @click="setSummaryTab('history')">
            <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'history' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">대화기록&nbsp;&nbsp;</button>
            <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'history' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
          </div>
        </nav>
      </div>

      <div v-show="activeSummaryTab === 'transcript'" class="summary-subcontent summary-transcript-wrap flex-1 min-h-0">
        <VoiceTransferSideTab
          :transcriptions="transcriptions"
          :recording-mode="recordingMode"
          variant="content"
          @askAi="emit('askAi', $event)"
          @addToNote="(text, source) => emit('addToNote', text, source)"
        />
      </div>

      <div v-show="activeSummaryTab === 'ai-summary'" class="summary-subcontent ai-summary-panel flex-1 min-h-0">
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
                  <h3>세션 요약</h3>
                  <p>{{ formatSummaryTime(sessionSummary.createdAt) || '저장된 요약' }}</p>
                </div>
              </div>

              <div class="speaker-summary-status">
                <span class="speaker-summary-dot speaker-summary-dot-blue"></span>
                <span>전체</span>
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

              <div class="speaker-summary-status">
                <span :class="['speaker-summary-dot', speaker.accent.dot]"></span>
                <span>{{ isRecording && !isRecordingPaused ? '실시간' : '요약' }}</span>
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

      <div v-show="activeSummaryTab === 'history'" class="summary-subcontent space-y-10"></div>
    </div>
  </section>
</template>

<style scoped>
.summary-transcript-wrap {
  padding-bottom: 120px;
}

.ai-summary-panel {
  overflow-y: auto;
  padding: 2px 2px 120px;
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
</style>
