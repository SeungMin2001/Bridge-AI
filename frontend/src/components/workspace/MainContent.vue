<!-- 워크스페이스의 중앙 영역으로, 강의 자료 뷰어, 녹음 조작, 노점 요약 내용을 표시합니다. -->
<script setup>
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import { useChat } from '../../composables/useChat'
import { PPTXViewer } from 'pptxviewjs'

const { selectedWordData, clearSelectedWord } = useChat()

const props = defineProps({
  isRecording: Boolean,
  recordingTimeText: String,
  activeFileName: String,
  activeFileId: String,
  materialAttachments: { type: Array, default: () => [] },
  currentPreviewMaterial: { type: Object, default: null },
  summaryNotes: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'startRecording',
  'stopRecording',
  'mainSidebarToggle',
  'rightSidebarToggle',
  'askAi',
  'addToNote',
  'uploadLectureMaterials',
  'closePreviewMaterial',
  'openStoredMaterial',
  'deleteStoredMaterial'
])

const activeTab = ref('note')
const activeSummaryTab = ref('ai-summary')
const noteContent = ref('')
const isNoteFocused = ref(false)
const tabAnim = ref('tab-slide-right')
const fileInputRef = ref(null)
const isNoteDragOver = ref(false)
const pptCanvasRef = ref(null)
const pptViewer = ref(null)
const pptSlideIndex = ref(0)
const pptSlideCount = ref(0)
const pptLoading = ref(false)
const pptError = ref('')
let prevTab = 'note'

const TAB_ORDER = ['note', 'summary-note', 'material', 'summary', 'quiz']

const tabs = computed(() => [
  { key: 'note', label: '메모' },
  { key: 'summary-note', label: '정리 노트' },
  { key: 'material', label: '자료' },
  { key: 'summary', label: '요약' },
  { key: 'quiz', label: '퀴즈' }
])

const noteTitle = computed(() => props.activeFileName || '강의1')

const allowedMaterialTypes = [
  'application/pdf',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation'
]

const handleTabChange = (newTab) => {
  const prevIdx = TAB_ORDER.indexOf(prevTab)
  const nextIdx = TAB_ORDER.indexOf(newTab)
  tabAnim.value = nextIdx > prevIdx ? 'tab-slide-right' : 'tab-slide-left'
  prevTab = newTab
  activeTab.value = newTab
}

watch(activeTab, (newVal) => {
  if (newVal !== prevTab) {
    handleTabChange(newVal)
  }
})

const formatFileSize = (bytes = 0) => {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

const fileIconName = (fileName = '') => {
  const lower = fileName.toLowerCase()
  if (lower.endsWith('.pdf')) return 'picture_as_pdf'
  return 'slideshow'
}

const isPdfAttachment = (file) => /\.pdf$/i.test(file?.name || '')
const isPptAttachment = (file) => /\.(ppt|pptx)$/i.test(file?.name || '')

const triggerMaterialPicker = () => {
  fileInputRef.value?.click()
}

const isLectureMaterialFile = (file) => {
  if (!file) return false
  return allowedMaterialTypes.includes(file.type) || /\.(pdf|ppt|pptx)$/i.test(file.name)
}

const openMaterialInMemo = (file) => {
  emit('uploadLectureMaterials', [file])
  activeTab.value = 'note'
}

const handleMaterialInputChange = (event) => {
  const file = Array.from(event.target.files || []).find(isLectureMaterialFile)
  if (file) {
    openMaterialInMemo(file)
  }
  event.target.value = ''
}

const handleDroppedMaterial = (event) => {
  event.preventDefault()
  isNoteDragOver.value = false
  const file = Array.from(event.dataTransfer?.files || []).find(isLectureMaterialFile)
  if (file) {
    openMaterialInMemo(file)
  }
}

const onNoteBlur = (e) => {
  isNoteFocused.value = false
  noteContent.value = e.target.innerText
}

const handleAskAi = () => {
  if (selectedWordData.value) {
    emit('askAi', selectedWordData.value.word)
    clearSelectedWord()
  }
}

const handleAddToNote = () => {
  if (selectedWordData.value) {
    emit('addToNote', selectedWordData.value.desc, selectedWordData.value.source)
    clearSelectedWord()
  }
}

const handleDeleteStoredMaterial = (fileId) => {
  emit('deleteStoredMaterial', fileId)
}

const destroyPptViewer = () => {
  pptViewer.value?.destroy?.()
  pptViewer.value = null
  pptSlideIndex.value = 0
  pptSlideCount.value = 0
  pptLoading.value = false
  pptError.value = ''
}

const renderCurrentPptSlide = async () => {
  if (!pptViewer.value || !pptCanvasRef.value) return
  await pptViewer.value.render(pptCanvasRef.value, { slideIndex: pptSlideIndex.value })
}

const loadPptPreview = async (file) => {
  if (!file?.sourceFile) {
    pptError.value = 'PPT 원본 파일을 찾을 수 없어 미리보기를 열 수 없습니다.'
    return
  }

  pptLoading.value = true
  pptError.value = ''

  try {
    await nextTick()

    if (!pptCanvasRef.value) {
      throw new Error('PPT 캔버스를 찾지 못했습니다.')
    }

    destroyPptViewer()
    pptLoading.value = true

    const viewer = new PPTXViewer({
      canvas: pptCanvasRef.value,
      slideSizeMode: 'fit',
      backgroundColor: '#ffffff'
    })

    await viewer.loadFile(file.sourceFile)
    pptViewer.value = viewer
    pptSlideCount.value = viewer.getSlideCount()
    pptSlideIndex.value = viewer.getCurrentSlideIndex()
    await renderCurrentPptSlide()
  } catch (error) {
    console.error(error)
    pptError.value = 'PPT를 화면에 불러오지 못했습니다.'
  } finally {
    pptLoading.value = false
  }
}

const goToPreviousPptSlide = async () => {
  if (!pptViewer.value || pptSlideIndex.value <= 0) return
  pptSlideIndex.value -= 1
  await renderCurrentPptSlide()
}

const goToNextPptSlide = async () => {
  if (!pptViewer.value || pptSlideIndex.value >= pptSlideCount.value - 1) return
  pptSlideIndex.value += 1
  await renderCurrentPptSlide()
}

watch(
  () => props.currentPreviewMaterial,
  async (file) => {
    if (!file || !isPptAttachment(file)) {
      destroyPptViewer()
      return
    }

    await loadPptPreview(file)
  }
)

onBeforeUnmount(() => {
  destroyPptViewer()
})
</script>

<template>
  <main class="flex-1 flex flex-col gap-[12px] h-full min-w-0" style="flex: 1 1 0%; min-width: 300px;">
    <transition name="word-card">
      <div v-if="selectedWordData" class="word-info-card card shrink-0">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <div class="word-badge">
              <span class="material-symbols-outlined text-[14px]">dictionary</span>
            </div>
            <span class="text-[15px] font-extrabold text-[#1d1d1f] tracking-tight">{{ selectedWordData.word }}</span>
          </div>
          <button @click="clearSelectedWord" class="p-1.5 rounded-full hover:bg-black/5 transition-colors cursor-pointer">
            <span class="material-symbols-outlined text-[16px] text-[#8e8e93]">close</span>
          </button>
        </div>
        <p class="text-[13px] text-[#3a3a3c] leading-[1.7] font-medium mt-2 mb-0">
          {{ selectedWordData.desc }}
        </p>
        <div class="flex items-center justify-between mt-2.5 pt-2.5 border-t border-black/5">
          <div class="flex items-center gap-1.5">
            <span class="material-symbols-outlined text-[13px] text-[#8e8e93]">link</span>
            <span class="text-[10px] font-bold text-[#8e8e93] uppercase tracking-wider">Source:</span>
            <span class="text-[10px] font-bold text-blue-500">{{ selectedWordData.source }}</span>
          </div>
          <div class="flex gap-1.5">
            <button class="word-card-btn word-card-btn-primary" @click="handleAskAi">
              <span class="material-symbols-outlined text-[13px]">auto_awesome</span>
              AI 질문
            </button>
            <button class="word-card-btn word-card-btn-secondary" @click="handleAddToNote">
              <span class="material-symbols-outlined text-[13px]">note_add</span>
              노트 추가
            </button>
          </div>
        </div>
      </div>
    </transition>

    <div id="tab-contents-container" class="card workspace-shell-card flex-1 flex flex-col relative min-h-0 min-w-0 overflow-hidden">
      <header class="workspace-embedded-header h-[56px] flex items-center px-6 shrink-0">
        <div class="flex items-center gap-1.5 shrink-0">
          <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0" title="사이드바 토글" @click="emit('mainSidebarToggle')">
            <span class="material-symbols-outlined text-[20px]">side_navigation</span>
          </button>

          <button v-if="!isRecording" class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] shrink-0" id="start" @click="emit('startRecording')">
            <span class="material-symbols-outlined text-[20px]">mic</span>
          </button>
          <div
            v-else
            id="recording-timer"
            class="flex items-center gap-2 bg-[#FFF4F6] hover:bg-[#FFECEE] px-3 py-1.5 rounded-full cursor-pointer transition-colors border border-white/80 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_8px_18px_rgba(148,163,184,0.12)] shrink-0"
            @click="emit('stopRecording')"
          >
            <div class="recording-wave-container w-6 h-6 shrink-0">
              <div class="recording-wave-ring"></div>
              <div class="recording-wave-ring"></div>
              <div class="recording-wave-ring"></div>
              <span class="live-dot" style="position: relative; z-index: 1;"></span>
            </div>
            <span class="text-[13px] font-bold text-[#1d1d1f] tabular-nums" id="recording-time">{{ recordingTimeText }}</span>
          </div>
        </div>

        <div class="ml-auto flex items-center gap-1.5 shrink-0 pl-2">
          <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
            <span class="material-symbols-outlined text-[20px]">play_circle</span>
          </button>

          <button
            class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]"
            title="강의 자료 열기"
            @click="triggerMaterialPicker"
          >
            <span class="material-symbols-outlined text-[20px]">folder_open</span>
          </button>

          <input
            ref="fileInputRef"
            type="file"
            accept=".pdf,.ppt,.pptx,application/pdf,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation"
            class="hidden"
            @change="handleMaterialInputChange"
          />

          <button class="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" title="우측 사이드바 토글" @click="emit('rightSidebarToggle')">
            <span class="material-symbols-outlined text-[20px] scale-x-[-1]">side_navigation</span>
          </button>
        </div>
      </header>

      <div class="workspace-embedded-divider shrink-0"></div>

      <div class="flex-1 flex flex-col relative min-h-0 min-w-0">
        <section
          v-if="activeTab === 'note'"
          :key="'tab-note'"
          :class="['tab-content flex-1 flex flex-col relative overflow-hidden note-canvas p-10 pt-4', tabAnim]"
          @dragover.prevent="isNoteDragOver = true"
          @dragenter.prevent="isNoteDragOver = true"
          @dragleave.prevent="isNoteDragOver = false"
          @drop="handleDroppedMaterial"
        >
          <div class="max-w-4xl mx-auto w-full h-full">
            <h1 v-if="!currentPreviewMaterial" class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">{{ noteTitle }}</h1>

            <div v-if="activeFileId === 'lecture-1' && currentPreviewMaterial" class="mb-6">
              <div class="lecture-preview-shell">
                <div class="lecture-preview-header">
                  <div class="flex items-center justify-end gap-3 flex-wrap">
                    <button type="button" class="preview-close-btn" @click="emit('closePreviewMaterial')">
                      닫기
                    </button>
                  </div>
                </div>

                <div v-if="isPdfAttachment(currentPreviewMaterial)" class="lecture-preview-frame-wrap">
                  <iframe
                    :src="currentPreviewMaterial.url"
                    class="lecture-preview-frame"
                    title="PDF Preview"
                  ></iframe>
                </div>

                <div v-else-if="isPptAttachment(currentPreviewMaterial)" class="lecture-preview-fallback">
                  <div class="ppt-preview-toolbar">
                    <button type="button" class="ppt-nav-btn" :disabled="pptSlideIndex <= 0 || pptLoading" @click="goToPreviousPptSlide">
                      이전
                    </button>
                    <span class="ppt-slide-status">
                      {{ pptSlideCount ? `${pptSlideIndex + 1} / ${pptSlideCount}` : '슬라이드 준비 중' }}
                    </span>
                    <button type="button" class="ppt-nav-btn" :disabled="pptSlideIndex >= pptSlideCount - 1 || pptLoading" @click="goToNextPptSlide">
                      다음
                    </button>
                  </div>
                  <div v-if="pptLoading" class="ppt-placeholder-copy">PPT 슬라이드를 불러오는 중입니다.</div>
                  <div v-else-if="pptError" class="ppt-placeholder-copy">{{ pptError }}</div>
                  <canvas v-show="!pptLoading && !pptError" ref="pptCanvasRef" class="ppt-preview-canvas"></canvas>
                </div>
              </div>
            </div>

            <div
              class="text-[16px] leading-relaxed min-h-[200px] focus:outline-none"
              :class="{ 'note-drop-target': isNoteDragOver }"
              id="note-body"
              contenteditable="true"
              v-show="!currentPreviewMaterial"
              :style="{ color: isNoteFocused || noteContent ? '#1d1d1f' : '#aeaeb2' }"
              @focus="isNoteFocused = true"
              @blur="onNoteBlur"
            >
              {{ (!noteContent && !isNoteFocused) ? '여기에 타이핑을 시작하거나 파일을 업로드하세요.' : noteContent }}
            </div>
          </div>
        </section>

        <section v-else-if="activeTab === 'summary-note'" :key="'tab-summary-note'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">정리 노트</h1>
            <div class="flex flex-col gap-4">
              <div v-if="summaryNotes.length === 0" class="text-[16px] text-[#aeaeb2] leading-relaxed italic">아직 추가된 내용이 없습니다. 전사 내용에서 '노트에 추가'를 눌러보세요.</div>
              <div v-else v-for="note in summaryNotes" :key="note.id" class="workspace-subpanel p-5 rounded-[24px] flex flex-col gap-2 transcription-item-enter">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-[18px] text-blue-500">auto_stories</span>
                    <span class="text-[13px] font-bold text-[#1d1d1f]">추가된 내용</span>
                  </div>
                  <span class="text-[11px] font-medium text-[#aeaeb2]">{{ note.time }}</span>
                </div>
                <p class="text-[15px] leading-[1.6] text-[#3a3a3c] font-medium">{{ note.text }}</p>
                <div class="flex items-center gap-1.5 mt-1 border-t border-black/5 pt-3">
                  <span class="material-symbols-outlined text-[14px] text-[#8e8e93]">link</span>
                  <span class="text-[11px] font-bold text-[#8e8e93] uppercase tracking-wider">Source:</span>
                  <span class="text-[11px] font-bold text-blue-500 cursor-pointer hover:underline decoration-blue-500/50 underline-offset-2">{{ note.source || 'AI 분석 결과' }}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section v-else-if="activeTab === 'material'" :key="'tab-material'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">자료</h1>
            <div v-if="materialAttachments.length" class="flex flex-col gap-3">
              <button
                v-for="file in materialAttachments"
                :key="file.id"
                type="button"
                class="pdf-file-row text-left"
                @click="emit('openStoredMaterial', file.id); handleTabChange('note')"
              >
                <div class="flex items-center gap-3 min-w-0">
                  <div class="pdf-file-badge">
                    <span class="material-symbols-outlined text-[18px]">{{ fileIconName(file.name) }}</span>
                  </div>
                  <div class="min-w-0">
                    <p class="text-[14px] font-bold text-[#1d1d1f] truncate">{{ file.name }}</p>
                    <p class="text-[12px] text-[#8e8e93] mt-0.5">{{ formatFileSize(file.size) }}</p>
                  </div>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                  <span class="pdf-file-action shrink-0">메모 탭에서 열기</span>
                  <button
                    type="button"
                    class="material-delete-btn"
                    title="자료 삭제"
                    @click.stop="handleDeleteStoredMaterial(file.id)"
                  >
                    삭제
                  </button>
                </div>
              </button>
            </div>
            <div v-else class="lecture-material-empty">
              아직 저장된 자료가 없습니다. 메모 탭에서 파일을 올리면 여기에 저장됩니다.
            </div>
          </div>
        </section>

        <section v-else-if="activeTab === 'summary'" :key="'tab-summary'" :class="['tab-content flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full">
            <div class="flex items-center justify-between border-b border-[#e5e5ea] mb-5 pb-0">
              <nav class="flex gap-8">
                <div class="relative cursor-pointer summary-subtab-btn group" @click="activeSummaryTab = 'ai-summary'">
                  <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'ai-summary' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">AI 요약&nbsp;&nbsp;</button>
                  <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'ai-summary' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
                </div>
                <div class="relative cursor-pointer summary-subtab-btn group" @click="activeSummaryTab = 'history'">
                  <button :class="['text-[15px] py-3 pointer-events-none transition-colors', activeSummaryTab === 'history' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]']">대화기록&nbsp;&nbsp;</button>
                  <div :class="['summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors', activeSummaryTab === 'history' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]']"></div>
                </div>
              </nav>
            </div>
            <div v-show="activeSummaryTab === 'ai-summary'" class="summary-subcontent space-y-10"></div>
            <div v-show="activeSummaryTab === 'history'" class="summary-subcontent space-y-10"></div>
          </div>
        </section>

        <section v-else-if="activeTab === 'quiz'" :key="'tab-quiz'" :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
          <div class="max-w-4xl mx-auto w-full h-full">
            <h1 class="text-[32px] font-heavy-heading text-[#d1d1d6] mb-5">퀴즈</h1>
            <div class="text-[16px] text-[#aeaeb2] leading-relaxed">생성된 퀴즈와 테스트가 여기에 표시됩니다.</div>
          </div>
        </section>

        <div class="workspace-tab-float-wrap">
          <div class="workspace-tab-float">
            <button
              v-for="tab in tabs"
              :key="tab.key"
              class="workspace-tab-chip"
              :class="{ 'is-active': activeTab === tab.key }"
              @click="handleTabChange(tab.key)"
            >
              {{ tab.label }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.word-info-card {
  padding: 14px 18px;
  border-left: 1px solid var(--workspace-sidebar-card-border);
  background: var(--workspace-sidebar-card-bg);
  border-color: var(--workspace-sidebar-card-border);
  box-shadow: var(--workspace-sidebar-card-shadow);
  backdrop-filter: blur(22px) saturate(135%);
  -webkit-backdrop-filter: blur(22px) saturate(135%);
}

.word-info-card::before {
  background: var(--workspace-sidebar-card-overlay);
}

.word-info-card::after {
  border-color: var(--workspace-sidebar-card-inner-border);
}

.workspace-shell-card {
  background: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow:
    0 26px 52px rgba(148, 163, 184, 0.08),
    0 10px 24px rgba(0, 0, 0, 0.02);
}

.workspace-shell-card::before {
  display: none;
}

.workspace-shell-card::after {
  display: none;
}

.workspace-embedded-header {
  background: #ffffff;
}

.workspace-embedded-divider {
  height: 1px;
  margin: 0 24px;
  background: rgba(0, 0, 0, 0.06);
}

.workspace-tab-float {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: #eee4d8;
  border: 1px solid rgba(228, 217, 203, 0.95);
  box-shadow:
    0 18px 36px rgba(208, 194, 177, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.34);
}

.workspace-tab-float::before {
  display: none;
}

.workspace-tab-float::after {
  display: none;
}

.workspace-tab-float-wrap {
  position: absolute;
  left: 50%;
  bottom: 28px;
  transform: translateX(-50%);
  z-index: 15;
}

.workspace-tab-chip {
  position: relative;
  z-index: 1;
  min-width: 0;
  padding: 12px 18px;
  border-radius: 999px;
  border: 1px solid transparent;
  background: transparent;
  color: rgba(86, 86, 92, 0.68);
  font-size: 13px;
  font-weight: 800;
  letter-spacing: -0.02em;
  transition: color 0.24s ease, transform 0.24s ease, box-shadow 0.24s ease, background 0.24s ease;
  cursor: pointer;
}

.workspace-tab-chip:hover {
  color: rgba(29, 29, 31, 0.84);
  background: rgba(255, 255, 255, 0.22);
}

.workspace-tab-chip.is-active {
  background:
    radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.96), transparent 48%),
    linear-gradient(180deg, rgba(251, 248, 243, 0.95), rgba(239, 230, 217, 0.88));
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.94);
  box-shadow:
    0 18px 28px rgba(211, 198, 180, 0.28),
    0 6px 18px rgba(255, 255, 255, 0.38),
    inset 0 1px 0 rgba(255, 255, 255, 0.98),
    inset 0 -2px 6px rgba(213, 197, 176, 0.3);
  backdrop-filter: blur(16px) saturate(150%);
  -webkit-backdrop-filter: blur(16px) saturate(150%);
}

.workspace-tab-chip.is-active::before {
  content: '';
  position: absolute;
  inset: 2px 6px auto;
  height: 52%;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.18));
  opacity: 0.95;
  pointer-events: none;
}

.workspace-tab-chip.is-active::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.38),
    inset 0 -10px 16px rgba(222, 206, 188, 0.16);
  pointer-events: none;
}

@media (max-width: 900px) {
  .workspace-tab-float-wrap {
    left: 24px;
    right: 24px;
    transform: none;
  }

  .workspace-tab-float {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    width: 100%;
    gap: 8px;
    padding: 8px;
  }

  .workspace-tab-chip {
    min-width: 0;
    padding: 12px 10px;
    font-size: 13px;
  }
}

.word-badge {
  width: 28px;
  height: 28px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.7));
  border: 1px solid rgba(255, 255, 255, 0.82);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  box-shadow: 0 10px 20px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.word-card-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  border-radius: 8px;
  border: none;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.word-card-btn-primary {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.72));
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.84);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.96);
}

.word-card-btn-primary:hover {
  background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(255, 255, 255, 0.78));
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(148, 163, 184, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.98);
}

.word-card-btn-secondary {
  background: rgba(255, 255, 255, 0.48);
  color: #1d1d1f;
  border: 1px solid rgba(255, 255, 255, 0.72);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.word-card-btn-secondary:hover {
  background: rgba(255, 255, 255, 0.68);
  transform: translateY(-1px);
}

.note-drop-target {
  border-radius: 24px;
  background: rgba(239, 246, 255, 0.5);
  outline: 1.5px dashed rgba(59, 130, 246, 0.42);
  outline-offset: 16px;
}

.lecture-material-empty {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255,255,255,0.72);
  color: #8e8e93;
  font-size: 13px;
  line-height: 1.7;
}

.pdf-file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.72);
  background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(248,250,252,0.7));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.96), 0 12px 24px rgba(148, 163, 184, 0.08);
  transition: all 0.2s ease;
}

.pdf-file-row:hover {
  transform: translateY(-1px);
  border-color: rgba(59, 130, 246, 0.24);
}

.pdf-file-badge {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: rgba(239, 246, 255, 0.95);
}

.pdf-file-action {
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 700;
  color: #4b5563;
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(229,231,235,0.9);
}

.material-delete-btn {
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid rgba(248, 113, 113, 0.24);
  background: rgba(254, 242, 242, 0.96);
  color: #b91c1c;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.material-delete-btn:hover {
  background: rgba(254, 226, 226, 0.98);
  border-color: rgba(239, 68, 68, 0.34);
}

.lecture-preview-shell {
  padding: 18px;
  border-radius: 22px;
  border: 1px solid rgba(255,255,255,0.84);
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(248,250,252,0.8));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.98), 0 18px 36px rgba(148, 163, 184, 0.1);
}

.lecture-preview-header {
  margin-bottom: 14px;
}

.preview-close-btn {
  padding: 8px 14px;
  border-radius: 12px;
  border: 1px solid rgba(229,231,235,0.9);
  background: rgba(255,255,255,0.9);
  font-size: 12px;
  font-weight: 700;
  color: #4b5563;
}

.lecture-preview-frame-wrap {
  width: 100%;
  height: min(70vh, 820px);
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(226,232,240,0.9);
  background: #fff;
}

.lecture-preview-frame {
  width: 100%;
  height: 100%;
  border: none;
  background: #fff;
}

.lecture-preview-fallback {
  min-height: 260px;
  border-radius: 18px;
  border: 1px dashed rgba(59, 130, 246, 0.28);
  background: rgba(239,246,255,0.5);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  gap: 10px;
  padding: 24px;
  text-align: center;
}

.ppt-preview-toolbar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 10px;
}

.ppt-nav-btn {
  padding: 8px 14px;
  border-radius: 12px;
  border: 1px solid rgba(229,231,235,0.9);
  background: rgba(255,255,255,0.92);
  font-size: 12px;
  font-weight: 700;
  color: #374151;
}

.ppt-nav-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ppt-slide-status {
  font-size: 12px;
  font-weight: 800;
  color: #1d1d1f;
}

.ppt-placeholder-copy {
  color: #6b7280;
  font-size: 13px;
  line-height: 1.6;
  margin: 8px 0;
}

.ppt-preview-canvas {
  width: 100%;
  max-width: 100%;
  min-height: 480px;
  border-radius: 16px;
  background: #ffffff;
  box-shadow: inset 0 0 0 1px rgba(226,232,240,0.9);
}

.word-card-enter-active {
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.word-card-leave-active {
  transition: all 0.2s ease;
}

.word-card-enter-from {
  opacity: 0;
  transform: translateY(-8px) scaleY(0.9);
  max-height: 0;
}

.word-card-enter-to {
  opacity: 1;
  transform: translateY(0) scaleY(1);
  max-height: 200px;
}

.word-card-leave-from {
  opacity: 1;
  transform: translateY(0) scaleY(1);
  max-height: 200px;
}

.word-card-leave-to {
  opacity: 0;
  transform: translateY(-8px) scaleY(0.9);
  max-height: 0;
}
</style>
