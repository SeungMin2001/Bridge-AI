<!-- 워크스페이스 메모 탭에서 PDF 또는 PPT 강의 자료 미리보기를 표시하는 패널입니다. -->
<script setup>
import { computed, ref, watch, nextTick, onBeforeUnmount } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import { PPTXViewer } from 'pptxviewjs'
import PptPreviewToolbar from './PptPreviewToolbar.vue'
import LoadingHourglass from '../../ui/LoadingHourglass.vue'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

const props = defineProps({
  material: { type: Object, default: null },
  evidenceRequest: { type: Object, default: null },
  pdfSearchQuery: { type: String, default: '' },
  pdfSearchCommand: { type: Object, default: null }
})

const emit = defineEmits(['pdf-search-results'])

const pptCanvasRef = ref(null)
const pptViewer = ref(null)
const pptSlideIndex = ref(0)
const pptSlideCount = ref(0)
const pptLoading = ref(false)
const pptError = ref('')
const pdfContainerRef = ref(null)
const pdfLoading = ref(false)
const pdfError = ref('')
const pdfPageCount = ref(0)
const pdfZoom = ref(1)

let activePdfTask = null
let activePdfDocument = null
let pdfRenderToken = 0
let activePdfTextLayers = []
let activePdfPageShells = []
let activePdfSearchMatches = []
let activePdfSearchActiveIndex = 0

const PDF_ZOOM_MIN = 0.7
const PDF_ZOOM_MAX = 1.8
const PDF_ZOOM_STEP = 0.1

const isPdfAttachment = (file) => /\.pdf$/i.test(file?.name || '')
const isPptAttachment = (file) => /\.(ppt|pptx)$/i.test(file?.name || '')
const normalizedPdfSearchQuery = computed(() => String(props.pdfSearchQuery || '').trim().toLowerCase())
const normalizePdfSearchText = (value = '') => String(value || '').toLowerCase().replace(/\s+/g, '')

const emitPdfSearchResults = () => {
  emit('pdf-search-results', {
    total: activePdfSearchMatches.length,
    activeIndex: activePdfSearchMatches.length ? activePdfSearchActiveIndex : 0
  })
}

const setActivePdfSearchMatch = async (index = activePdfSearchActiveIndex, behavior = 'smooth') => {
  if (!activePdfSearchMatches.length) {
    emitPdfSearchResults()
    return
  }

  activePdfSearchActiveIndex = (index + activePdfSearchMatches.length) % activePdfSearchMatches.length
  activePdfSearchMatches.forEach((match) => {
    match.elements.forEach((element) => {
      element.classList.remove('is-active-pdf-search-match')
    })
  })
  activePdfSearchMatches[activePdfSearchActiveIndex].elements.forEach((element) => {
    element.classList.add('is-active-pdf-search-match')
  })

  await nextTick()
  activePdfSearchMatches[activePdfSearchActiveIndex]?.elements?.[0]?.scrollIntoView?.({ behavior, block: 'center', inline: 'nearest' })
  emitPdfSearchResults()
}

const clearPdfSearchMatches = () => {
  activePdfSearchMatches.forEach((match) => {
    match.elements.forEach((element) => {
      element.classList.remove('pdf-search-match', 'is-active-pdf-search-match')
    })
  })
  activePdfSearchMatches = []
  activePdfSearchActiveIndex = 0
  emitPdfSearchResults()
}

const refreshPdfSearchMatches = async () => {
  activePdfSearchMatches.forEach((match) => {
    match.elements.forEach((element) => {
      element.classList.remove('pdf-search-match', 'is-active-pdf-search-match')
    })
  })
  activePdfSearchMatches = []
  activePdfSearchActiveIndex = 0

  const query = normalizePdfSearchText(normalizedPdfSearchQuery.value)
  if (!query || !pdfContainerRef.value) {
    emitPdfSearchResults()
    return
  }

  await nextTick()
  const textItems = Array.from(pdfContainerRef.value.querySelectorAll('.pdf-text-layer span'))
    .filter((element) => normalizePdfSearchText(element.textContent).length > 0)
  let normalizedText = ''
  const charElementMap = []

  textItems.forEach((element) => {
    Array.from(normalizePdfSearchText(element.textContent)).forEach((char) => {
      normalizedText += char
      charElementMap.push(element)
    })
  })

  let cursor = 0
  while (cursor < normalizedText.length) {
    const matchStart = normalizedText.indexOf(query, cursor)
    if (matchStart === -1) break

    const matchEnd = matchStart + query.length
    const elements = Array.from(new Set(charElementMap.slice(matchStart, matchEnd))).filter(Boolean)
    if (elements.length) {
      activePdfSearchMatches.push({ elements })
    }
    cursor = matchStart + Math.max(query.length, 1)
  }

  activePdfSearchMatches.forEach((match) => {
    match.elements.forEach((element) => {
      element.classList.add('pdf-search-match')
    })
  })

  if (activePdfSearchMatches.length) {
    await setActivePdfSearchMatch(0, 'auto')
    return
  }

  emitPdfSearchResults()
}

const movePdfSearchMatch = (direction = 1) => {
  if (!activePdfSearchMatches.length) return
  setActivePdfSearchMatch(activePdfSearchActiveIndex + direction)
}

// PDF 미리보기 배율을 조절합니다. 다시 렌더링하지 않고 CSS 크기만 바꿉니다.
const updatePdfZoom = (nextZoom) => {
  pdfZoom.value = Math.min(PDF_ZOOM_MAX, Math.max(PDF_ZOOM_MIN, Number(nextZoom.toFixed(2))))
}

const zoomOutPdf = () => {
  updatePdfZoom(pdfZoom.value - PDF_ZOOM_STEP)
}

const zoomInPdf = () => {
  updatePdfZoom(pdfZoom.value + PDF_ZOOM_STEP)
}

const resetPdfZoom = () => {
  updatePdfZoom(1)
}

// 업로드한 PDF의 텍스트를 페이지별 JSON 형태로 추출합니다.
const createPdfJsonSkeleton = (file, pageCount) => ({
  fileName: file?.name || '',
  fileType: file?.type || '',
  fileSize: file?.size || 0,
  pageCount,
  extractedAt: new Date().toISOString(),
  pages: []
})

// PDF/PPT 리소스 정리
const destroyPptViewer = () => {
  pptViewer.value?.destroy?.()
  pptViewer.value = null
  pptSlideIndex.value = 0
  pptSlideCount.value = 0
  pptLoading.value = false
  pptError.value = ''
}

const clearPdfPreview = () => {
  clearPdfSearchMatches()
  pdfPageCount.value = 0
  pdfLoading.value = false
  pdfError.value = ''
  activePdfPageShells = []
  if (pdfContainerRef.value) {
    pdfContainerRef.value.innerHTML = ''
  }
}

const clearEvidenceTarget = () => {
  activePdfPageShells.forEach((pageShell) => {
    pageShell?.classList?.remove('is-evidence-target')
    pageShell?.querySelector?.('.pdf-evidence-banner')?.remove()
  })
}

const addEvidenceBanner = (pageShell, request) => {
  pageShell.querySelector('.pdf-evidence-banner')?.remove()

  const banner = document.createElement('div')
  banner.className = 'pdf-evidence-banner'

  const label = document.createElement('span')
  label.textContent = 'AI가 참조한 PDF 페이지'

  const page = document.createElement('strong')
  page.textContent = `p.${Number(request?.page || 1)}`

  banner.appendChild(label)
  banner.appendChild(page)
  pageShell.prepend(banner)
}

const scrollToEvidencePage = async (request, attempt = 0) => {
  if (!request || !pdfContainerRef.value) return

  const targetPage = Number(request.page || 1)
  const pageShell = activePdfPageShells[targetPage]
  if (!pageShell) {
    if (attempt < 8) {
      window.setTimeout(() => scrollToEvidencePage(request, attempt + 1), 120)
    }
    return
  }

  await nextTick()
  clearEvidenceTarget()
  pageShell.classList.add('is-evidence-target')
  addEvidenceBanner(pageShell, request)
  pageShell.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

const destroyPdfPreview = async ({ incrementToken = true } = {}) => {
  if (incrementToken) {
    pdfRenderToken += 1
  }

  activePdfTextLayers.forEach((textLayer) => {
    try {
      textLayer?.cancel?.()
    } catch (error) {
      console.warn('PDF text layer cleanup failed.', error)
    }
  })
  activePdfTextLayers = []
  clearPdfPreview()

  try {
    await activePdfTask?.destroy?.()
  } catch (error) {
    console.warn('PDF loading task cleanup failed.', error)
  }

  try {
    await activePdfDocument?.destroy?.()
  } catch (error) {
    console.warn('PDF document cleanup failed.', error)
  }

  activePdfTask = null
  activePdfDocument = null
}

// PDF 로딩
const renderPdfPreview = async (file) => {
  if (!file?.url) {
    pdfError.value = 'PDF 파일을 찾을 수 없어 미리보기를 열 수 없습니다.'
    return
  }

  pdfLoading.value = true
  pdfError.value = ''
  pdfPageCount.value = 0
  const renderToken = ++pdfRenderToken

  try {
    await nextTick()

    if (!pdfContainerRef.value) {
      throw new Error('PDF 컨테이너를 찾지 못했습니다.')
    }

    await destroyPdfPreview({ incrementToken: false })
    pdfLoading.value = true
    pdfError.value = ''

    activePdfTask = pdfjsLib.getDocument(file.url)
    const pdfDocument = await activePdfTask.promise

    if (renderToken !== pdfRenderToken) return

    activePdfDocument = pdfDocument
    pdfPageCount.value = pdfDocument.numPages

    const availableWidth = Math.max((pdfContainerRef.value.clientWidth || 960) - 12, 320)
    const extractedPdfJson = createPdfJsonSkeleton(file, pdfDocument.numPages)

    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
      if (renderToken !== pdfRenderToken) return

      // PDF 페이지별 canvas 렌더링
      const page = await pdfDocument.getPage(pageNumber)
      const textContent = await page.getTextContent()
      const initialViewport = page.getViewport({ scale: 1 })
      const scale = Math.max(0.75, Math.min(2.15, availableWidth / initialViewport.width))
      const viewport = page.getViewport({ scale })
      const canvas = document.createElement('canvas')
      const context = canvas.getContext('2d')

      if (!context) {
        throw new Error('PDF 캔버스 컨텍스트를 가져오지 못했습니다.')
      }

      canvas.width = viewport.width
      canvas.height = viewport.height
      canvas.className = 'pdf-preview-canvas'

      const pageShell = document.createElement('div')
      pageShell.className = 'pdf-page-shell'
      pageShell.dataset.pageNumber = String(pageNumber)
      activePdfPageShells[pageNumber] = pageShell

      const pageMeta = document.createElement('div')
      pageMeta.className = 'pdf-page-meta'
      pageMeta.textContent = `${pageNumber} / ${pdfDocument.numPages}`

      const pageStage = document.createElement('div')
      pageStage.className = 'pdf-page-stage'
      pageStage.style.aspectRatio = `${viewport.width} / ${viewport.height}`
      pageStage.style.setProperty('--total-scale-factor', '1')

      const textLayerDiv = document.createElement('div')
      textLayerDiv.className = 'textLayer pdf-text-layer'
      textLayerDiv.style.width = '100%'
      textLayerDiv.style.height = '100%'
      textLayerDiv.style.setProperty('--total-scale-factor', '1')

      const extractedItems = textContent.items
        .filter((item) => item.str?.trim())
        .map((item) => ({
          text: item.str,
          x: item.transform?.[4] || 0,
          y: item.transform?.[5] || 0,
          width: item.width || 0,
          height: item.height || 0
        }))

      extractedPdfJson.pages.push({
        page: pageNumber,
        text: extractedItems.map((item) => item.text).join(' '),
        items: extractedItems
      })

      // PDF text layer 렌더링
      const textLayer = new pdfjsLib.TextLayer({
        textContentSource: textContent,
        container: textLayerDiv,
        viewport
      })
      activePdfTextLayers.push(textLayer)

      pageStage.appendChild(canvas)
      pageStage.appendChild(textLayerDiv)
      pageShell.appendChild(pageMeta)
      pageShell.appendChild(pageStage)
      pdfContainerRef.value.appendChild(pageShell)

      await Promise.all([
        page.render({
          canvasContext: context,
          viewport
        }).promise,
        textLayer.render()
      ])

      const endOfContent = document.createElement('div')
      endOfContent.className = 'endOfContent'
      textLayerDiv.appendChild(endOfContent)
    }

    if (renderToken === pdfRenderToken) {
      console.log('PDF JSON 추출 결과:', extractedPdfJson)
      await refreshPdfSearchMatches()
      if (props.evidenceRequest) {
        await scrollToEvidencePage(props.evidenceRequest)
      }
    }
  } catch (error) {
    console.error(error)
    pdfError.value = 'PDF를 화면에 불러오지 못했습니다.'
  } finally {
    if (renderToken === pdfRenderToken) {
      pdfLoading.value = false
    }
  }
}

// PPT 슬라이드 이동
const renderCurrentPptSlide = async () => {
  if (!pptViewer.value || !pptCanvasRef.value) return
  await pptViewer.value.render(pptCanvasRef.value, { slideIndex: pptSlideIndex.value })
}

// PPT 로딩
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
  () => props.material,
  async (file) => {
    if (!file) {
      destroyPptViewer()
      await destroyPdfPreview()
      return
    }

    if (isPdfAttachment(file)) {
      destroyPptViewer()
      resetPdfZoom()
      await renderPdfPreview(file)
      return
    }

    await destroyPdfPreview()

    if (!isPptAttachment(file)) {
      destroyPptViewer()
      return
    }

    await loadPptPreview(file)
  },
  { immediate: true }
)

watch(
  () => props.evidenceRequest,
  async (request) => {
    if (!request) return
    await scrollToEvidencePage(request)
  },
  { deep: true }
)

watch(
  normalizedPdfSearchQuery,
  () => refreshPdfSearchMatches()
)

watch(
  () => props.pdfSearchCommand,
  (command) => {
    if (!command?.action) return
    if (command.action === 'next') movePdfSearchMatch(1)
    if (command.action === 'prev') movePdfSearchMatch(-1)
    if (command.action === 'clear') clearPdfSearchMatches()
  },
  { deep: true }
)

onBeforeUnmount(() => {
  destroyPptViewer()
  destroyPdfPreview()
})
</script>

<template>
  <!-- PDF/PPT 상태 UI -->
  <div class="lecture-preview-shell">
    <div v-if="isPdfAttachment(material)" class="lecture-preview-frame-wrap">
      <div class="lecture-preview-surface">
        <div class="pdf-preview-stage">
          <div
            ref="pdfContainerRef"
            class="pdf-preview-scroll custom-scrollbar"
            :class="{ 'is-hidden': pdfLoading || pdfError }"
            :style="{ '--pdf-zoom': pdfZoom }"
          ></div>
          <div v-if="pdfLoading" class="pdf-preview-placeholder pdf-preview-loading-state">
            <LoadingHourglass
              class="pdf-upload-animation"
              src="/animations/upload%20file.json"
              width="190px"
              height="190px"
              fallback-icon="upload_file"
            />
            <span>PDF를 불러오는 중입니다.</span>
          </div>
          <div v-else-if="pdfError" class="pdf-preview-placeholder">{{ pdfError }}</div>
          <div v-else class="pdf-zoom-controls" aria-label="PDF 확대 축소">
            <button type="button" class="pdf-zoom-btn" :disabled="pdfZoom <= PDF_ZOOM_MIN" title="축소" @click="zoomOutPdf">
              −
            </button>
            <button type="button" class="pdf-zoom-btn" :disabled="pdfZoom >= PDF_ZOOM_MAX" title="확대" @click="zoomInPdf">
              +
            </button>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="isPptAttachment(material)" class="lecture-preview-fallback">
      <PptPreviewToolbar
        :slide-index="pptSlideIndex"
        :slide-count="pptSlideCount"
        :loading="pptLoading"
        @previous="goToPreviousPptSlide"
        @next="goToNextPptSlide"
      />
      <div v-if="pptLoading" class="ppt-placeholder-copy">PPT 슬라이드를 불러오는 중입니다.</div>
      <div v-else-if="pptError" class="ppt-placeholder-copy">{{ pptError }}</div>
      <canvas v-show="!pptLoading && !pptError" ref="pptCanvasRef" class="ppt-preview-canvas"></canvas>
    </div>
  </div>
</template>

<!-- 미리보기 스타일 -->
<style scoped>
.lecture-preview-shell {
  height: 100%;
}

.lecture-preview-frame-wrap {
  width: 100%;
  height: 100%;
}

.lecture-preview-surface {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 0;
  border: none;
  background: transparent;
  box-shadow: none;
}

.pdf-preview-stage {
  flex: 1;
  min-height: 0;
  position: relative;
}

.pdf-preview-scroll {
  height: 100%;
  overflow-y: auto;
  padding: 0 0 40px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.pdf-zoom-controls {
  position: absolute;
  right: 18px;
  bottom: 28px;
  z-index: 5;
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  pointer-events: none;
  transition: bottom 0.22s ease;
}

:global(.workspace-unified-card:has(.is-unified-audio-player) .pdf-zoom-controls) {
  bottom: 96px;
}

.pdf-zoom-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border: none;
  border-radius: 999px;
  color: #ffffff;
  background: rgba(17, 24, 39, 0.86);
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.22);
  font-size: 24px;
  line-height: 1;
  pointer-events: auto;
  backdrop-filter: blur(14px) saturate(130%);
  -webkit-backdrop-filter: blur(14px) saturate(130%);
  transition: background-color 0.2s ease, transform 0.2s ease, opacity 0.2s ease;
}

.pdf-zoom-btn:not(:disabled):hover {
  background: rgba(17, 24, 39, 0.96);
  transform: translateY(-1px);
}

.pdf-zoom-btn:disabled {
  cursor: not-allowed;
  opacity: 0.38;
}

.pdf-preview-scroll.is-hidden {
  visibility: hidden;
}

.pdf-preview-placeholder {
  position: absolute;
  inset: 0;
  border-radius: 28px 28px 0 0;
  border: 1px dashed rgba(148, 163, 184, 0.28);
  background: rgba(248, 250, 252, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #64748b;
  font-size: 14px;
  font-weight: 600;
}

.pdf-preview-loading-state {
  flex-direction: column;
  gap: 14px;
}

.pdf-upload-animation {
  opacity: 0.92;
  user-select: none;
  pointer-events: none;
}

.lecture-preview-fallback {
  min-height: 100%;
  border-radius: 0;
  border: none;
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  gap: 10px;
  padding: 0 0 32px;
  text-align: center;
  box-shadow: none;
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
  box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.9);
}

:deep(.pdf-page-shell) {
  width: calc(100% * var(--pdf-zoom, 1));
  margin: 0 auto;
  padding: 0;
  background: transparent;
  border: none;
  box-shadow: none;
  transition: width 0.18s ease;
}

:deep(.pdf-page-shell.is-evidence-target .pdf-preview-canvas) {
  border-color: rgba(245, 158, 11, 0.78);
  box-shadow: none;
}

:deep(.pdf-evidence-banner) {
  width: fit-content;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 10px 6px;
  padding: 7px 11px;
  border-radius: 999px;
  color: #854d0e;
  background: rgba(254, 249, 195, 0.95);
  border: 1px solid rgba(250, 204, 21, 0.55);
  font-size: 12px;
  font-weight: 900;
}

:deep(.pdf-evidence-banner strong) {
  color: #713f12;
  font-size: 12px;
}

:deep(.pdf-page-meta) {
  margin-bottom: 10px;
  padding-left: 6px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  color: #94a3b8;
  text-transform: uppercase;
}

:deep(.pdf-preview-canvas) {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 30px;
  background: #ffffff;
  border: 1px solid rgba(226, 232, 240, 0.7);
  box-shadow: none;
}

:deep(.pdf-page-stage) {
  position: relative;
  margin: 0 auto;
  width: 100%;
}

:deep(.pdf-text-layer) {
  color-scheme: only light;
  position: absolute;
  text-align: initial;
  inset: 0;
  overflow: clip;
  opacity: 1;
  line-height: 1;
  text-size-adjust: none;
  forced-color-adjust: none;
  transform-origin: 0 0;
  caret-color: CanvasText;
  z-index: 2;
}

:deep(.pdf-text-layer :is(span, br)) {
  color: transparent;
  position: absolute;
  white-space: pre;
  cursor: text;
  transform-origin: 0 0;
}

:deep(.pdf-text-layer) {
  --min-font-size: 1;
  --text-scale-factor: calc(var(--total-scale-factor) * var(--min-font-size));
}

:deep(.pdf-text-layer > :not(.markedContent)),
:deep(.pdf-text-layer .markedContent span:not(.markedContent)) {
  z-index: 1;
  --font-height: 0;
  font-size: calc(var(--text-scale-factor) * var(--font-height));
  --scale-x: 1;
  --rotate: 0deg;
  transform: rotate(var(--rotate)) scaleX(var(--scale-x)) scale(calc(1 / var(--min-font-size)));
}

:deep(.pdf-text-layer .markedContent) {
  display: contents;
}

:deep(.pdf-text-layer span[role="img"]) {
  user-select: none;
  cursor: default;
}

:deep(.pdf-text-layer ::selection) {
  background: color-mix(in srgb, AccentColor, transparent 75%);
}

:deep(.pdf-text-layer .pdf-search-match) {
  background: rgba(253, 224, 71, 0.45);
  border-radius: 2px;
  box-shadow: 0 0 0 2px rgba(250, 204, 21, 0.2);
}

:deep(.pdf-text-layer .pdf-search-match.is-active-pdf-search-match) {
  background: rgba(250, 204, 21, 0.72);
  box-shadow:
    0 0 0 2px rgba(37, 99, 235, 0.72),
    0 8px 18px rgba(37, 99, 235, 0.18);
}

:deep(.pdf-text-layer br::selection) {
  background: transparent;
}

:deep(.pdf-text-layer .endOfContent) {
  display: block;
  position: absolute;
  inset: 100% 0 0;
  z-index: 0;
  cursor: default;
  user-select: none;
}
</style>
