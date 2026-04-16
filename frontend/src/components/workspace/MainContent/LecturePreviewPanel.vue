<!-- 워크스페이스 메모 탭에서 PDF 또는 PPT 강의 자료 미리보기를 표시하는 패널입니다. -->
<script setup>
import { ref, watch, nextTick, onBeforeUnmount } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import { PPTXViewer } from 'pptxviewjs'
import PptPreviewToolbar from './PptPreviewToolbar.vue'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

const props = defineProps({
  material: { type: Object, default: null }
})

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

let activePdfTask = null
let activePdfDocument = null
let pdfRenderToken = 0
let activePdfTextLayers = []

const isPdfAttachment = (file) => /\.pdf$/i.test(file?.name || '')
const isPptAttachment = (file) => /\.(ppt|pptx)$/i.test(file?.name || '')

const destroyPptViewer = () => {
  pptViewer.value?.destroy?.()
  pptViewer.value = null
  pptSlideIndex.value = 0
  pptSlideCount.value = 0
  pptLoading.value = false
  pptError.value = ''
}

const clearPdfPreview = () => {
  pdfPageCount.value = 0
  pdfLoading.value = false
  pdfError.value = ''
  if (pdfContainerRef.value) {
    pdfContainerRef.value.innerHTML = ''
  }
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

    const availableWidth = Math.max((pdfContainerRef.value.clientWidth || 960) - 40, 320)

    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
      if (renderToken !== pdfRenderToken) return

      const page = await pdfDocument.getPage(pageNumber)
      const initialViewport = page.getViewport({ scale: 1 })
      const scale = Math.max(0.75, Math.min(1.6, availableWidth / initialViewport.width))
      const viewport = page.getViewport({ scale })
      const canvas = document.createElement('canvas')
      const context = canvas.getContext('2d')

      if (!context) {
        throw new Error('PDF 캔버스 컨텍스트를 가져오지 못했습니다.')
      }

      canvas.width = viewport.width
      canvas.height = viewport.height
      canvas.className = 'pdf-preview-canvas'
      canvas.style.width = `${viewport.width}px`
      canvas.style.height = `${viewport.height}px`

      const pageShell = document.createElement('div')
      pageShell.className = 'pdf-page-shell'

      const pageMeta = document.createElement('div')
      pageMeta.className = 'pdf-page-meta'
      pageMeta.textContent = `${pageNumber} / ${pdfDocument.numPages}`

      const pageStage = document.createElement('div')
      pageStage.className = 'pdf-page-stage'
      pageStage.style.width = `${viewport.width}px`
      pageStage.style.height = `${viewport.height}px`
      pageStage.style.setProperty('--total-scale-factor', '1')

      const textLayerDiv = document.createElement('div')
      textLayerDiv.className = 'textLayer pdf-text-layer'
      textLayerDiv.style.width = `${viewport.width}px`
      textLayerDiv.style.height = `${viewport.height}px`
      textLayerDiv.style.setProperty('--total-scale-factor', '1')

      const textLayer = new pdfjsLib.TextLayer({
        textContentSource: page.streamTextContent({
          includeMarkedContent: true,
          disableNormalization: true
        }),
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
  } catch (error) {
    console.error(error)
    pdfError.value = 'PDF를 화면에 불러오지 못했습니다.'
  } finally {
    if (renderToken === pdfRenderToken) {
      pdfLoading.value = false
    }
  }
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
  () => props.material,
  async (file) => {
    if (!file) {
      destroyPptViewer()
      await destroyPdfPreview()
      return
    }

    if (isPdfAttachment(file)) {
      destroyPptViewer()
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

onBeforeUnmount(() => {
  destroyPptViewer()
  destroyPdfPreview()
})
</script>

<template>
  <div class="lecture-preview-shell">
    <div v-if="isPdfAttachment(material)" class="lecture-preview-frame-wrap">
      <div class="lecture-preview-surface">
        <div class="lecture-preview-surface-header">
          <div>
            <p class="lecture-preview-type">PDF Preview</p>
            <h3 class="lecture-preview-title">{{ material?.name }}</h3>
          </div>
          <span v-if="pdfPageCount" class="lecture-preview-page-count">{{ pdfPageCount }} pages</span>
        </div>

        <div class="pdf-preview-stage">
          <div ref="pdfContainerRef" class="pdf-preview-scroll custom-scrollbar" :class="{ 'is-hidden': pdfLoading || pdfError }"></div>
          <div v-if="pdfLoading" class="pdf-preview-placeholder">PDF를 불러오는 중입니다.</div>
          <div v-else-if="pdfError" class="pdf-preview-placeholder">{{ pdfError }}</div>
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
  gap: 16px;
  padding: 28px;
  border-radius: 28px;
  border: 1px solid rgba(255, 255, 255, 0.88);
  background:
    radial-gradient(circle at top left, rgba(255, 255, 255, 0.94), transparent 36%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.86));
  box-shadow: 0 20px 36px rgba(148, 163, 184, 0.1);
}

.lecture-preview-surface-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 10px;
}

.lecture-preview-type {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  color: #94a3b8;
  text-transform: uppercase;
}

.lecture-preview-title {
  margin-top: 4px;
  font-size: 34px;
  font-weight: 800;
  color: #1d1d1f;
  letter-spacing: -0.03em;
  word-break: break-word;
  line-height: 1.08;
}

.lecture-preview-page-count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(226, 232, 240, 0.92);
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
}

.pdf-preview-stage {
  flex: 1;
  min-height: 0;
  position: relative;
}

.pdf-preview-scroll {
  height: 100%;
  overflow-y: auto;
  padding: 4px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.pdf-preview-scroll.is-hidden {
  visibility: hidden;
}

.pdf-preview-placeholder {
  position: absolute;
  inset: 0;
  border-radius: 22px;
  border: 1px dashed rgba(148, 163, 184, 0.32);
  background: rgba(248, 250, 252, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #64748b;
  font-size: 14px;
  font-weight: 600;
}

.lecture-preview-fallback {
  min-height: 100%;
  border-radius: 28px;
  border: 1px solid rgba(255, 255, 255, 0.88);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.82));
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  gap: 10px;
  padding: 24px;
  text-align: center;
  box-shadow: 0 20px 36px rgba(148, 163, 184, 0.1);
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
  padding: 16px;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid rgba(226, 232, 240, 0.82);
  box-shadow: 0 14px 30px rgba(148, 163, 184, 0.08);
}

:deep(.pdf-page-meta) {
  margin-bottom: 12px;
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
  border-radius: 18px;
  background: #ffffff;
}

:deep(.pdf-page-stage) {
  position: relative;
  margin: 0 auto;
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
