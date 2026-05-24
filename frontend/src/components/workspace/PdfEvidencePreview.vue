<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

const props = defineProps({
  cite: { type: Object, default: null },
  mode: { type: String, default: 'compact' },
})

const canvasRef = ref(null)
const isLoading = ref(false)
const hasError = ref(false)

let loadingTask = null
let pdfDocument = null
let renderTask = null
let renderToken = 0

const pageNumber = computed(() => {
  const page = Number(props.cite?.page || 1)
  return Number.isFinite(page) && page > 0 ? page : 1
})

const materialTitle = computed(() => (
  props.cite?.material_name
  || props.cite?.file_title
  || props.cite?.stored_name
  || 'PDF 자료'
))

const pdfUrl = computed(() => {
  if (props.cite?.material_url) return props.cite.material_url
  if (props.cite?.url) return props.cite.url
  if (!props.cite?.stored_name) return ''
  return `/workspace/uploads/materials/${encodeURIComponent(props.cite.stored_name)}`
})

async function cleanup() {
  try {
    renderTask?.cancel?.()
  } catch {
    // 렌더링 취소 실패는 무시합니다.
  }
  renderTask = null

  try {
    await loadingTask?.destroy?.()
  } catch {
    // 로딩 취소 실패는 무시합니다.
  }
  loadingTask = null

  try {
    await pdfDocument?.destroy?.()
  } catch {
    // 문서 정리 실패는 무시합니다.
  }
  pdfDocument = null
}

async function renderPreview() {
  const url = pdfUrl.value
  if (!url) {
    hasError.value = true
    return
  }

  const token = ++renderToken
  isLoading.value = true
  hasError.value = false
  await cleanup()

  try {
    await nextTick()
    if (!canvasRef.value) {
      throw new Error('PDF preview canvas is unavailable.')
    }

    loadingTask = pdfjsLib.getDocument(url)
    const document = await loadingTask.promise
    if (token !== renderToken) return

    pdfDocument = document
    const page = await document.getPage(Math.min(pageNumber.value, document.numPages))
    if (token !== renderToken) return

    const canvas = canvasRef.value
    const context = canvas.getContext('2d')
    if (!context) throw new Error('PDF preview canvas context is unavailable.')

    const baseViewport = page.getViewport({ scale: 1 })
    const targetWidth = props.mode === 'page' ? 520 : 260
    const scale = Math.max(0.18, targetWidth / baseViewport.width)
    const viewport = page.getViewport({ scale })

    canvas.width = Math.floor(viewport.width)
    canvas.height = Math.floor(viewport.height)
    renderTask = page.render({ canvasContext: context, viewport })
    await renderTask.promise
  } catch (error) {
    if (error?.name !== 'RenderingCancelledException') {
      hasError.value = true
    }
  } finally {
    if (token === renderToken) {
      isLoading.value = false
    }
  }
}

watch(
  () => [pdfUrl.value, pageNumber.value],
  () => renderPreview(),
  { immediate: true }
)

onBeforeUnmount(() => {
  renderToken += 1
  cleanup()
})
</script>

<template>
  <div class="pdf-evidence-preview" :class="{ 'is-page-preview': mode === 'page' }">
    <div v-if="mode !== 'page'" class="pdf-evidence-preview-top">
      <span class="material-symbols-outlined">picture_as_pdf</span>
      <span class="pdf-evidence-title">{{ materialTitle }}</span>
      <strong>p.{{ pageNumber }}</strong>
    </div>
    <div class="pdf-evidence-canvas-wrap">
      <canvas v-show="!hasError" ref="canvasRef"></canvas>
      <div v-if="isLoading" class="pdf-evidence-state">PDF 페이지를 불러오는 중</div>
      <div v-else-if="hasError" class="pdf-evidence-state">PDF 미리보기를 열 수 없습니다.</div>
    </div>
  </div>
</template>

<style scoped>
.pdf-evidence-preview {
  min-width: 0;
  display: grid;
  gap: 10px;
}

.pdf-evidence-preview-top {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
  color: #334155;
  font-size: 12px;
  font-weight: 850;
}

.pdf-evidence-preview-top .material-symbols-outlined {
  color: #4b6a4e;
  font-size: 17px;
}

.pdf-evidence-title {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pdf-evidence-preview-top strong {
  color: #4b6a4e;
  font-size: 11px;
  font-weight: 900;
}

.pdf-evidence-canvas-wrap {
  position: relative;
  width: 100%;
  height: 164px;
  overflow: hidden;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #e5edf6;
}

.pdf-evidence-canvas-wrap canvas {
  display: block;
  width: 100%;
  height: auto;
  background: #ffffff;
}

.pdf-evidence-state {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  text-align: center;
  color: #64748b;
  background: rgba(248, 250, 252, 0.92);
  font-size: 12px;
  font-weight: 800;
}

.pdf-evidence-preview.is-page-preview {
  gap: 0;
}

.pdf-evidence-preview.is-page-preview .pdf-evidence-canvas-wrap {
  height: auto;
  min-height: 280px;
  overflow: visible;
  border-radius: 0;
  border: 0;
  background: #fff;
}

.pdf-evidence-preview.is-page-preview .pdf-evidence-canvas-wrap canvas {
  width: 100%;
  height: auto;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
}
</style>
