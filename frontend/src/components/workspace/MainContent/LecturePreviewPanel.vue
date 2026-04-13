<!-- 워크스페이스 메모 탭에서 PDF 또는 PPT 강의 자료 미리보기를 표시하는 패널입니다. -->
<script setup>
import { ref, watch, nextTick, onBeforeUnmount } from 'vue'
import { PPTXViewer } from 'pptxviewjs'
import PptPreviewToolbar from './PptPreviewToolbar.vue'

const props = defineProps({
  material: { type: Object, default: null }
})

defineEmits(['close'])

const pptCanvasRef = ref(null)
const pptViewer = ref(null)
const pptSlideIndex = ref(0)
const pptSlideCount = ref(0)
const pptLoading = ref(false)
const pptError = ref('')

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
    if (!file || !isPptAttachment(file)) {
      destroyPptViewer()
      return
    }

    await loadPptPreview(file)
  },
  { immediate: true }
)

onBeforeUnmount(() => {
  destroyPptViewer()
})
</script>

<template>
  <div class="lecture-preview-shell">
    <div class="lecture-preview-header">
      <div class="flex items-center justify-end gap-3 flex-wrap">
        <button type="button" class="preview-close-btn" @click="$emit('close')">
          닫기
        </button>
      </div>
    </div>

    <div v-if="isPdfAttachment(material)" class="lecture-preview-frame-wrap">
      <iframe :src="material.url" class="lecture-preview-frame" title="PDF Preview"></iframe>
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
  padding: 18px;
  border-radius: 22px;
  border: 1px solid rgba(255, 255, 255, 0.84);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(248, 250, 252, 0.8));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.98), 0 18px 36px rgba(148, 163, 184, 0.1);
}

.lecture-preview-header {
  margin-bottom: 14px;
}

.preview-close-btn {
  padding: 8px 14px;
  border-radius: 12px;
  border: 1px solid rgba(229, 231, 235, 0.9);
  background: rgba(255, 255, 255, 0.9);
  font-size: 12px;
  font-weight: 700;
  color: #4b5563;
}

.lecture-preview-frame-wrap {
  width: 100%;
  height: min(70vh, 820px);
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(226, 232, 240, 0.9);
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
  background: rgba(239, 246, 255, 0.5);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  gap: 10px;
  padding: 24px;
  text-align: center;
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
</style>
