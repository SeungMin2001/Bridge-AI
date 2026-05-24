<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { buildHighlightedCitationHtml, normalizeCitation } from './citationUtils'
import PdfEvidencePreview from '../PdfEvidencePreview.vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  cite: { type: Object, default: null },
  position: { type: Object, default: () => ({ x: 0, y: 0 }) },
})

const emit = defineEmits(['close', 'openSource'])

const scrollRef = ref(null)

const citation = computed(() => (
  props.cite ? normalizeCitation(props.cite, 0) : null
))

const isMaterialCitation = computed(() => citation.value?.type === 'material')

const highlightedBody = computed(() => {
  const item = citation.value
  if (!item) return ''
  return buildHighlightedCitationHtml(item.fullText, item.excerpt)
})

const citationLabel = computed(() => props.cite?.citation || citation.value?.locationLabel || '근거 정보')

const popoverStyle = computed(() => {
  const viewportWidth = typeof window === 'undefined' ? 1200 : window.innerWidth
  const viewportHeight = typeof window === 'undefined' ? 900 : window.innerHeight
  const inset = 16
  const width = Math.min(390, viewportWidth - inset * 2)
  const maxHeight = Math.min(620, viewportHeight - inset * 2)
  const left = clamp(Number(props.position?.x || inset), inset, viewportWidth - width - inset)
  const top = clamp(Number(props.position?.y || inset), inset, viewportHeight - maxHeight - inset)

  return {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    maxHeight: `${maxHeight}px`,
  }
})

watch(
  () => [props.visible, props.cite],
  async () => {
    if (!props.visible) return
    await nextTick()
    if (isMaterialCitation.value) return
    scrollRef.value?.querySelector?.('.cite-highlighted-script')?.scrollIntoView?.({
      block: 'center',
      inline: 'nearest',
    })
  },
  { flush: 'post' }
)

function openSource() {
  if (!props.cite) return
  emit('openSource', props.cite)
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max))
}
</script>

<template>
  <Teleport to="body">
    <transition name="popover-fade">
      <div v-if="visible && citation" class="cite-popover-overlay" @click.self="emit('close')">
        <div class="cite-popover" :style="popoverStyle">
          <div class="cite-popover-header">
            <div class="cite-popover-title">
              <span
                class="cite-popover-title-icon material-symbols-outlined"
                :class="{ 'is-audio': citation.type === 'transcript' }"
              >
                {{ citation.icon }}
              </span>
              <span class="cite-popover-title-text">
                <span class="cite-popover-main-title">{{ citation.title }}</span>
                <p>{{ citationLabel }}</p>
              </span>
            </div>
            <button class="cite-popover-close-btn" aria-label="근거 정보 닫기" @click="emit('close')">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>

          <div
            v-if="isMaterialCitation"
            class="cite-pdf-scroll custom-scrollbar"
          >
            <PdfEvidencePreview :cite="props.cite" mode="page" />
          </div>

          <div v-else ref="scrollRef" class="cite-transcript-scroll custom-scrollbar">
            <div
              class="cite-transcript-body whitespace-pre-wrap break-keep"
              v-html="highlightedBody"
            />
          </div>

          <div class="cite-source-wrap">
            <button
              type="button"
              class="cite-source-title"
              @click="openSource"
              :title="citation.title"
            >
              <span
                class="cite-source-icon material-symbols-outlined"
                :class="{ 'is-audio': citation.type === 'transcript' }"
              >
                {{ citation.icon }}
              </span>
              <span>
                <small>{{ citation.sourceCaption }} · {{ citation.locationLabel }}</small>
                <strong>소스 보기</strong>
              </span>
              <span class="material-symbols-outlined cite-source-arrow">open_in_new</span>
            </button>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<style scoped>
.cite-popover-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: transparent;
}

.cite-popover {
  position: fixed;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 24px 64px rgba(15, 23, 42, 0.16);
  border: 1px solid rgba(226, 232, 240, 0.95);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transform-origin: right top;
}

.cite-popover-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 18px 20px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.95);
}

.cite-popover-title {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  align-items: flex-start;
  gap: 10px;
  min-width: 0;
}

.cite-popover-title-icon {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #2563eb;
  background: #eff6ff;
  border: 1px solid #dbeafe;
  border-radius: 9px;
  font-size: 17px;
}

.cite-popover-title-icon.is-audio {
  color: #f59e0b;
  background: rgba(255, 242, 207, 0.9);
  border-color: rgba(255, 242, 207, 0.9);
}

.cite-popover-title-text {
  min-width: 0;
}

.cite-popover-main-title {
  display: block;
  color: #111827;
  font-size: 15px;
  font-weight: 800;
  line-height: 1.35;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cite-popover-title p {
  max-width: 300px;
  margin-top: 5px;
  color: #64748b;
  font-size: 11px;
  font-weight: 650;
  line-height: 1.45;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cite-popover-close-btn {
  width: 30px;
  height: 30px;
  color: #64748b;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.cite-popover-close-btn:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.cite-popover-close-btn .material-symbols-outlined {
  font-size: 19px;
}

.cite-transcript-scroll {
  min-height: 180px;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 20px 22px;
}

.cite-pdf-scroll {
  flex: 1 1 auto;
  min-height: 280px;
  overflow-y: auto;
  padding: 18px 20px;
  background: #f8fafc;
}

.cite-transcript-body {
  color: #1f2937;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.78;
}

:deep(.cite-highlighted-script) {
  background: rgba(253, 224, 71, 0.44);
  color: #111827;
  font-weight: 850;
  border-radius: 6px;
  padding: 2px 4px;
  margin: 0 -2px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

.cite-source-wrap {
  padding: 14px 20px;
  border-top: 1px solid rgba(226, 232, 240, 0.95);
  background: #fff;
}

.cite-source-title {
  width: 100%;
  min-width: 0;
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) 18px;
  align-items: center;
  gap: 10px;
  color: #334155;
  text-align: left;
  cursor: pointer;
}

.cite-source-title > .cite-source-icon {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #475569;
  font-size: 17px;
  border-radius: 10px;
  background: #f1f5f9;
}

.cite-source-title > .cite-source-icon.is-audio {
  color: #f59e0b;
  background: rgba(255, 242, 207, 0.9);
}

.cite-source-title small {
  display: block;
  color: #94a3b8;
  font-size: 10px;
  font-weight: 850;
  line-height: 1.1;
}

.cite-source-title strong {
  display: block;
  margin-top: 3px;
  color: #2563eb;
  font-size: 14px;
  font-weight: 800;
  line-height: 1.25;
}

.cite-source-title:hover strong {
  text-decoration: underline;
  text-underline-offset: 3px;
}

.cite-source-arrow {
  color: #94a3b8;
  font-size: 17px;
}

.popover-fade-enter-active {
  transition: all 0.2s ease;
}

.popover-fade-leave-active {
  transition: all 0.15s ease;
}

.popover-fade-enter-from,
.popover-fade-leave-to {
  opacity: 0;
  transform: scale(0.96) translateY(4px);
}
</style>
