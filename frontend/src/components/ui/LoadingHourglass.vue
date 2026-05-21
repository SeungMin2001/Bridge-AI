<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import lottie from 'lottie-web/build/player/lottie_light'


const props = defineProps({
  src: { type: String, default: '/animations/Remix%20Hourglass.json' },
  size: { type: [Number, String], default: 72 },
  width: { type: [Number, String], default: null },
  height: { type: [Number, String], default: null },
  fit: { type: String, default: 'meet' },
  contentScale: { type: [Number, String], default: 1 },
  autoplay: { type: Boolean, default: true },
  loop: { type: Boolean, default: true },
  fallbackIcon: { type: String, default: 'hourglass_top' }
})

const containerRef = ref(null)
const hasError = ref(false)
let animation = null
let loadToken = 0
let pendingPlayFromStart = false

const cssSize = computed(() => (
  typeof props.size === 'number' ? `${props.size}px` : props.size
))
const cssWidth = computed(() => {
  const value = props.width ?? props.size
  return typeof value === 'number' ? `${value}px` : value
})
const cssHeight = computed(() => {
  const value = props.height ?? props.size
  return typeof value === 'number' ? `${value}px` : value
})
const preserveAspectRatio = computed(() => (
  props.fit === 'slice' ? 'xMidYMid slice' : 'xMidYMid meet'
))
const cssContentScale = computed(() => `${props.contentScale || 1}`)

const destroyAnimation = () => {
  if (!animation) return
  animation.destroy()
  animation = null
}

const playFromStart = () => {
  if (!animation) {
    pendingPlayFromStart = true
    return
  }
  pendingPlayFromStart = false
  animation.stop()
  animation.goToAndPlay(0, true)
}

const loadAnimation = async () => {
  const currentToken = ++loadToken
  destroyAnimation()
  hasError.value = false

  await nextTick()

  const container = containerRef.value
  if (!container) return

  try {
    const response = await fetch(props.src)
    if (!response.ok) throw new Error(`Failed to load Lottie: ${response.status}`)
    const animationData = await response.json()

    if (currentToken !== loadToken || !containerRef.value) return

    animation = lottie.loadAnimation({
      container,
      renderer: 'svg',
      loop: props.loop,
      autoplay: props.autoplay,
      animationData,
      rendererSettings: {
        preserveAspectRatio: preserveAspectRatio.value,
        progressiveLoad: true
      }
    })

    animation.addEventListener('DOMLoaded', () => {
      if (pendingPlayFromStart) playFromStart()
    })

    if (pendingPlayFromStart) {
      requestAnimationFrame(() => playFromStart())
    }
  } catch (error) {
    if (currentToken !== loadToken) return
    console.warn('[loading-hourglass] animation unavailable:', error)
    hasError.value = true
  }
}

watch(() => props.src, loadAnimation)

onMounted(loadAnimation)

onBeforeUnmount(() => {
  loadToken += 1
  destroyAnimation()
})

defineExpose({
  playFromStart
})
</script>

<template>
  <span
    class="loading-hourglass"
    :style="{
      width: cssWidth || cssSize,
      height: cssHeight || cssSize,
      '--lottie-content-scale': cssContentScale
    }"
    aria-hidden="true"
  >
    <span
      v-if="hasError"
      class="material-symbols-outlined loading-hourglass__fallback"
    >
      {{ fallbackIcon }}
    </span>
    <span
      v-else
      ref="containerRef"
      class="loading-hourglass__canvas"
    ></span>
  </span>
</template>

<style scoped>
.loading-hourglass {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  line-height: 0;
  overflow: visible;
}

.loading-hourglass__canvas {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.loading-hourglass__canvas :deep(svg) {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
  transform: scale(var(--lottie-content-scale, 1));
  transform-origin: center;
}

.loading-hourglass__fallback {
  color: #c7c7cc;
  font-size: min(42px, 72%);
  line-height: 1;
}
</style>
