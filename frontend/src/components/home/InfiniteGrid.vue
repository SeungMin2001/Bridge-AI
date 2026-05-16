<!-- 배경에 무한히 펼쳐지는 그리드 패턴을 그려주는 장식용 UI 컴포넌트입니다. -->
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const mouseX = ref(0)
const mouseY = ref(0)
const gridOffsetX = ref(0)
const gridOffsetY = ref(0)
const count = ref(0)

let animationFrameId = null
const speedX = 0.5
const speedY = 0.5

const handleMouseMove = (e) => {
  const { left, top } = e.currentTarget.getBoundingClientRect()
  mouseX.value = e.clientX - left
  mouseY.value = e.clientY - top
}

const animate = () => {
  gridOffsetX.value = (gridOffsetX.value + speedX) % 40
  gridOffsetY.value = (gridOffsetY.value + speedY) % 40
  animationFrameId = requestAnimationFrame(animate)
}

onMounted(() => {
  animationFrameId = requestAnimationFrame(animate)
})

onUnmounted(() => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
  }
})
</script>

<template>
  <div
    class="infinite-grid-container"
    @mousemove="handleMouseMove"
  >
    <!-- Background Grid (Static/Low Opacity) -->
    <div class="grid-layer low-opacity">
      <svg class="w-full h-full">
        <defs>
          <pattern
            id="grid-pattern-base"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
            :x="gridOffsetX"
            :y="gridOffsetY"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="currentColor"
              stroke-width="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid-pattern-base)" />
      </svg>
    </div>

    <!-- Active Grid (Masked by Mouse) -->
    <div 
      class="grid-layer high-opacity"
      :style="{
        maskImage: `radial-gradient(300px circle at ${mouseX}px ${mouseY}px, black, transparent)`,
        WebkitMaskImage: `radial-gradient(300px circle at ${mouseX}px ${mouseY}px, black, transparent)`
      }"
    >
      <svg class="w-full h-full">
        <defs>
          <pattern
            id="grid-pattern-active"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
            :x="gridOffsetX"
            :y="gridOffsetY"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="currentColor"
              stroke-width="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid-pattern-active)" />
      </svg>
    </div>

  </div>
</template>

<style scoped>
.infinite-grid-container {
  position: absolute;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  background-color: var(--copy-bg);
  pointer-events: auto;
}

.grid-layer {
  position: absolute;
  inset: 0;
  z-index: 0;
}

.low-opacity {
  opacity: 0.05;
  color: #64748b; /* text-muted-foreground */
}

.high-opacity {
  opacity: 0.4;
  color: #3b82f6; /* primary or specific color */
}

.w-full { width: 100%; }
.h-full { height: 100%; }

</style>
