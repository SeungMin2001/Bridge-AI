<!-- 애니메이션 효과가 적용된 탭 전환 인터페이스를 제공하는 UI 컴포넌트입니다. -->
<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'

const props = defineProps({
  tabs: {
    type: Array,
    required: true
  },
  modelValue: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['update:modelValue'])

const containerRef = ref(null)
const activeTabRef = ref(null)

const updateClipPath = () => {
  const container = containerRef.value
  const activeTabElement = activeTabRef.value

  if (container && activeTabElement) {
    const { offsetLeft, offsetWidth } = activeTabElement

    const clipLeft = offsetLeft
    const clipRight = offsetLeft + offsetWidth

    const leftPercent = ((clipLeft / container.offsetWidth) * 100).toFixed(2)
    const rightPercent = (100 - (clipRight / container.offsetWidth) * 100).toFixed(2)

    container.style.clipPath = `inset(0 ${rightPercent}% 0 ${leftPercent}% round 10px)`
  }
}

watch(() => props.modelValue, async () => {
  await nextTick()
  updateClipPath()
})

onMounted(() => {
  setTimeout(() => {
    updateClipPath()
  }, 100)
})
</script>

<template>
  <div class="relative flex w-fit items-center p-0">
    
    <!-- Animated Background Layer (The Black "active-tab" Look) -->
    <div
      ref="containerRef"
      class="absolute z-10 top-0 bottom-0 left-0 right-0 overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] pointer-events-none"
    >
      <div class="relative flex h-full bg-[#1d1d1f] rounded-[10px]">
        <div 
          v-for="tab in tabs" 
          :key="'bg-' + tab.key"
          class="px-4 py-1.5 text-[13px] font-bold whitespace-nowrap opacity-100 text-white flex items-center"
        >
          {{ tab.label }}
        </div>
      </div>
    </div>

    <!-- Interactive Foreground Layer (Inactive Tabs Look) -->
    <div class="relative flex z-20">
      <button
        v-for="tab in tabs"
        :key="'fg-' + tab.key"
        :ref="(el) => { if (modelValue === tab.key) activeTabRef = el }"
        @click="emit('update:modelValue', tab.key)"
        type="button"
        class="px-4 py-1.5 rounded-[10px] text-[13px] whitespace-nowrap z-30 text-[#8e8e93] font-medium hover:text-[#1d1d1f] transition-colors"
      >
        {{ tab.label }}
      </button>
    </div>
    
  </div>
</template>

<style scoped>
/* 
   별도의 스코프 스타일이 필요하지 않도록 인라인 색상을 사용했습니다. 
   배경색 #1d1d1f는 기존 .active-tab과 동일합니다.
*/
</style>
