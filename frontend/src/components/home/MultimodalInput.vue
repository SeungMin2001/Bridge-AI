<!-- 텍스트 및 이미지 등 다양한 입력을 처리하는 홈 화면의 통합 검색창 컴포넌트입니다. -->
<script setup> import { ref, watch, onMounted, nextTick } from 'vue'

const props = defineProps({
  chatId: { type: String, default: 'default' },
  isGenerating: { type: Boolean, default: false },
  canSend: { type: Boolean, default: true },
})

const emit = defineEmits(['sendMessage', 'stopGenerating'])

const input = ref('')
const attachments = ref([])
const uploadQueue = ref([])
const textareaRef = ref(null)
const fileInputRef = ref(null)

const adjustHeight = () => {
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
    const scrollHeight = textareaRef.value.scrollHeight
    textareaRef.value.style.height = `${scrollHeight}px`
    
    // 200px(max-h)를 넘을 때만 스크롤바 표시
    if (scrollHeight > 200) {
      textareaRef.value.style.overflowY = 'auto'
    } else {
      textareaRef.value.style.overflowY = 'hidden'
    }
  }
}

watch(input, () => {
  nextTick(adjustHeight)
})

const handleFileUpload = (e) => {
  const files = Array.from(e.target.files || [])
  files.forEach(file => {
    const attachment = {
      name: file.name,
      url: URL.createObjectURL(file),
      type: file.type,
      size: file.size
    }
    attachments.value.push(attachment)
  })
  e.target.value = ''
}

const removeAttachment = (index) => {
  const att = attachments.value[index]
  if (att.url.startsWith('blob:')) {
    URL.revokeObjectURL(att.url)
  }
  attachments.value.splice(index, 1)
}

const handleSubmit = () => {
  if (input.value.trim() || attachments.value.length) {
    emit('sendMessage', { input: input.value, attachments: [...attachments.value] })
    input.value = ''
    attachments.value = []
  }
}

onMounted(() => {
  adjustHeight()
})
</script>

<template>
  <div class="multimodal-container w-full max-w-[680px] mx-auto flex flex-col gap-4 relative z-50">

    <!-- Attachments Preview -->
    <div v-if="attachments.length" class="flex gap-3 overflow-x-auto pb-2">
      <div v-for="(att, idx) in attachments" :key="idx" class="relative group shrink-0">
        <div class="w-20 h-16 bg-black/5 rounded-lg flex items-center justify-center overflow-hidden border border-black/10">
          <img v-if="att.type.startsWith('image/')" :src="att.url" class="w-full h-full object-cover" />
          <span v-else class="text-[10px] text-black font-bold opacity-40">{{ att.name.split('.').pop().toUpperCase() }}</span>
        </div>
        <button 
          @click="removeAttachment(idx)"
          class="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <span class="material-symbols-outlined text-[14px]">close</span>
        </button>
      </div>
    </div>

    <!-- Input Area -->
    <div class="relative neo-inner border border-white/50 rounded-[24px] p-2 flex flex-col">
      <textarea
        ref="textareaRef"
        v-model="input"
        rows="1"
        placeholder="무엇이든 물어보세요..."
        class="w-full bg-transparent border-none outline-none focus:ring-0 focus:outline-none text-[#1d1d1f] placeholder:text-[#1d1d1f]/40 px-4 py-2 resize-none min-h-[40px] max-h-[200px] custom-scrollbar text-[15px] leading-relaxed overflow-hidden"
        @keydown.enter.prevent="!$event.isComposing && !$event.shiftKey && handleSubmit()"
      ></textarea>

      <div class="flex items-center justify-between px-2 pb-1">
        <button 
          @click="fileInputRef.click()"
          class="p-2 text-[#1d1d1f]/50 hover:text-[#1d1d1f] transition-colors"
        >
          <span class="material-symbols-outlined text-[20px]">attach_file</span>
        </button>
        
        <input type="file" ref="fileInputRef" class="hidden" multiple @change="handleFileUpload" />

        <div class="flex gap-2">
          <button 
            v-if="isGenerating"
            @click="emit('stopGenerating')"
            class="p-2 bg-gray-300 text-[#1d1d1f] rounded-full flex items-center justify-center neo-card hover:bg-gray-200"
          >
            <span class="material-symbols-outlined text-[20px]">stop</span>
          </button>
          <button 
            v-else
            @click="handleSubmit"
            :disabled="!input.trim() && !attachments.length"
            class="p-2 neo-active-btn text-white rounded-full flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_upward</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.1);
  border-radius: 10px;
}

</style>
