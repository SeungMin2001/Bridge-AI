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
  <div class="multimodal-container w-full max-w-[820px] mx-auto flex flex-col gap-4 relative z-50">

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
    <div class="multimodal-input-card relative p-2 flex flex-col">
      <textarea
        ref="textareaRef"
        v-model="input"
        rows="1"
        placeholder="무엇이든 물어보세요..."
        class="w-full bg-transparent border-none outline-none focus:ring-0 focus:outline-none text-[#1d1d1f] placeholder:text-[#aaa6b0] px-0 py-0 resize-none min-h-[42px] max-h-[200px] custom-scrollbar text-[16px] font-extrabold leading-relaxed overflow-hidden"
        @keydown.enter.prevent="!$event.isComposing && !$event.shiftKey && handleSubmit()"
      ></textarea>

      <div class="flex items-center justify-between pb-0">
        <button 
          @click="fileInputRef.click()"
          class="multimodal-attach-btn p-2 transition-colors"
        >
          <span class="material-symbols-outlined text-[20px]">attach_file</span>
        </button>
        
        <input type="file" ref="fileInputRef" class="hidden" multiple @change="handleFileUpload" />

        <div class="flex gap-2">
          <button 
            v-if="isGenerating"
            @click="emit('stopGenerating')"
            class="multimodal-send-btn is-stop p-2 rounded-full flex items-center justify-center"
          >
            <span class="material-symbols-outlined text-[20px]">stop</span>
          </button>
          <button 
            v-else
            @click="handleSubmit"
            :disabled="!input.trim() && !attachments.length"
            class="multimodal-send-btn p-2 rounded-full flex items-center justify-center hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_upward</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.multimodal-input-card {
  min-height: 132px;
  border-radius: 32px;
  border: 1px solid rgba(255, 255, 255, 0.92);
  background-color: rgba(255, 255, 255, 0.58);
  padding: 24px 92px 24px 30px !important;
  overflow: hidden;
  box-shadow:
    inset 0 2px 18px rgba(21, 22, 26, 0.09),
    inset 0 -8px 22px rgba(255, 255, 255, 0.72),
    0 18px 42px rgba(48, 42, 58, 0.08);
}

.multimodal-input-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  border: 2px solid rgba(255, 255, 255, 0.88);
  box-shadow: inset 0 0 0 1px rgba(214, 211, 218, 0.52);
  pointer-events: none;
}

.multimodal-attach-btn {
  position: absolute;
  left: 30px;
  bottom: 22px;
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  color: #8f8993;
}

.multimodal-attach-btn:hover {
  color: var(--copy-text);
}

.multimodal-send-btn {
  position: absolute;
  right: 30px;
  top: 50%;
  width: 56px;
  height: 56px;
  transform: translateY(-50%);
  color: #fff;
  background: rgba(170, 163, 151, 0.72);
  box-shadow: none;
}

.multimodal-send-btn.is-stop {
  background: #e5e1ea;
  color: var(--copy-text);
}

.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.1);
  border-radius: 10px;
}

</style>
