<script setup>
import { ref, defineEmits, nextTick } from 'vue'
import MultimodalInput from './MultimodalInput.vue'

const emit = defineEmits(['sendMessage'])

const messages = ref([])
const isGenerating = ref(false)
const chatScrollRef = ref(null)

const scrollToBottom = () => {
  nextTick(() => {
    if (chatScrollRef.value) {
      chatScrollRef.value.scrollTop = chatScrollRef.value.scrollHeight
    }
  })
}

const recentFiles = ref([
  { id: '1', name: '자료구조 강의 노트.pdf', type: 'pdf', date: '오늘' },
  { id: '2', name: 'AI 프로젝트 기획서.docx', type: 'doc', date: '어제' },
  { id: '3', name: '중간고사 요약본.pptx', type: 'ppt', date: '2일 전' }
])

const getFileIcon = (type) => {
  switch (type) {
    case 'pdf': return 'picture_as_pdf'
    case 'ppt': return 'slideshow'
    case 'doc': return 'description'
    default: return 'insert_drive_file'
  }
}

const onSendMessage = (params) => {
  // Add user message
  messages.value.push({ role: 'user', content: params.input, attachments: params.attachments })
  emit('sendMessage', params)
  scrollToBottom()
  
  // Simulate AI generating
  isGenerating.value = true
  
  // Mock AI Response
  setTimeout(() => {
    const aiMessage = { role: 'assistant', content: '' }
    messages.value.push(aiMessage)
    
    const fullResponse = "안녕하세요! 파일 요약이나 새로운 문서 작업 등 어떤 것을 도와드릴까요?"
    let charIndex = 0
    
    const interval = setInterval(() => {
      if (charIndex < fullResponse.length) {
        aiMessage.content += fullResponse[charIndex]
        charIndex++
        scrollToBottom()
      } else {
        clearInterval(interval)
        isGenerating.value = false
      }
    }, 50)
  }, 1000)
}

const onStopGenerating = () => {
  isGenerating.value = false
}
</script>

<template>
  <div class="w-full h-full relative transition-all duration-700">
    
    <!-- Chat History -->
    <Transition name="fade">
      <div v-if="messages.length > 0" 
           ref="chatScrollRef"
           class="absolute inset-0 overflow-y-auto px-4 w-full flex flex-col items-center custom-scrollbar z-10">
        <div class="w-full max-w-[700px] flex flex-col gap-4 pt-12 pb-[160px]">
          <div v-for="(msg, idx) in messages" :key="idx" 
               :class="['flex w-full', msg.role === 'user' ? 'justify-end' : 'justify-start']">
            <div :class="[
              'max-w-[85%] rounded-2xl px-5 py-3.5 text-[15px] leading-relaxed break-words shadow-sm',
              msg.role === 'user' 
                ? 'bg-[#1d1d1f] text-white rounded-tr-none' 
                : 'bg-white/80 backdrop-blur-xl text-[#1d1d1f] border border-black/5 rounded-tl-none shadow-[0_4px_20px_rgba(0,0,0,0.03)]'
            ]">
              <div v-if="msg.attachments?.length" class="flex gap-2 mb-3">
                <div v-for="att in msg.attachments" :key="att.url" class="w-14 h-14 rounded-lg overflow-hidden border border-black/10">
                  <img :src="att.url" class="w-full h-full object-cover" />
                </div>
              </div>
              {{ msg.content }}
            </div>
          </div>
        </div>
      </div>
    </Transition>

    <!-- Layout structure for Title & Input (Flex-grow animation) -->
    <div class="absolute inset-0 flex flex-col items-center pointer-events-none z-20 transition-all duration-700">
      
      <!-- Top dynamic space -->
      <div 
        class="w-full flex-shrink-0 transition-all duration-700" 
        :style="{ flexGrow: 1, transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)' }"
      ></div>

      <!-- Title (Hides when chat starts) -->
      <Transition name="fade">
        <div v-if="messages.length === 0" class="flex flex-col items-center text-center gap-3 pb-8 pointer-events-auto shrink-0 w-full transition-all duration-500">
          <div class="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg transform -rotate-6">
            <span class="material-symbols-outlined text-[28px] text-white">auto_awesome</span>
          </div>
          <div class="text-[32px] font-extrabold text-[#1d1d1f] tracking-tight leading-tight">무엇을 도와드릴까요?</div>
        </div>
      </Transition>

      <!-- Input component -->
      <div class="w-full max-w-[730px] pointer-events-auto flex-shrink-0 z-50 transition-all duration-700">
        <MultimodalInput 
          :is-generating="isGenerating"
          @sendMessage="onSendMessage"
          @stopGenerating="onStopGenerating"
        />
      </div>

      <!-- Recent Files Section (New Position) -->
      <div v-if="messages.length === 0" class="w-full max-w-[600px] pointer-events-auto flex flex-col gap-4 mt-12 opacity-80 animate-fade-in-up shrink-0" style="animation-duration: 0.6s; animation-delay: 0.2s; animation-fill-mode: both;">
        <h3 class="text-sm font-semibold text-gray-500 px-2 uppercase tracking-wider">최근 연 파일</h3>
        <div class="flex gap-4">
          <div v-for="file in recentFiles" :key="file.id" 
               class="flex-1 bg-white/40 backdrop-blur-md border border-white/50 rounded-2xl p-4 flex flex-col gap-3 cursor-pointer hover:-translate-y-1 hover:bg-white/60 hover:shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-all duration-300">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center text-gray-600 shadow-sm border border-black/5">
              <span class="material-symbols-outlined text-[20px]">{{ getFileIcon(file.type) }}</span>
            </div>
            <div class="flex flex-col">
              <span class="text-[14px] font-bold text-[#1d1d1f] truncate leading-tight">{{ file.name }}</span>
              <span class="text-[12px] text-gray-500 mt-0.5">{{ file.date }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom dynamic space -->
      <div 
        class="w-full flex-shrink-0 transition-all duration-700"
        :style="{ 
           flexGrow: messages.length > 0 ? 0 : 1.2, 
           height: messages.length > 0 ? '32px' : '0px',
           transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)' 
        }"
      ></div>

    </div>
    
  </div>
</template>

<style scoped>
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.4s ease, transform 0.4s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.98);
}

.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 10px;
}
</style>
