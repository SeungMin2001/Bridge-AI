<script setup>
import { ref, defineEmits, nextTick } from 'vue'
import MultimodalInput from './MultimodalInput.vue'

const emit = defineEmits(['sendMessage', 'openReference'])

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
    const isMathQuery = params.input.includes('수학')
    const aiMessage = { role: 'assistant', content: '', references: [] }
    messages.value.push(aiMessage)
    
    // add references for math
    if (isMathQuery) {
      aiMessage.summary = "수학은 논리와 기호학을 기반으로 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문입니다. 각 강의에서는 수학적 사고의 뼈대가 되는 공리부터 실생활에 적용되는 응용 수학까지 폭넓게 다룹니다."
      aiMessage.references = [
        { id: 'lec1', title: '강의 1: 수학의 기초', script: '이 강의에서는 수학의 가장 기초가 되는 논리와 집합론에 대해 다룹니다.\n\n수학은 우리 생활 모든 곳에 스며들어 있으며 변해야 할 것과 변하지 않아야 할 것을 명확히 구분하는 학문입니다.\n\n먼저 기본 공리에 대해 알아보겠습니다...' },
        { id: 'lec2', title: '강의 2: 대수학 입문', script: '방정식과 변수에 대한 이해를 돕는 대수학 입문 강의 전사 내용입니다.\n\n미지수 x를 구하기 위해 우리는 양변에 같은 조작을 가해야 합니다.\n이러한 원칙은 복잡한 식을 간결하게 만듭니다.' },
        { id: 'lec3', title: '강의 3: 실생활 미적분', script: '우리 주변에서 발견할 수 있는 변화율과 미적분 활용 사례에 대한 스크립트입니다.\n\n자동차가 가속할 때 속도의 변화량, 즉 가속도를 계산하는 것이 미분의 기초이며, 총 이동 거리를 구하는 것이 적분의 기초입니다.' }
      ]
    }

    const fullResponse = isMathQuery 
      ? "수학에 관한 자료를 바탕으로 종합된 설명을 요약해 보았습니다. 더 자세한 원본 스크립트는 아래 근거 링크를 클릭하여 확인해 보세요:"
      : "안녕하세요! 파일 요약이나 새로운 문서 작업 등 어떤 것을 도와드릴까요?"
    
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
    }, 30) // slightly faster typing
  }, 800) // 800ms "thinking" delay
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
        <div class="w-full max-w-[700px] flex flex-col gap-6 pt-12 pb-[160px]">
          
          <div v-for="(msg, idx) in messages" :key="idx" 
               :class="['flex w-full gap-3', msg.role === 'user' ? 'flex-row-reverse' : 'flex-row items-start']">
            
            <!-- AI Avatar (Animating when it's the latest message being generated) -->
            <div v-if="msg.role === 'assistant'" class="flex-shrink-0 mt-1">
              <div :class="[
                'w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm border border-white/20 overflow-hidden',
                isGenerating && idx === messages.length - 1 ? 'ring-2 ring-indigo-400/30' : ''
              ]">
                <span :class="[
                  'material-symbols-outlined text-[18px] text-white',
                  isGenerating && idx === messages.length - 1 ? 'animate-spin-slow' : ''
                ]" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
              </div>
            </div>

            <!-- Message Bubble -->
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
              <div class="whitespace-pre-wrap">{{ msg.content }}</div>
              
              <!-- Summary Block -->
              <div v-if="msg.summary" class="mt-4 p-4 bg-indigo-50/40 rounded-xl border border-indigo-100/50">
                <div class="flex items-center gap-2 mb-2 text-indigo-800 font-semibold text-[13px] uppercase tracking-wider">
                  <span class="material-symbols-outlined text-[16px]">summarize</span>
                  종합된 설명
                </div>
                <p class="text-[14px] text-gray-700 leading-relaxed">{{ msg.summary }}</p>
              </div>

              <!-- Reference Links -->
              <div v-if="msg.references && msg.references.length > 0" class="flex flex-wrap gap-2 mt-4 pt-4 border-t border-black/10">
                <button 
                  v-for="ref in msg.references" 
                  :key="ref.id" 
                  @click="emit('openReference', ref)"
                  class="flex items-center gap-1 text-[13px] bg-indigo-50/50 hover:bg-indigo-100 text-indigo-700 px-3 py-1.5 rounded-full border border-indigo-200/50 transition-colors shadow-sm font-medium"
                >
                  <span class="material-symbols-outlined text-[14px]">link</span>
                  {{ ref.title }}
                </button>
              </div>
            </div>
          </div>

          <!-- Thinking Dots Animation (Shown before the AI message starts typing) -->
          <Transition name="fade-fast">
            <div v-if="isGenerating && messages.length > 0 && messages[messages.length-1].role === 'user'" 
                 class="flex w-full gap-3 flex-row items-start">
              <div class="flex-shrink-0 mt-1">
                <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm border border-white/20 ring-2 ring-indigo-400/30 overflow-hidden animate-pulse-slow">
                  <span class="material-symbols-outlined text-[18px] text-white animate-spin-slow" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
                </div>
              </div>
              <div class="bg-white/80 backdrop-blur-xl border border-black/5 rounded-2xl rounded-tl-none px-6 py-4 shadow-sm flex items-center gap-1.5">
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-400 rounded-full animate-thinking-dot"></div>
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-500 rounded-full animate-thinking-dot [animation-delay:0.2s]"></div>
                <div class="thinking-dot w-1.5 h-1.5 bg-indigo-600 rounded-full animate-thinking-dot [animation-delay:0.4s]"></div>
              </div>
            </div>
          </Transition>

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
          <div class="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg transform -rotate-6 transition-transform hover:rotate-0 duration-500">
            <span class="material-symbols-outlined text-[28px] text-white" style="font-variation-settings: 'FILL' 1">auto_awesome</span>
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

      <!-- Recent Files Section -->
      <div v-if="messages.length === 0" class="w-full max-w-[600px] pointer-events-auto flex flex-col gap-4 mt-12 opacity-80 animate-fade-in-up shrink-0" style="animation-duration: 0.6s; animation-delay: 0.2s; animation-fill-mode: both;">
        <h3 class="text-sm font-semibold text-gray-500 px-2 uppercase tracking-wider font-sans">최근 연 파일</h3>
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

@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes thinking-dot {
  0%, 100% { transform: translateY(0); opacity: 0.5; }
  50% { transform: translateY(-4px); opacity: 1; }
}

@keyframes pulse-slow {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.8; transform: scale(1.05); }
}

.animate-fade-in-up {
  animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.animate-spin-slow {
  animation: spin-slow 3s linear infinite;
}

.animate-thinking-dot {
  animation: thinking-dot 1.2s ease-in-out infinite;
}

.animate-pulse-slow {
  animation: pulse-slow 2s ease-in-out infinite;
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

.fade-fast-enter-active,
.fade-fast-leave-active {
  transition: opacity 0.2s ease;
}
.fade-fast-enter-from,
.fade-fast-leave-to {
  opacity: 0;
}

.custom-scrollbar::-webkit-scrollbar {
  width: 5px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 10px;
}
</style>
