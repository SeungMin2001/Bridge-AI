<script setup>
const FOLDER_COLORS = {
  '#3b82f6': { body: 'fc-blue', tab: 'fc-blue-tab' },
  '#2dd4bf': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#ef4444': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f87171': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f59e0b': { body: 'fc-amber', tab: 'fc-amber-tab' },
  '#10b981': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#8b5cf6': { body: 'fc-purple', tab: 'fc-purple-tab' },
  '#a78bfa': { body: 'fc-purple', tab: 'fc-purple-tab' },
}

defineProps({
  currentItems: { type: Array, default: () => [] },
  favorites: { type: Set, default: () => new Set() },
  navigationStack: { type: Array, default: () => [] },
  currentTitle: { type: String, default: '' }
})

const emit = defineEmits([
  'toggleStar',
  'enterFolder',
  'navigate',
  'goBack',
  'openFolderModal',
  'openFileModal'
])
</script>

<template>
  <div class="flex-1 mt-6"> 
    <div class="shrink-0">
      <div class="grid-header flex items-center justify-between">
        <div class="flex items-center gap-3">
          <button 
            v-if="navigationStack.length > 0"
            @click="emit('goBack')"
            class="flex items-center justify-center p-2 bg-white border-none text-[#1d1d1f] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
          >
            <span class="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <span class="section-title !m-0 transition-all duration-300">{{ currentTitle }}</span>
        </div>

        <div class="flex items-center gap-4">
          <!-- Create Group moved from bottom -->
          <div class="flex items-center gap-2">
            <button class="header-action-btn group" @click="emit('openFolderModal')">
              <span class="material-symbols-outlined group-hover:scale-110 transition-transform">create_new_folder</span>
              <span>새 폴더</span>
            </button>
            <button class="header-action-btn group" @click="emit('openFileModal')">
              <span class="material-symbols-outlined group-hover:scale-110 transition-transform">description</span>
              <span>새 파일</span>
            </button>
          </div>

          <div class="w-[1px] h-4 bg-black/10 mx-1"></div>

          <div class="flex items-center gap-2">
            <button 
              @click="emit('navigate', 'ai-history')"
              class="flex items-center justify-center p-2 bg-white border-none text-[#3a3a3c] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
              title="AI 명령 기록"
            >
              <span class="material-symbols-outlined text-[20px]">history</span>
            </button>
            <button class="flex items-center gap-1 px-3 py-1.5 bg-white border-none text-[14px] font-bold text-[#3b82f6] cursor-pointer rounded-xl hover:bg-blue-50 transition-colors shadow-sm">
              전체보기
              <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
            </button>
          </div>
        </div>
      </div>
      
      <div class="folder-grid">
        <template v-for="item in currentItems" :key="item.id">
          <!-- Folder Card -->
          <div v-if="item.type === 'folder'" class="folder-card" @click="emit('enterFolder', $event, item)">
            <div :class="['folder-back', FOLDER_COLORS[item.color]?.body || 'fc-blue']">
              <div :class="['folder-tab', FOLDER_COLORS[item.color]?.tab || 'fc-blue-tab']" style="width: 45%;"></div>
            </div>
            <div class="folder-paper"></div>
            <div :class="['folder-body', FOLDER_COLORS[item.color]?.body || 'fc-blue']">
              <button
                :class="['star-btn', { starred: favorites.has(item.id) }]"
                @click="emit('toggleStar', $event, item.id)"
              >
                <span class="material-symbols-outlined" :style="{ fontSize: '16px', fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
              </button>
              <div class="folder-icon-area">
                <span class="material-symbols-outlined" style="font-size: 22px; color: #fff; font-variation-settings: 'FILL' 1">folder</span>
              </div>
              <div class="folder-card-name">{{ item.name }}</div>
              <div class="folder-card-date">{{ item.date || '' }}</div>
            </div>
          </div>
          
          <!-- File Card -->
          <div v-else class="folder-card file-card" @click="emit('navigate', 'workspace')" style="display: flex; flex-direction: column; height: 160px;">
            <div style="height: 10px; flex-shrink: 0;"></div>
            <div style="background: #fff; border-radius: 14px; padding: 0; flex: 1; position: relative; overflow: hidden; box-shadow: 2px 3px 0px #e0e0e8; border: 1.5px solid #e5e5ea; display: flex; flex-direction: column;">
              <div style="height: 6px; background: linear-gradient(90deg, #6366f1, #a78bfa); border-radius: 12px 12px 0 0;"></div>
              <div style="position: absolute; top: 30px; left: 0; right: 0; bottom: 0; background-image: repeating-linear-gradient(transparent, transparent 22px, #f0f0f5 22px, #f0f0f5 23px); opacity: 0.6;"></div>
              <div style="position: relative; z-index: 1; padding: 14px; display: flex; flex-direction: column; flex: 1;">
                <button
                  :class="['star-btn', { starred: favorites.has(item.id) }]"
                  @click="emit('toggleStar', $event, item.id)"
                  style="position: absolute; top: 14px; right: 10px; background: rgba(0,0,0,0.04); color: #d1d1d6;"
                >
                  <span class="material-symbols-outlined" :style="{ fontSize: '16px', fontVariationSettings: `'FILL' ${favorites.has(item.id) ? 1 : 0}` }">star</span>
                </button>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                  <div style="width: 36px; height: 36px; background: #ede9fe; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span class="material-symbols-outlined" style="font-size: 20px; color: #6366f1; font-variation-settings: 'FILL' 1">article</span>
                  </div>
                  <span style="font-size: 10px; font-weight: 700; color: #6366f1; background: #ede9fe; padding: 2px 8px; border-radius: 100px; letter-spacing: 0.04em;">FILE</span>
                </div>
                <div style="margin-top: auto;">
                  <div class="folder-card-name" style="color: #1d1d1f; font-size: 13px;">{{ item.name }}</div>
                  <div class="folder-card-date" style="color: #8e8e93;">{{ item.date || '' }}</div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <div class="h-[60px] shrink-0"></div>
  </div>
</template>
