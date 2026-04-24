<!-- 폴더 생성 및 파일/폴더 수정에 사용하는 홈 화면 모달 모음입니다. -->
<script setup>
import { computed } from 'vue'

const props = defineProps({
  isFolderModalOpen: Boolean,
  isFileModalOpen: Boolean,
  isMeetingFileModalOpen: Boolean,
  isEditItemModalOpen: Boolean,
  newFolderName: String,
  newFileName: String,
  selectedColor: String,
  editingItemType: { type: String, default: 'file' },
  editingFileKind: { type: String, default: 'lecture' },
  folderColors: { type: Array, default: () => [] },
  lectureFileColors: { type: Array, default: () => [] },
  meetingFileColors: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'update:isFolderModalOpen',
  'update:isFileModalOpen',
  'update:isMeetingFileModalOpen',
  'update:isEditItemModalOpen',
  'update:newFolderName',
  'update:newFileName',
  'update:selectedColor',
  'createFolder',
  'createFile',
  'createMeetingFile',
  'updateItem',
  'deleteEditingItem'
])

const editModalTitle = computed(() => props.editingItemType === 'folder' ? '폴더 정보 수정' : '파일 정보 수정')
const editModalDescription = computed(() => props.editingItemType === 'folder'
  ? '이름과 색상을 변경하거나 폴더를 삭제할 수 있습니다.'
  : '이름과 색상을 변경하거나 파일을 삭제할 수 있습니다.')
const editPlaceholder = computed(() => {
  if (props.editingItemType === 'folder') return '폴더 이름 입력'
  return props.editingFileKind === 'meeting' ? '회의 파일 이름 입력' : '파일 이름 입력'
})
const editColors = computed(() => {
  if (props.editingItemType === 'folder') return props.folderColors
  return props.editingFileKind === 'meeting' ? props.meetingFileColors : props.lectureFileColors
})

const closeFolderModal = () => {
  emit('update:isFolderModalOpen', false)
  emit('update:newFolderName', '')
}

const closeFileModal = () => {
  emit('update:isFileModalOpen', false)
  emit('update:newFileName', '')
}

const closeMeetingFileModal = () => {
  emit('update:isMeetingFileModalOpen', false)
  emit('update:newFileName', '')
}

const closeEditItemModal = () => {
  emit('update:isEditItemModalOpen', false)
  emit('update:newFileName', '')
}
</script>

<template>
  <div :class="['modal-overlay', { open: isFolderModalOpen }]" @click="closeFolderModal">
    <div class="modal-card" @click.stop>
      <h2 class="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 폴더 생성</h2>
      <p class="text-[14px] text-[#8e8e93] mb-6">이름과 색상을 지정해주세요.</p>
      <input
        class="modal-input mb-5"
        placeholder="폴더 이름 입력"
        :value="newFolderName"
        @input="emit('update:newFolderName', $event.target.value)"
        type="text"
      />
      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
      <div class="color-picker-container">
        <div
          v-for="color in folderColors.length ? folderColors : ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="closeFolderModal">취소</button>
        <button class="modal-btn-primary" @click="emit('createFolder')">추가</button>
      </div>
    </div>
  </div>

  <div :class="['modal-overlay', { open: isFileModalOpen }]" @click="closeFileModal">
    <div class="modal-card" @click.stop>
      <h2 class="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 파일 생성</h2>
      <p class="text-[14px] text-[#8e8e93] mb-6">이름과 색상을 지정해주세요.</p>
      <input
        class="modal-input mb-5"
        placeholder="파일 이름 입력"
        :value="newFileName"
        @input="emit('update:newFileName', $event.target.value)"
        type="text"
      />
      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
      <div class="color-picker-container">
        <div
          v-for="color in lectureFileColors.length ? lectureFileColors : ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="closeFileModal">취소</button>
        <button class="modal-btn-primary" @click="emit('createFile')">생성</button>
      </div>
    </div>
  </div>

  <div :class="['modal-overlay', { open: isMeetingFileModalOpen }]" @click="closeMeetingFileModal">
    <div class="modal-card" @click.stop>
      <h2 class="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 회의 파일 생성</h2>
      <p class="text-[14px] text-[#8e8e93] mb-6">회의용 파일 이름과 색상을 지정해주세요.</p>
      <input
        class="modal-input mb-5"
        placeholder="회의 파일 이름 입력"
        :value="newFileName"
        @input="emit('update:newFileName', $event.target.value)"
        type="text"
      />
      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
      <div class="color-picker-container">
        <div
          v-for="color in meetingFileColors.length ? meetingFileColors : ['#ec4899', '#f97316', '#14b8a6', '#6366f1', '#0ea5e9']"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="closeMeetingFileModal">취소</button>
        <button class="modal-btn-primary" @click="emit('createMeetingFile')">생성</button>
      </div>
    </div>
  </div>

  <div :class="['modal-overlay', { open: isEditItemModalOpen }]" @click="closeEditItemModal">
    <div class="modal-card" @click.stop>
      <h2 class="text-[20px] font-bold tracking-[-0.02em] mb-1.5">{{ editModalTitle }}</h2>
      <p class="text-[14px] text-[#8e8e93] mb-6">{{ editModalDescription }}</p>
      <input
        class="modal-input mb-5"
        :placeholder="editPlaceholder"
        :value="newFileName"
        @input="emit('update:newFileName', $event.target.value)"
        type="text"
      />
      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
      <div class="color-picker-container">
        <div
          v-for="color in editColors"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="closeEditItemModal">취소</button>
        <button class="modal-btn-danger" @click="emit('deleteEditingItem')">삭제</button>
        <button class="modal-btn-primary" @click="emit('updateItem')">변경</button>
      </div>
    </div>
  </div>
</template>
