<!-- 폴더 생성 및 파일/폴더 수정에 사용하는 홈 화면 모달 모음입니다. -->
<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  isFolderModalOpen: Boolean,
  isFileModalOpen: Boolean,
  isEditItemModalOpen: Boolean,
  newFolderName: String,
  newFileName: String,
  selectedColor: String,
  selectedTag: { type: String, default: '수업' },
  selectedFileIcon: { type: String, default: 'article' },
  placement: { type: String, default: 'rail' },
  editingItemType: { type: String, default: 'file' },
  editingFileKind: { type: String, default: 'lecture' },
  folderColors: { type: Array, default: () => [] },
  lectureFileColors: { type: Array, default: () => [] },
  meetingFileColors: { type: Array, default: () => [] },
  fileTags: { type: Array, default: () => [] },
  fileIcons: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'update:isFolderModalOpen',
  'update:isFileModalOpen',
  'update:isEditItemModalOpen',
  'update:newFolderName',
  'update:newFileName',
  'update:selectedColor',
  'update:selectedTag',
  'update:selectedFileIcon',
  'createFolder',
  'createFile',
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
const fileColors = computed(() => props.lectureFileColors.length ? props.lectureFileColors : ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'])
const fileTags = computed(() => props.fileTags.length ? props.fileTags : ['수업', '회의', '프로젝트', '개인', '중요'])
const fileIcons = computed(() => props.fileIcons.length
  ? props.fileIcons
  : ['article', 'groups_2', 'workspaces', 'person', 'priority_high', 'star', 'task_alt', 'lightbulb', 'bookmark', 'school'])
const customTagValue = computed(() => fileTags.value.includes(props.selectedTag) ? '' : props.selectedTag)
const isCustomTagInputOpen = ref(false)
const shouldShowCustomTagInput = computed(() => isCustomTagInputOpen.value || !!customTagValue.value)

watch(() => [props.isFileModalOpen, props.isEditItemModalOpen], ([isFileOpen, isEditOpen]) => {
  if (isFileOpen || isEditOpen) {
    isCustomTagInputOpen.value = !!customTagValue.value
  }
})

const selectPresetTag = (tag) => {
  isCustomTagInputOpen.value = false
  emit('update:selectedTag', tag)
}

const openCustomTagInput = () => {
  isCustomTagInputOpen.value = !isCustomTagInputOpen.value
}

const closeFolderModal = () => {
  emit('update:isFolderModalOpen', false)
  emit('update:newFolderName', '')
}

const closeFileModal = () => {
  emit('update:isFileModalOpen', false)
  emit('update:newFileName', '')
}

const closeEditItemModal = () => {
  emit('update:isEditItemModalOpen', false)
  emit('update:newFileName', '')
}
</script>

<template>
  <div :class="['modal-overlay', `modal-overlay-${placement}`, { open: isFolderModalOpen }]" @click="closeFolderModal">
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

  <div :class="['modal-overlay', `modal-overlay-${placement}`, { open: isFileModalOpen }]" @click="closeFileModal">
    <div class="modal-card" @click.stop>
      <h2 class="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 파일 생성</h2>
      <p class="text-[14px] text-[#8e8e93] mb-6">이름, 태그와 색상을 지정해주세요.</p>
      <input
        class="modal-input mb-5"
        placeholder="파일 이름 입력"
        :value="newFileName"
        @input="emit('update:newFileName', $event.target.value)"
        type="text"
      />

      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">태그</p>
      <div class="tag-picker-container">
        <button
          v-for="tag in fileTags"
          :key="tag"
          type="button"
          :class="['tag-choice', { selected: selectedTag === tag }]"
          @click="selectPresetTag(tag)"
        >
          {{ tag }}
        </button>
        <button
          type="button"
          :class="['tag-choice', 'tag-add-choice', { selected: shouldShowCustomTagInput }]"
          @click="openCustomTagInput"
        >
          <span class="material-symbols-outlined">add</span>
        </button>
      </div>
      <input
        v-if="shouldShowCustomTagInput"
        class="modal-input modal-compact-input mb-5"
        placeholder="직접 태그 입력"
        :value="customTagValue"
        @input="emit('update:selectedTag', $event.target.value)"
        type="text"
      />

      <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
      <div class="color-picker-container">
        <div
          v-for="color in fileColors"
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

  <div :class="['modal-overlay', `modal-overlay-${placement}`, { open: isEditItemModalOpen }]" @click="closeEditItemModal">
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
      <template v-if="editingItemType === 'file'">
        <p class="text-[14px] font-bold text-[#3a3a3c] mb-3">태그</p>
        <div class="tag-picker-container">
          <button
            v-for="tag in fileTags"
            :key="tag"
            type="button"
            :class="['tag-choice', { selected: selectedTag === tag }]"
            @click="selectPresetTag(tag)"
          >
            {{ tag }}
          </button>
          <button
            type="button"
            :class="['tag-choice', 'tag-add-choice', { selected: shouldShowCustomTagInput }]"
            @click="openCustomTagInput"
          >
            <span class="material-symbols-outlined">add</span>
          </button>
        </div>
        <input
          v-if="shouldShowCustomTagInput"
          class="modal-input modal-compact-input mb-5"
          placeholder="직접 태그 입력"
          :value="customTagValue"
          @input="emit('update:selectedTag', $event.target.value)"
          type="text"
        />

      </template>
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

<style scoped>
.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 120;
  align-items: flex-start;
  justify-content: flex-start;
  background: transparent;
  pointer-events: none;
}

.modal-overlay.open {
  display: flex;
}

.modal-overlay-rail {
  padding-left: calc(var(--copy-rail-width) - 6px);
  padding-top: 142px;
}

.modal-overlay-work-top {
  justify-content: flex-end;
  padding-top: 96px;
  padding-right: 112px;
}

.modal-card {
  position: relative;
  width: min(310px, calc(100vw - 28px));
  max-height: calc(100vh - 36px);
  overflow-y: auto;
  border-radius: 24px;
  background: var(--copy-surface);
  border: 0;
  box-shadow: 0 28px 60px rgba(48, 42, 58, 0.14);
  padding: 21px 21px 19px;
  pointer-events: auto;
}

.modal-card::before {
  content: '';
  position: absolute;
  left: -10px;
  top: 38px;
  width: 22px;
  height: 22px;
  background: #fff;
  transform: rotate(45deg);
  border-radius: 3px;
}

.modal-overlay-work-top .modal-card::before {
  left: auto;
  right: 62px;
  top: -10px;
}

.modal-card h2 {
  color: #1f2937;
  font-size: 21px !important;
  line-height: 1.18;
  margin-bottom: 6px !important;
  font-weight: 950;
  letter-spacing: -0.04em;
}

.modal-card p {
  letter-spacing: -0.01em;
}

.modal-input {
  width: 100%;
  min-height: 44px;
  border: 2px solid #e2e0e8;
  border-radius: 15px;
  padding: 0 14px;
  outline: none;
  color: #1f2937;
  background: #fff;
  font-size: 14px;
  font-weight: 850;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.modal-input:focus {
  border-color: var(--copy-black);
  box-shadow: 0 0 0 3px rgba(21, 22, 26, 0.06);
}

.modal-compact-input {
  min-height: 44px;
  margin-top: -8px;
}

.tag-picker-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.tag-choice {
  min-height: 34px;
  padding: 0 13px;
  border-radius: 999px;
  border: 2px solid #e2e0e8;
  background: #fff;
  color: #73717d;
  font-size: 13px;
  font-weight: 900;
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.tag-choice:hover {
  transform: translateY(-1px);
}

.tag-choice.selected {
  color: #fff;
  border-color: var(--copy-black);
  background: var(--copy-black);
}

.tag-add-choice {
  width: 38px;
  padding: 0;
}

.icon-picker-container {
  display: grid;
  grid-template-columns: repeat(5, 38px);
  gap: 8px;
  margin-bottom: 18px;
}

.icon-choice {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  border: 2px solid #e2e0e8;
  background: #fff;
  color: #777b88;
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
}

.icon-choice:hover {
  transform: translateY(-1px);
}

.icon-choice.selected {
  color: #fff;
  border-color: var(--copy-black);
  background: var(--copy-black);
}

.icon-choice .material-symbols-outlined {
  font-size: 20px;
}

.color-picker-container {
  display: flex;
  align-items: center;
  gap: 11px;
  margin-bottom: 18px;
}

.color-circle {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  border: 3px solid #fff;
  box-shadow: 0 0 0 0 transparent;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.color-circle:hover {
  transform: translateY(-1px);
}

.color-circle.selected {
  box-shadow: 0 0 0 2px var(--copy-black);
}

.modal-btn-primary,
.modal-btn-secondary,
.modal-btn-danger {
  min-width: 72px;
  min-height: 38px;
  border-radius: 999px;
  border: 0;
  padding: 0 16px;
  font-size: 13px;
  font-weight: 950;
  transition: transform 0.18s ease, opacity 0.18s ease, background 0.18s ease;
}

.modal-btn-primary {
  background: var(--copy-black);
  color: #fff;
}

.modal-btn-secondary {
  background: #efedf4;
  color: var(--copy-text);
}

.modal-btn-danger {
  background: #fff0f0;
  color: #f04444;
}

.modal-btn-primary:hover,
.modal-btn-secondary:hover,
.modal-btn-danger:hover {
  transform: translateY(-1px);
}
</style>
