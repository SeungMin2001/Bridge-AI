<!-- 폴더 생성 및 파일 추가 등 홈 화면에서 사용하는 각종 모달 창을 관리하는 컴포넌트입니다. -->
<script setup>
defineProps({
  isFolderModalOpen: Boolean,
  isFileModalOpen: Boolean,
  newFolderName: String,
  newFileName: String,
  selectedColor: String
})

const emit = defineEmits([
  'update:isFolderModalOpen',
  'update:isFileModalOpen',
  'update:newFolderName',
  'update:newFileName',
  'update:selectedColor',
  'createFolder',
  'createFile'
])
</script>

<template>
  <!-- Folder Creation Modal -->
  <div :class="['modal-overlay', { open: isFolderModalOpen }]" @click="emit('update:isFolderModalOpen', false)">
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
          v-for="color in ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="emit('update:isFolderModalOpen', false); emit('update:newFolderName', '')">취소</button>
        <button class="modal-btn-primary" @click="emit('createFolder')">추가</button>
      </div>
    </div>
  </div>

  <!-- File Creation Modal -->
  <div :class="['modal-overlay', { open: isFileModalOpen }]" @click="emit('update:isFileModalOpen', false)">
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
          v-for="color in ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']"
          :key="color"
          :class="['color-circle', { selected: selectedColor === color }]"
          :style="{ backgroundColor: color }"
          @click="emit('update:selectedColor', color)"
        ></div>
      </div>
      <div class="flex justify-end gap-3 mt-4">
        <button class="modal-btn-secondary" @click="emit('update:isFileModalOpen', false); emit('update:newFileName', '')">취소</button>
        <button class="modal-btn-primary" @click="emit('createFile')">생성</button>
      </div>
    </div>
  </div>
</template>
