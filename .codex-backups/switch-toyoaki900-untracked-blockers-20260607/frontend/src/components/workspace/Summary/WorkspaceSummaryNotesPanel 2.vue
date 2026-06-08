<script setup>
defineProps({
  tabAnim: { type: String, default: 'tab-slide-right' },
  summaryNotes: { type: Array, default: () => [] }
})
</script>

<template>
  <section :class="['tab-content note-canvas flex-1 flex flex-col relative overflow-hidden p-10 pt-4', tabAnim]">
    <div class="max-w-4xl mx-auto w-full h-full overflow-y-auto custom-scrollbar">
      <h1 class="text-[32px] font-heavy-heading text-[#8e8e93] mb-5">정리 노트</h1>
      <div class="flex flex-col gap-4">
        <div v-if="summaryNotes.length === 0" class="text-[16px] text-[#aeaeb2] leading-relaxed italic">아직 추가된 내용이 없습니다. 전사 내용에서 '노트에 추가'를 눌러보세요.</div>
        <div v-else v-for="note in summaryNotes" :key="note.id" class="workspace-subpanel p-5 rounded-[24px] flex flex-col gap-2 transcription-item-enter">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-[18px] text-[#2563eb]">auto_stories</span>
              <span class="text-[13px] font-bold text-[#1d1d1f]">추가된 내용</span>
            </div>
            <span class="text-[11px] font-bold text-[#6b7280]">{{ note.time }}</span>
          </div>
          <p class="text-[15px] leading-[1.6] text-[#1f2937] font-semibold">{{ note.text }}</p>
          <div class="flex items-center gap-1.5 mt-1 border-t border-slate-200 pt-3">
            <span class="material-symbols-outlined text-[14px] text-[#64748b]">link</span>
            <span class="text-[11px] font-bold text-[#64748b] uppercase tracking-wider">Source:</span>
            <span class="text-[11px] font-bold text-[#2563eb] cursor-pointer hover:underline decoration-blue-500/50 underline-offset-2">{{ note.source || 'AI 분석 결과' }}</span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
