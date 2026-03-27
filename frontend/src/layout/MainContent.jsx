import React, { useState } from 'react';

export default function MainContent({ isRecording, recordingTimeText, startRecording, stopRecording, activeFileName, onMainSidebarToggle, onRightSidebarToggle }) {
  const [activeTab, setActiveTab] = useState('note');
  const [activeSummaryTab, setActiveSummaryTab] = useState('ai-summary');
  const [noteContent, setNoteContent] = useState('');
  const [isNoteFocused, setIsNoteFocused] = useState(false);

  // 탭에 표시될 이름 (활성 파일이 있으면 그 이름 사용)
  const noteTabName = activeFileName || '새 노트';

  return (
    <main className="flex-1 flex flex-col gap-[12px] h-full min-w-0" style={{ flex: '1 1 0%', minWidth: '300px' }}>
      {/* Header Card */}
      <header className="card h-[56px] flex items-center px-5 shrink-0">
        {/* 좌측 사이드바 토글 */}
        <button
          className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93] mr-2 shrink-0"
          title="사이드바 토글"
          onClick={onMainSidebarToggle}
        >
          <span className="material-symbols-outlined text-[20px]">side_navigation</span>
        </button>

        {/* 탭 네비게이션 */}
        <nav className="flex gap-1 overflow-x-auto no-scrollbar" id="main-tabs">
          {[
            { key: 'note', label: noteTabName },
            { key: 'ai', label: 'AI' },
            { key: 'material', label: '자료' },
            { key: 'summary', label: '요약' },
            { key: 'quiz', label: '퀴즈' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              data-tab={tab.key}
              className={`px-4 py-1.5 rounded-[10px] text-[13px] whitespace-nowrap transition-colors ${
                activeTab === tab.key
                  ? 'active-tab font-bold'
                  : 'text-[#8e8e93] hover:text-[#1d1d1f] font-medium'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* 우측 버튼들 */}
        <div className="ml-auto flex items-center gap-1.5 shrink-0 pl-2">
          <button className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
            <span className="material-symbols-outlined text-[20px]">play_circle</span>
          </button>

          {!isRecording ? (
            <button className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]" onClick={startRecording} id="start">
              <span className="material-symbols-outlined text-[20px]">mic</span>
            </button>
          ) : (
            <div
              id="recording-timer"
              onClick={stopRecording}
              className="flex items-center gap-2 bg-[#FFF0F3] hover:bg-[#FFE4E9] px-3 py-1.5 rounded-full cursor-pointer transition-colors border border-[#FFD1DA] shrink-0"
            >
              <div className="w-6 h-6 flex items-center justify-center shrink-0"><span className="live-dot"></span></div>
              <span className="text-[13px] font-bold text-[#1d1d1f] tabular-nums" id="recording-time">{recordingTimeText}</span>
            </div>
          )}

          {/* 우측 사이드바 토글 */}
          <button
            className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]"
            title="우측 사이드바 토글"
            onClick={onRightSidebarToggle}
          >
            <span className="material-symbols-outlined text-[20px] scale-x-[-1]">side_navigation</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div id="tab-contents-container" className="flex-1 flex flex-col relative min-h-0 min-w-0">
        {/* Note Tab */}
        <section className={`tab-content card flex-1 flex flex-col relative overflow-hidden note-canvas p-10 pt-12 ${activeTab === 'note' ? 'flex' : 'hidden'}`}>
          <div className="max-w-4xl mx-auto w-full h-full">
            <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8" id="note-title">{noteTabName}</h1>
            <div
              className="text-[16px] leading-relaxed min-h-[200px] focus:outline-none"
              id="note-body"
              contentEditable={true}
              suppressContentEditableWarning={true}
              style={{ color: isNoteFocused || noteContent ? '#1d1d1f' : '#aeaeb2' }}
              onFocus={() => setIsNoteFocused(true)}
              onBlur={(e) => {
                setIsNoteFocused(false);
                setNoteContent(e.currentTarget.textContent || '');
              }}
            >
              {!noteContent && !isNoteFocused ? '여기에 타이핑을 시작하거나 파일을 업로드하세요.' : noteContent}
            </div>
          </div>

          {/* Floating Toolbar */}
          <div className="floating-toolbar absolute bottom-8 left-1/2 -translate-x-1/2 flex p-1.5 gap-1 z-10 bg-white">
            <button className="tool-btn-active w-[48px] h-[48px] flex items-center justify-center rounded-full">
              <span className="material-symbols-outlined text-[24px]">near_me</span>
            </button>
            <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
              <span className="material-symbols-outlined text-[24px]">ink_pen</span>
            </button>
            <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
              <span className="material-symbols-outlined text-[24px]">history_edu</span>
            </button>
            <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
              <span className="material-symbols-outlined text-[24px]">add_circle</span>
            </button>
          </div>
        </section>

        {/* AI Tab */}
        <section className={`tab-content card main-card-enhanced flex-1 flex flex-col relative overflow-hidden note-canvas ${activeTab === 'ai' ? 'flex' : 'hidden'}`}>
          <div className="flex-1 flex flex-col items-center justify-center text-center px-10 gap-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#3b82f6] to-[#a855f7] flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform duration-300 cursor-pointer">
              <span className="material-symbols-outlined text-white text-[32px]">auto_awesome</span>
            </div>
            <h2 className="text-[24px] font-heavy-heading text-[#1d1d1f]">무엇을 도와드릴까요?</h2>
            <div className="flex flex-wrap items-center justify-center gap-3 mt-2 max-w-2xl">
              <button className="px-4 py-2.5 rounded-xl bg-white border border-[#e5e5ea] text-[14px] text-[#3a3a3c] hover:border-[#3b82f6] hover:text-[#3b82f6] shadow-sm transition-all flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px]">edit_document</span> 강의 노트 요약하기
              </button>
              <button className="px-4 py-2.5 rounded-xl bg-white border border-[#e5e5ea] text-[14px] text-[#3a3a3c] hover:border-[#3b82f6] hover:text-[#3b82f6] shadow-sm transition-all flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px]">quiz</span> 핵심 개념 퀴즈 생성
              </button>
              <button className="px-4 py-2.5 rounded-xl bg-white border border-[#e5e5ea] text-[14px] text-[#3a3a3c] hover:border-[#3b82f6] hover:text-[#3b82f6] shadow-sm transition-all flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px]">translate</span> 외국어 자료 번역
              </button>
            </div>
          </div>
          <div className="p-6 pt-0 w-full relative z-10">
            <div className="max-w-4xl mx-auto flex flex-col gap-3">
              <div className="chat-input-glow rounded-[24px] px-3 py-3 flex flex-col gap-2">
                <textarea
                  className="prompt-textarea w-full bg-transparent border-none focus:ring-0 text-[15px] text-[#1d1d1f] placeholder-[#aeaeb2] resize-none px-3 pt-2 pb-1 min-h-[44px] leading-relaxed"
                  onInput={(e) => { e.target.style.height = 'auto'; e.target.style.height = (e.target.scrollHeight) + 'px'; }}
                  placeholder="여기에 프롬프트를 입력하거나 파일을 업로드하세요..." rows={1}
                />
                <div className="flex items-center justify-between px-1">
                  <div className="flex items-center gap-1">
                    <button className="btn-ghost-icon w-9 h-9 rounded-full flex items-center justify-center text-[#8e8e93] hover:text-[#1d1d1f] hover:bg-[#f2f2f7] transition-colors" title="파일 첨부">
                      <span className="material-symbols-outlined text-[20px]">attach_file</span>
                    </button>
                    <button className="btn-ghost-icon w-9 h-9 rounded-full flex items-center justify-center text-[#8e8e93] hover:text-[#1d1d1f] hover:bg-[#f2f2f7] transition-colors" title="음성 입력">
                      <span className="material-symbols-outlined text-[20px]">mic</span>
                    </button>
                  </div>
                  <div className="flex items-center gap-3">
                    <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#f2f2f7] hover:bg-[#e5e5ea] transition-colors text-[12px] font-medium text-[#3a3a3c]">
                      <span className="material-symbols-outlined text-[16px]">library_books</span> 소스 0개
                    </button>
                    <button className="w-[36px] h-[36px] bg-[#1d1d1f] rounded-full flex items-center justify-center text-white hover:bg-[#3a3a3c] transition-all shadow-md transform active:scale-95">
                      <span className="material-symbols-outlined text-[18px]">arrow_upward</span>
                    </button>
                  </div>
                </div>
              </div>
              <p className="text-center text-[12px] text-[#aeaeb2]">AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.</p>
            </div>
          </div>
        </section>

        {/* Material Tab */}
        <section className={`tab-content card flex-1 flex flex-col relative overflow-hidden p-10 pt-12 ${activeTab === 'material' ? 'flex' : 'hidden'}`}>
          <div className="max-w-4xl mx-auto w-full h-full">
            <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">자료</h1>
            <div className="text-[16px] text-[#aeaeb2] leading-relaxed">학습 자료 및 관련 문서가 여기에 표시됩니다.</div>
          </div>
        </section>

        {/* Summary Tab */}
        <section className={`tab-content card flex-1 flex flex-col relative overflow-hidden note-canvas p-10 overflow-y-auto custom-scrollbar pt-[32px] ${activeTab === 'summary' ? 'flex' : 'hidden'}`}>
          <div className="max-w-4xl mx-auto w-full">
            <div className="flex items-center justify-between border-b border-[#e5e5ea] mb-8 pb-0">
              <nav className="flex gap-8">
                <div className="relative cursor-pointer summary-subtab-btn group" onClick={() => setActiveSummaryTab('ai-summary')}>
                  <button className={`text-[15px] py-3 pointer-events-none transition-colors ${activeSummaryTab === 'ai-summary' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]'}`}>AI 요약&nbsp;&nbsp;</button>
                  <div className={`summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors ${activeSummaryTab === 'ai-summary' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]'}`}></div>
                </div>
                <div className="relative cursor-pointer summary-subtab-btn group" onClick={() => setActiveSummaryTab('history')}>
                  <button className={`text-[15px] py-3 pointer-events-none transition-colors ${activeSummaryTab === 'history' ? 'text-[#1d1d1f] font-bold' : 'text-[#8e8e93] font-medium group-hover:text-[#1d1d1f]'}`}>대화기록&nbsp;&nbsp;</button>
                  <div className={`summary-subtab-indicator absolute bottom-0 left-0 right-0 h-[3px] transition-colors ${activeSummaryTab === 'history' ? 'bg-[#1d1d1f]' : 'bg-transparent group-hover:bg-[#1d1d1f]'}`}></div>
                </div>
              </nav>
              <div className="flex items-center gap-6 pb-2">
                <button className="flex items-center gap-1.5 text-[#ff3b30] hover:opacity-80 transition-opacity">
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                  <span className="text-[14px] font-medium">삭제</span>
                </button>
                <button className="flex items-center gap-1.5 text-[#8e8e93] hover:text-[#1d1d1f] transition-colors">
                  <span className="material-symbols-outlined text-[20px]">content_copy</span>
                  <span className="text-[14px] font-medium">복사</span>
                </button>
              </div>
            </div>

            <div className={`summary-subcontent space-y-10 ${activeSummaryTab === 'ai-summary' ? 'block' : 'hidden'}`}></div>
            <div className={`summary-subcontent space-y-10 ${activeSummaryTab === 'history' ? 'block' : 'hidden'}`}></div>
          </div>
        </section>

        {/* Quiz Tab */}
        <section className={`tab-content card flex-1 flex flex-col relative overflow-hidden p-10 pt-12 ${activeTab === 'quiz' ? 'flex' : 'hidden'}`}>
          <div className="max-w-4xl mx-auto w-full h-full">
            <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">퀴즈</h1>
            <div className="text-[16px] text-[#aeaeb2] leading-relaxed">생성된 퀴즈와 테스트가 여기에 표시됩니다.</div>
          </div>
        </section>
      </div>
    </main>
  );
}
