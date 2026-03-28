import React from 'react';

export default function AiAssistant({
  isOpen,
  setIsOpen,
  aiWinRef,
  aiBtnRef
}) {
  return (
    <>
      <div className="home-ai-fab z-50" title="Ask AI Assistant" ref={aiBtnRef} onClick={() => setIsOpen(!isOpen)}>
        <span className="material-symbols-outlined text-[28px]" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
      </div>

      <div ref={aiWinRef} className={`home-ai-chat-window z-[60] ${isOpen ? 'open' : ''}`}>
        <div className="chat-header">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-[#373549] rounded-lg flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-[18px]">auto_awesome</span>
            </div>
            <span className="text-[16px] font-bold tracking-[-0.01em]">Lecto AI Assistant</span>
          </div>
          <button className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]" onClick={() => setIsOpen(false)}>
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>
        <div className="chat-content custom-scrollbar">
          <div className="chat-bubble bubble-ai">
            안녕하세요! 어떤 것을 도와드릴까요? <br />강의 노트 요약이나 시험 문제 생성 등을 도와드릴 수 있습니다.
          </div>
          <div className="chat-bubble bubble-user">
            지난 '컴퓨터 네트워크' 수업 내용을 요약해줘.
          </div>
          <div className="chat-bubble bubble-ai">
            네, '컴퓨터 네트워크' 폴더의 최신 노트를 분석하여 요약해 드릴게요. <br /><br />
            1. OSI 7계층의 구조와 각 계층의 역할<br />
            2. TCP/IP 4계층 모델과의 차이점<br />
            3. 데이터 캡슐화와 비캡슐화 과정...
          </div>
        </div>
        <div className="chat-footer">
          <div className="chat-input-container">
            <input className="chat-input" placeholder="AI에게 질문해보세요..." type="text" />
            <button className="btn-ghost-icon p-1 text-[#373549]">
              <span className="material-symbols-outlined text-[20px]">send</span>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
