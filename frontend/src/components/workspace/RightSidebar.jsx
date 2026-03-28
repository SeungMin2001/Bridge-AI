import React, { useState, useEffect, useRef } from 'react';

export default function RightSidebar({ visible = true }) {
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const [width, setWidth] = useState(420);

  const isResizingRef = useRef(false);
  const aiWinRef = useRef(null);
  const aiBtnRef = useRef(null);

  // Outside click for AI chat
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (
        aiWinRef.current && !aiWinRef.current.contains(e.target) &&
        aiBtnRef.current && !aiBtnRef.current.contains(e.target)
      ) {
        setIsAiChatOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  // Resizer logic
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
      // 우측 패딩 12px + 리사이저 폭 고려
      const newWidth = window.innerWidth - e.clientX - 12;
      if (newWidth > 180 && newWidth < 600) {
        setWidth(newWidth);
      }
    };
    const handleMouseUp = () => {
      if (isResizingRef.current) {
        isResizingRef.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.body.classList.remove('is-resizing');
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleMouseDown = () => {
    isResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.body.classList.add('is-resizing');
  };

  if (!visible) return null;

  return (
    <>
      {/* 우측 리사이저 */}
      <div
        className="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
        id="resizer-right"
        onMouseDown={handleMouseDown}
      >
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
      </div>

      <aside className="h-full shrink-0" id="right-sidebar" style={{ width: `${width}px`, minWidth: `${width}px`, maxWidth: `${width}px` }}>
        <div className="card h-full flex flex-col p-4 pt-3.5 relative">
          {/* AI 어시스턴트 메인 뷰 */}
          <div className="flex-1 flex flex-col items-center justify-center px-2">
            <div className="w-14 h-14 rounded-2xl ai-gradient-bg flex items-center justify-center mb-6 shadow-lg">
              <span className="material-symbols-outlined text-white text-[32px]">auto_awesome</span>
            </div>
            <h3 className="text-[18px] font-bold text-[#1d1d1f] mb-8">무엇을 도와드릴까요?</h3>
            <div className="w-full flex flex-col gap-3 mb-10">
              <button className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left">
                <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">description</span>
                <span className="text-[13px] font-medium text-[#1d1d1f]">강의 노트 요약하기</span>
              </button>
              <button className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left">
                <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">quiz</span>
                <span className="text-[13px] font-medium text-[#1d1d1f]">핵심 개념 퀴즈 생성</span>
              </button>
              <button className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left">
                <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">translate</span>
                <span className="text-[13px] font-medium text-[#1d1d1f]">외국어 자료 번역</span>
              </button>
            </div>
          </div>

          {/* 하단 입력창 */}
          <div className="mt-auto">
            <div className="sidebar-search-bg rounded-[14px] px-4 py-3 flex items-center gap-3 border border-transparent focus-within:border-[#3b82f6] transition-all">
              <input
                className="bg-transparent border-none focus:ring-0 p-0 text-[13px] flex-1 text-[#1d1d1f] placeholder-[#aeaeb2]"
                placeholder="AI에게 질문하기..." type="text"
              />
              <button className="text-[#3b82f6] hover:text-blue-700 transition-colors">
                <span className="material-symbols-outlined text-[20px]">arrow_upward</span>
              </button>
            </div>
            <p className="text-[10px] text-center text-[#aeaeb2] mt-3">AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.</p>
          </div>

          {/* AI FAB 버튼 (숨겨진 상태 — file1.html에서도 hidden) */}
          <button
            ref={aiBtnRef}
            className="absolute bottom-4 right-4 w-12 h-12 bg-white rounded-full ai-btn-shadow flex items-center justify-center hover:bg-gray-50 transition-all active:scale-95 z-30 hidden"
            id="ai-assistant-btn"
            onClick={() => setIsAiChatOpen(!isAiChatOpen)}
          >
            <span className="material-symbols-outlined text-[24px] ai-gradient-icon">auto_awesome</span>
          </button>

          {/* AI Chat Window (팝업 채팅) */}
          <div
            ref={aiWinRef}
            className={`absolute bottom-[72px] right-4 w-[calc(100%-32px)] bg-[#ffffff] rounded-2xl border border-gray-100 flex flex-col overflow-hidden z-30 h-[380px] ${isAiChatOpen ? 'open' : ''}`}
            id="ai-chat-window"
          >
            <div className="px-4 py-3 flex items-center justify-between border-b border-[#f2f2f7]">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] ai-gradient-icon">auto_awesome</span>
                <span className="text-[13px] font-bold text-[#1d1d1f]">AI 어시스턴트</span>
              </div>
              <button className="text-[#aeaeb2] hover:text-[#1d1d1f]" onClick={() => setIsAiChatOpen(false)}>
                <span className="material-symbols-outlined text-[18px]">close</span>
              </button>
            </div>
            <div className="flex-1 p-4 overflow-y-auto custom-scrollbar flex flex-col gap-4 bg-[#fbfbfd]" id="sidebar-ai-messages">
              <div className="flex flex-col items-end">
                <div className="user-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal rounded-xl">
                  '컴퓨터 네트워크' 수업 내용을 요약해줘.
                </div>
              </div>
              <div className="flex flex-col items-start">
                <div className="ai-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal shadow-sm rounded-xl">
                  네, '컴퓨터 네트워크' 폴더의 내용을 분석하여 요약해 드릴게요.<br /><br />
                  1. OSI 7계층 구조<br />
                  2. TCP/IP 모델 비교<br />
                  3. 데이터 캡슐화...
                </div>
              </div>
            </div>
            <div className="p-3 bg-white border-t border-[#f2f2f7]">
              <div className="sidebar-search-bg rounded-[12px] px-3 py-2 flex items-center gap-2">
                <input
                  className="bg-transparent border-none focus:ring-0 p-0 text-[13px] flex-1 text-[#1d1d1f] placeholder-[#aeaeb2]"
                  placeholder="AI에게 질문하기..." type="text" id="sidebar-ai-input"
                />
                <button className="text-[#3b82f6]" id="sidebar-ai-send">
                  <span className="material-symbols-outlined text-[18px]">send</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
