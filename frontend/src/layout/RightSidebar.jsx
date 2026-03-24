import React, { useState, useEffect, useRef } from 'react';

export default function RightSidebar({ transcriptions = [] }) {
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const [width, setWidth] = useState(300);
  
  const isResizingRef = useRef(false);
  const scrollRef = useRef(null);
  const aiWinRef = useRef(null);
  const aiBtnRef = useRef(null);

  // Auto-scroll transcriptions (전사 내용 자동 스크롤)
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcriptions]);

  // Outside click for AI chat (AI 채팅창 외부 클릭 시 닫기)
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

  // Resizer logic (사이즈 조절 로직)
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
      const newWidth = window.innerWidth - e.clientX - 6;
      if (newWidth > 200 && newWidth < 600) {
        setWidth(newWidth);
      }
    };
    const handleMouseUp = () => {
      if (isResizingRef.current) {
        isResizingRef.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
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
  };

  return (
    <>
      <div
        className="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
        id="resizer-right"
        onMouseDown={handleMouseDown}
      >
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
      </div>

      <aside className="h-full shrink-0" id="right-sidebar" style={{ width: `${width}px`, minWidth: `${width}px`, maxWidth: `${width}px` }}>
        <div className="card h-full flex flex-col p-4 pt-3.5 relative">
          <div className="flex items-center gap-2 mb-5">
            <div className="sidebar-search-bg flex-1 rounded-[12px] px-3 py-1.5 flex items-center gap-2">
              <span className="material-symbols-outlined text-[#8e8e93] text-[16px]">search</span>
              <input
                className="bg-transparent border-none focus:ring-0 p-0 text-[12px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
                placeholder="전사 내용 검색" type="text"
              />
            </div>
            <button className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]">
              <span className="material-symbols-outlined text-[20px]">view_sidebar</span>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6 pr-1" id="tsList" ref={scrollRef}>
            {transcriptions.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center h-full text-center px-6 opacity-40" id="emptyState">
                <div className="w-16 h-16 rounded-full bg-[#f2f2f7] flex items-center justify-center mb-4">
                  <span className="material-symbols-outlined text-[32px] text-[#8e8e93]">mic_none</span>
                </div>
                <p className="text-[13px] text-[#1d1d1f] font-medium leading-relaxed">
                  음성 녹음을 시작하면<br />실시간 전사가 여기에 표시됩니다
                </p>
              </div>
            ) : (
              transcriptions.map((t, idx) => (
                <div key={idx} className="flex flex-col gap-1.5 mt-2">
                  <span className="text-[11px] font-bold text-[#aeaeb2] px-1.5">{t.time}</span>
                  <div className="flex items-center gap-2 px-1.5 mb-1">
                    <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                      <span className="text-[10px] font-bold text-blue-600">나</span>
                    </div>
                    <span className="text-[11px] font-bold text-[#1d1d1f]">나</span>
                  </div>
                  <div className="message-bubble px-3.5 py-3 text-[13px] leading-[1.6]">
                    {t.text}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* AI FAB (AI 플로팅 버튼) */}
          <button
            ref={aiBtnRef}
            className="absolute bottom-4 right-4 w-12 h-12 bg-white rounded-full ai-btn-shadow flex items-center justify-center hover:bg-gray-50 transition-all active:scale-95 z-30"
            id="ai-assistant-btn"
            onClick={() => setIsAiChatOpen(!isAiChatOpen)}
          >
            <span className="material-symbols-outlined text-[24px] ai-gradient-icon">auto_awesome</span>
          </button>

          {/* AI Chat Window (AI 채팅창) */}
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
              <button className="text-[#aeaeb2] hover:text-[#1d1d1f]" id="close-chat" onClick={() => setIsAiChatOpen(false)}>
                <span className="material-symbols-outlined text-[18px]">close</span>
              </button>
            </div>
            <div className="flex-1 p-4 overflow-y-auto custom-scrollbar flex flex-col gap-4 bg-[#fbfbfd]" id="sidebar-ai-messages">
              <div className="flex flex-col items-end">
                <div className="user-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal">
                  '컴퓨터 네트워크' 수업 내용을 요약해줘.
                </div>
              </div>
              <div className="flex flex-col items-start">
                <div className="ai-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal shadow-sm">
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
