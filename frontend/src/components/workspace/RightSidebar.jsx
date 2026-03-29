import React, { useState, useEffect, useRef } from 'react';

export default function RightSidebar({ visible = true }) {
  const [width, setWidth] = useState(420);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const isResizingRef = useRef(false);
  const messagesEndRef = useRef(null);

  // 메시지 추가 시 자동 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Resizer logic
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
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

  // AI 채팅 전송
  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage = { role: 'user', content: trimmed };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed }),
      });

      if (!res.ok) {
        throw new Error(`서버 오류 (${res.status})`);
      }

      const data = await res.json();
      setMessages(prev => [
        ...prev,
        { role: 'ai', content: data.answer },
      ]);
    } catch (err) {
      console.error('AI 채팅 오류:', err);
      setMessages(prev => [
        ...prev,
        { role: 'ai', content: `⚠️ 오류가 발생했습니다: ${err.message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Enter 키 전송
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 빠른 액션 버튼 클릭
  const handleQuickAction = (text) => {
    setInput(text);
  };

  if (!visible) return null;

  const hasMessages = messages.length > 0;

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

          {/* 채팅 내역이 없을 때: 초기 화면 */}
          {!hasMessages && (
            <div className="flex-1 flex flex-col items-center justify-center px-2">
              <div className="w-14 h-14 rounded-2xl ai-gradient-bg flex items-center justify-center mb-6 shadow-lg">
                <span className="material-symbols-outlined text-white text-[32px]">auto_awesome</span>
              </div>
              <h3 className="text-[18px] font-bold text-[#1d1d1f] mb-8">무엇을 도와드릴까요?</h3>
              <div className="w-full flex flex-col gap-3 mb-10">
                <button
                  className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left"
                  onClick={() => handleQuickAction('강의 노트를 요약해줘')}
                >
                  <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">description</span>
                  <span className="text-[13px] font-medium text-[#1d1d1f]">강의 노트 요약하기</span>
                </button>
                <button
                  className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left"
                  onClick={() => handleQuickAction('핵심 개념으로 퀴즈를 만들어줘')}
                >
                  <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">quiz</span>
                  <span className="text-[13px] font-medium text-[#1d1d1f]">핵심 개념 퀴즈 생성</span>
                </button>
                <button
                  className="action-card w-full flex items-center gap-3 p-3.5 rounded-xl bg-white text-left"
                  onClick={() => handleQuickAction('이 자료를 영어로 번역해줘')}
                >
                  <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">translate</span>
                  <span className="text-[13px] font-medium text-[#1d1d1f]">외국어 자료 번역</span>
                </button>
              </div>
            </div>
          )}

          {/* 채팅 내역이 있을 때: 메시지 리스트 */}
          {hasMessages && (
            <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-3 pb-2 pr-1">
              {/* 상단 헤더 */}
              <div className="flex items-center justify-between mb-2 sticky top-0 bg-inherit z-10 pb-2">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px] ai-gradient-icon">auto_awesome</span>
                  <span className="text-[14px] font-bold text-[#1d1d1f]">AI 어시스턴트</span>
                </div>
                <button
                  className="text-[#aeaeb2] hover:text-[#1d1d1f] transition-colors"
                  onClick={() => setMessages([])}
                  title="대화 초기화"
                >
                  <span className="material-symbols-outlined text-[18px]">refresh</span>
                </button>
              </div>

              {messages.map((msg, idx) => (
                <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div
                    className={`px-3 py-2 text-[13px] max-w-[85%] leading-relaxed rounded-xl whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'user-bubble'
                        : 'ai-bubble shadow-sm'
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}

              {/* 로딩 인디케이터 */}
              {isLoading && (
                <div className="flex flex-col items-start">
                  <div className="ai-bubble px-3 py-2 text-[13px] max-w-[85%] leading-relaxed rounded-xl shadow-sm flex items-center gap-2">
                    <span className="inline-block w-1.5 h-1.5 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="inline-block w-1.5 h-1.5 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="inline-block w-1.5 h-1.5 bg-[#8e8e93] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                    <span className="text-[#aeaeb2] ml-1">생각하고 있어요...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {/* 하단 입력창 */}
          <div className="mt-auto pt-2">
            <div className="sidebar-search-bg rounded-[14px] px-4 py-3 flex items-center gap-3 border border-transparent focus-within:border-[#3b82f6] transition-all">
              <input
                className="bg-transparent border-none focus:ring-0 p-0 text-[13px] flex-1 text-[#1d1d1f] placeholder-[#aeaeb2]"
                placeholder="AI에게 질문하기..."
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
              />
              <button
                className={`transition-colors ${isLoading || !input.trim() ? 'text-[#d1d1d6] cursor-not-allowed' : 'text-[#3b82f6] hover:text-blue-700'}`}
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
              >
                <span className="material-symbols-outlined text-[20px]">arrow_upward</span>
              </button>
            </div>
            <p className="text-[10px] text-center text-[#aeaeb2] mt-3">AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.</p>
          </div>
        </div>
      </aside>
    </>
  );
}
