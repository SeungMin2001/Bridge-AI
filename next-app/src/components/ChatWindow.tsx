"use client";

import React from "react";

interface ChatWindowProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ChatWindow({ isOpen, onClose }: ChatWindowProps) {
  return (
    <>
      {/* AI Chat Window */}
      <div
        className={`absolute bottom-[72px] right-4 w-[calc(100%-32px)] bg-[#ffffff] rounded-2xl border border-gray-100 flex flex-col overflow-hidden z-30 h-[380px] transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] ${
          isOpen
            ? "translate-y-0 opacity-100 pointer-events-auto"
            : "translate-y-5 opacity-0 pointer-events-none"
        }`}
        id="ai-chat-window"
      >
        <div className="px-4 py-3 flex items-center justify-between border-b border-[#f2f2f7]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px] ai-gradient-icon">
              auto_awesome
            </span>
            <span className="text-[13px] font-bold text-[#1d1d1f]">
              AI 어시스턴트
            </span>
          </div>
          <button
            className="text-[#aeaeb2] hover:text-[#1d1d1f]"
            onClick={onClose}
          >
            <span className="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>
        <div className="flex-1 p-4 overflow-y-auto custom-scrollbar flex flex-col gap-4 bg-[#fbfbfd]">
          <div className="flex flex-col items-end">
            <div className="user-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal">
              '컴퓨터 네트워크' 수업 내용을 요약해줘.
            </div>
          </div>
          <div className="flex flex-col items-start">
            <div className="ai-bubble px-3 py-2 text-[13px] max-w-[85%] leading-normal shadow-sm">
              네, '컴퓨터 네트워크' 폴더의 내용을 분석하여 요약해 드릴게요.<br />
              <br />
              1. OSI 7계층 구조<br />
              2. TCP/IP 모델 비교<br />
              3. 데이터 캡슐화...
            </div>
          </div>
        </div>
        <div className="p-3 bg-white border-t border-[#f2f2f7]">
          <div className="sidebar-search-bg rounded-[12px] px-3 py-2 flex items-center gap-2">
            <input
              className="bg-transparent border-none focus:ring-0 p-0 text-[13px] flex-1 text-[#1d1d1f] placeholder-[#aeaeb2] outline-none"
              placeholder="AI에게 질문하기..."
              type="text"
            />
            <button className="text-[#3b82f6]">
              <span className="material-symbols-outlined text-[18px]">
                send
              </span>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
