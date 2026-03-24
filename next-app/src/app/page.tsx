"use client";

import React, { useState, useEffect } from "react";
import ChatWindow from "@/components/ChatWindow";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

export default function WorkspacePage() {
  const [activeTab, setActiveTab] = useState("note");
  const [isAiOpen, setIsAiOpen] = useState(false);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  const [summarySubTab, setSummarySubTab] = useState("ai-summary");

  const { isRecording, recordingSeconds, transcriptions, startRecording, stopRecording } = useAudioRecorder();

  const formatTime = (totalSeconds: number) => {
    const min = Math.floor(totalSeconds / 60);
    const sec = totalSeconds % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <main className="flex-1 flex gap-[12px] h-full min-w-0 flex-row">
      <div className="flex-1 flex flex-col gap-[12px] h-full min-w-0">
        {/* Header Ribbon */}
        <header className="card h-[56px] flex items-center px-5 shrink-0">
          <nav className="flex gap-1 overflow-x-auto no-scrollbar">
            {[
              { id: "note", label: "새 노트" },
              { id: "ai", label: "AI" },
              { id: "material", label: "자료" },
              { id: "summary", label: "요약" },
              { id: "quiz", label: "퀴즈" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-btn px-4 py-1.5 rounded-[10px] text-[13px] whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? "active-tab font-bold"
                    : "text-[#8e8e93] hover:text-[#1d1d1f] font-medium"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1.5 shrink-0">
            <button className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
              <span className="material-symbols-outlined text-[20px]">
                play_circle
              </span>
            </button>
            {!isRecording ? (
              <button
                className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]"
                onClick={startRecording}
              >
                <span className="material-symbols-outlined text-[20px]">mic</span>
              </button>
            ) : (
              <div
                className="flex items-center gap-2 bg-[#FFF0F3] hover:bg-[#FFE4E9] px-3 py-1.5 rounded-full cursor-pointer transition-colors border border-[#FFD1DA] shrink-0"
                onClick={stopRecording}
              >
                <div className="w-6 h-6 flex items-center justify-center shrink-0">
                  <span className="live-dot"></span>
                </div>
                <span className="text-[13px] font-bold text-[#1d1d1f] tabular-nums">
                  {formatTime(recordingSeconds)}
                </span>
              </div>
            )}
            <button className="btn-ghost-icon p-2 rounded-lg text-[#8e8e93]">
              <span className="material-symbols-outlined text-[20px]">share</span>
            </button>
          </div>
        </header>

        {/* Tab Contents */}
        <div id="tab-contents-container" className="flex-1 flex flex-col relative min-h-0 min-w-0">
          {activeTab === "note" && (
            <section className="tab-content card flex-1 flex flex-col relative overflow-hidden bg-white p-10 pt-12">
              <div className="max-w-4xl mx-auto w-full h-full">
                <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">
                  새 노트
                </h1>
                <div className="text-[16px] text-[#aeaeb2] leading-relaxed relative">
                  <textarea 
                    className="w-full h-[500px] border-none resize-none outline-none font-inherit text-[#1d1d1f]"
                    placeholder="여기에 타이핑을 시작하거나 파일을 업로드하세요."
                  />
                </div>
              </div>
              <div className="floating-toolbar absolute bottom-8 left-1/2 -translate-x-1/2 flex p-1.5 gap-1 z-10 bg-white shadow-sm">
                <button className="tool-btn-active w-[48px] h-[48px] flex items-center justify-center rounded-full">
                  <span className="material-symbols-outlined text-[24px]">
                    near_me
                  </span>
                </button>
                <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
                  <span className="material-symbols-outlined text-[24px]">
                    ink_pen
                  </span>
                </button>
                <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
                  <span className="material-symbols-outlined text-[24px]">
                    history_edu
                  </span>
                </button>
                <button className="w-[48px] h-[48px] flex items-center justify-center rounded-full text-[#8e8e93] hover:bg-gray-100 transition-colors">
                  <span className="material-symbols-outlined text-[24px]">
                    add_circle
                  </span>
                </button>
              </div>
            </section>
          )}

          {activeTab === "ai" && (
            <section className="tab-content card main-card-enhanced flex-1 flex flex-col relative overflow-hidden bg-white">
              <div className="flex-1 flex flex-col items-center justify-center text-center px-10 gap-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#3b82f6] to-[#a855f7] flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform duration-300 cursor-pointer">
                  <span className="material-symbols-outlined text-white text-[32px]">
                    auto_awesome
                  </span>
                </div>
                <h2 className="text-[24px] font-heavy-heading text-[#1d1d1f]">
                  무엇을 도와드릴까요?
                </h2>
                <div className="flex flex-wrap items-center justify-center gap-3 mt-2 max-w-2xl">
                  {["강의 노트 요약하기", "핵심 개념 퀴즈 생성", "외국어 자료 번역"].map(
                    (action, i) => (
                      <button
                        key={i}
                        className="px-4 py-2.5 rounded-xl bg-white border border-[#e5e5ea] text-[14px] text-[#3a3a3c] hover:border-[#3b82f6] hover:text-[#3b82f6] shadow-sm transition-all flex items-center gap-2"
                      >
                        <span className="material-symbols-outlined text-[18px]">
                          {i === 0 ? "edit_document" : i === 1 ? "quiz" : "translate"}
                        </span>
                        {action}
                      </button>
                    )
                  )}
                </div>
              </div>
              <div className="p-6 pt-0 w-full relative z-10">
                <div className="max-w-4xl mx-auto flex flex-col gap-3">
                  <div className="chat-input-glow rounded-[24px] px-3 py-3 flex flex-col gap-2">
                    <textarea
                      className="w-full bg-transparent border-none focus:ring-0 text-[15px] text-[#1d1d1f] placeholder-[#aeaeb2] resize-none px-3 pt-2 pb-1 min-h-[44px] leading-relaxed outline-none"
                      placeholder="여기에 프롬프트를 입력하거나 파일을 업로드하세요..."
                      rows={1}
                    ></textarea>
                    <div className="flex items-center justify-between px-1">
                      <div className="flex items-center gap-1">
                        <button className="btn-ghost-icon w-9 h-9 rounded-full flex items-center justify-center text-[#8e8e93] hover:text-[#1d1d1f] hover:bg-[#f2f2f7] transition-colors">
                          <span className="material-symbols-outlined text-[20px]">
                            attach_file
                          </span>
                        </button>
                        <button className="btn-ghost-icon w-9 h-9 rounded-full flex items-center justify-center text-[#8e8e93] hover:text-[#1d1d1f] hover:bg-[#f2f2f7] transition-colors">
                          <span className="material-symbols-outlined text-[20px]">
                            mic
                          </span>
                        </button>
                      </div>
                      <div className="flex items-center gap-3">
                        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#f2f2f7] hover:bg-[#e5e5ea] transition-colors text-[12px] font-medium text-[#3a3a3c]">
                          <span className="material-symbols-outlined text-[16px]">
                            library_books
                          </span>
                          소스 0개
                        </button>
                        <button className="w-[36px] h-[36px] bg-[#1d1d1f] rounded-full flex items-center justify-center text-white hover:bg-[#3a3a3c] transition-all shadow-md transform active:scale-95">
                          <span className="material-symbols-outlined text-[18px]">
                            arrow_upward
                          </span>
                        </button>
                      </div>
                    </div>
                  </div>
                  <p className="text-center text-[12px] text-[#aeaeb2]">
                    AI는 실수를 할 수 있으므로 중요한 정보는 확인해 주세요.
                  </p>
                </div>
              </div>
            </section>
          )}

          {activeTab === "material" && (
            <section className="tab-content card flex-1 flex flex-col relative overflow-hidden bg-white p-10 pt-12">
              <div className="max-w-4xl mx-auto w-full h-full">
                <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">
                  자료
                </h1>
                <div className="text-[16px] text-[#aeaeb2] leading-relaxed">
                  학습 자료 및 관련 문서가 여기에 표시됩니다.
                </div>
              </div>
            </section>
          )}

          {activeTab === "summary" && (
            <section className="tab-content card flex-1 flex flex-col relative overflow-hidden bg-white p-10 pt-[32px] overflow-y-auto custom-scrollbar">
              <div className="max-w-4xl mx-auto w-full">
                <div className="flex items-center justify-between border-b border-[#e5e5ea] mb-8 pb-0">
                  <nav className="flex gap-8">
                    <div
                      className="relative cursor-pointer"
                      onClick={() => setSummarySubTab("ai-summary")}
                    >
                      <button
                        className={`text-[15px] py-3 transition-colors ${
                          summarySubTab === "ai-summary"
                            ? "text-[#1d1d1f] font-bold"
                            : "text-[#8e8e93] font-medium hover:text-[#1d1d1f]"
                        }`}
                      >
                        AI 요약&nbsp;&nbsp;
                      </button>
                      {summarySubTab === "ai-summary" && (
                        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#1d1d1f]"></div>
                      )}
                    </div>
                    <div
                      className="relative cursor-pointer"
                      onClick={() => setSummarySubTab("history")}
                    >
                      <button
                        className={`text-[15px] py-3 transition-colors ${
                          summarySubTab === "history"
                            ? "text-[#1d1d1f] font-bold"
                            : "text-[#8e8e93] font-medium hover:text-[#1d1d1f]"
                        }`}
                      >
                        대화기록&nbsp;&nbsp;
                      </button>
                      {summarySubTab === "history" && (
                        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#1d1d1f]"></div>
                      )}
                    </div>
                  </nav>
                </div>
                <div className="space-y-10">
                  {/* Content goes here based on subtab */}
                </div>
              </div>
            </section>
          )}

          {activeTab === "quiz" && (
            <section className="tab-content card flex-1 flex flex-col relative overflow-hidden bg-white p-10 pt-12">
              <div className="max-w-4xl mx-auto w-full h-full">
                <h1 className="text-[32px] font-heavy-heading text-[#d1d1d6] mb-8">
                  퀴즈
                </h1>
                <div className="text-[16px] text-[#aeaeb2] leading-relaxed">
                  생성된 퀴즈와 테스트가 여기에 표시됩니다.
                </div>
              </div>
            </section>
          )}
        </div>
      </div>

      <div className="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20">
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
      </div>

      {/* Right Sidebar - Transcriptions */}
      {isRightSidebarOpen && (
        <aside className="w-[300px] h-full shrink-0">
          <div className="card h-full flex flex-col p-4 pt-3.5 relative">
            <div className="flex items-center gap-2 mb-5">
              <div className="sidebar-search-bg flex-1 rounded-[12px] px-3 py-1.5 flex items-center gap-2">
                <span className="material-symbols-outlined text-[#8e8e93] text-[16px]">
                  search
                </span>
                <input
                  className="bg-transparent border-none focus:ring-0 p-0 text-[12px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full outline-none"
                  placeholder="전사 내용 검색"
                  type="text"
                />
              </div>
              <button
                className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]"
                onClick={() => setIsRightSidebarOpen(false)}
              >
                <span className="material-symbols-outlined text-[20px]">
                  view_sidebar
                </span>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6 pr-1">
              {transcriptions.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center h-full text-center px-6 opacity-40">
                  <div className="w-16 h-16 rounded-full bg-[#f2f2f7] flex items-center justify-center mb-4">
                    <span className="material-symbols-outlined text-[32px] text-[#8e8e93]">
                      mic_none
                    </span>
                  </div>
                  <p className="text-[13px] text-[#1d1d1f] font-medium leading-relaxed">
                    음성 녹음을 시작하면<br />
                    실시간 전사가 여기에 표시됩니다
                  </p>
                </div>
              ) : (
                transcriptions.map((t) => (
                  <div key={t.id} className="flex flex-col gap-1.5 mt-2">
                    <span className="text-[11px] font-bold text-[#aeaeb2] px-1.5">
                      {t.timestamp.toLocaleTimeString("ko-KR", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
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

            <button
              className="absolute bottom-4 right-4 w-12 h-12 bg-white rounded-full ai-btn-shadow flex items-center justify-center hover:bg-gray-50 transition-all active:scale-95 z-30"
              onClick={() => setIsAiOpen(!isAiOpen)}
            >
              <span className="material-symbols-outlined text-[24px] ai-gradient-icon">
                auto_awesome
              </span>
            </button>

            <ChatWindow isOpen={isAiOpen} onClose={() => setIsAiOpen(false)} />
          </div>
        </aside>
      )}
    </main>
  );
}
