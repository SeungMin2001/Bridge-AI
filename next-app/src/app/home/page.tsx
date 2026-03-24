"use client";

import React, { useState } from "react";
import Link from "next/link";
import ChatWindow from "@/components/ChatWindow";

export default function HomePage() {
  const [isAiOpen, setIsAiOpen] = useState(false);

  return (
    <main className="custom-scrollbar flex-1 min-w-0 h-full overflow-y-auto flex flex-col gap-5 relative bg-white border border-gray-100 rounded-2xl shadow-sm p-4">
      {/* Banner */}
      <div className="banner bg-[#373549] rounded-3xl p-8 md:p-10 flex flex-col md:flex-row items-center justify-between overflow-hidden relative">
        <div className="absolute w-[140px] h-[140px] bg-[#2d2b3e] rounded-full top-[10px] left-[40%]"></div>
        <div className="absolute w-[220px] h-[220px] bg-[#2d2b3e] rounded-full -bottom-[60px] left-[10%]"></div>

        <div className="relative z-10 w-full max-w-[480px] flex flex-col gap-3 items-start">
          <div className="text-[36px] font-[800] text-white tracking-[-0.03em] leading-[1.1]">
            하이
          </div>
          <div className="text-[15px] text-white/80 leading-[1.6] font-[500]">
            Learn fun anywhere and anytime without any time limit just through the
            application.
          </div>
          <button className="bg-white text-[#373549] border-none rounded-full px-7 py-3.5 text-[14px] font-[800] cursor-pointer hover:bg-gray-100 transition-opacity tracking-[-0.01em]">
            Get Started
          </button>
        </div>

        <div className="relative z-10 shrink-0 w-[240px] h-[140px] hidden md:flex items-end justify-center">
          <svg
            fill="none"
            height="180"
            className="absolute -bottom-5 -right-5"
            viewBox="0 0 240 180"
            width="240"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* SVG Content matching home.html banner graphic */}
            <path
              d="M40 50 h12 v-12 h8 v12 h12 v8 h-12 v12 h-8 v-12 h-12 z"
              fill="#fff"
              opacity="0.9"
              transform="rotate(-15 50 50)"
            ></path>
            <path
              d="M50 100 h20 v6 h-20 z"
              fill="#fff"
              opacity="0.9"
              transform="rotate(10 60 100)"
            ></path>
            <g transform="translate(180, 20) rotate(15)">
              <path
                d="M15 0 C6.7 0 0 6.7 0 15 C0 20.3 2.7 25 6.7 27.8 L6.7 33.3 C6.7 34.2 7.5 35 8.3 35 L21.7 35 C22.6 35 23.3 34.2 23.3 33.3 L23.3 27.8 C27.3 25 30 20.3 30 15 Z"
                fill="none"
                stroke="#fff"
                strokeWidth="2.5"
              ></path>
              <path
                d="M10 40 h10 M12 45 h6"
                stroke="#fff"
                strokeLinecap="round"
                strokeWidth="2.5"
              ></path>
              <path
                d="M15 15 v10"
                stroke="#fff"
                strokeLinecap="round"
                strokeWidth="2.5"
              ></path>
            </g>
            <path
              d="M130 90 C130 65 170 65 170 90 C170 105 160 115 150 115 C140 115 130 105 130 90 Z"
              fill="#fff"
            ></path>
            <path
              d="M125 60 C140 45 165 45 175 60 C185 75 160 80 150 70 C140 80 115 75 125 60 Z"
              fill="#2d2b3e"
            ></path>
            <circle cx="140" cy="85" fill="#2d2b3e" r="2.5"></circle>
            <circle cx="160" cy="85" fill="#2d2b3e" r="2.5"></circle>
            <path
              d="M145 95 Q150 100 155 95"
              fill="none"
              stroke="#2d2b3e"
              strokeLinecap="round"
              strokeWidth="2"
            ></path>
            <path
              d="M110 180 C110 130 190 130 190 180 Z"
              fill="#2d2b3e"
            ></path>
            <path
              d="M90 160 C110 145 125 155 135 165"
              fill="none"
              stroke="#fff"
              strokeLinecap="round"
              strokeWidth="12"
            ></path>
            <path
              d="M210 160 C190 145 175 155 165 165"
              fill="none"
              stroke="#fff"
              strokeLinecap="round"
              strokeWidth="12"
            ></path>
            <path
              d="M85 140 L150 165 L150 200 L85 175 Z"
              fill="#fff"
              stroke="#2d2b3e"
              strokeLinejoin="round"
              strokeWidth="2"
            ></path>
            <path
              d="M215 140 L150 165 L150 200 L215 175 Z"
              fill="#f4f4f5"
              stroke="#2d2b3e"
              strokeLinejoin="round"
              strokeWidth="2"
            ></path>
            <path
              d="M95 150 L140 168 M95 158 L140 176"
              stroke="#2d2b3e"
              strokeLinecap="round"
              strokeWidth="2"
            ></path>
            <path
              d="M205 150 L160 168 M205 158 L160 176"
              stroke="#2d2b3e"
              strokeLinecap="round"
              strokeWidth="2"
            ></path>
          </svg>
        </div>
      </div>

      {/* Grid Content */}
      <div className="shrink-0 mt-4">
        <div className="flex items-center justify-end mb-4">
          <button className="btn-ghost-icon text-[14px] font-[700] text-[#3b82f6] hover:bg-transparent bg-transparent border-none cursor-pointer flex items-center gap-1 px-2 py-1 rounded-lg">
            전체보기
            <span className="material-symbols-outlined text-[16px]">
              arrow_forward
            </span>
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
          {/* Folders List */}
          {[
            {
              id: "folder-26",
              title: "26년도 폴더",
              date: "2026. 3. 9. 오후 2:27",
              starred: true,
            },
            {
              id: "folder-sqld",
              title: "SQLD",
              date: "2026. 3. 9. 오전 11:15",
              starred: false,
            },
            {
              id: "folder-25",
              title: "25년 1학기",
              date: "2026. 3. 8. 오후 6:40",
              starred: true,
            },
            {
              id: "folder-network",
              title: "컴퓨터 네트워크",
              date: "2026. 3. 7. 오후 1:12",
              starred: false,
            },
            {
              id: "folder-note",
              title: "2학기 필기폴더",
              date: "2026. 3. 5. 오전 9:45",
              starred: true,
            },
            {
              id: "folder-lecture",
              title: "2학기 강의 폴더",
              date: "2026. 3. 4. 오후 10:20",
              starred: false,
            },
          ].map((folder) => (
            <Link
              key={folder.id}
              href="/"
              className="bg-white rounded-3xl p-6 flex flex-col gap-3 cursor-pointer relative transition-all duration-200 hover:-translate-y-1 hover:shadow-[0_4px_12px_rgba(0,0,0,0.05)] aspect-square border border-gray-100"
            >
              <button className="absolute top-4 right-4 bg-transparent border-none cursor-pointer p-0 flex items-center justify-center text-[#d1d1d6] transition-colors">
                <span
                  className={`material-symbols-outlined text-[20px] ${
                    folder.starred ? "text-[#fbbf24]" : ""
                  }`}
                  style={{ fontVariationSettings: folder.starred ? '"FILL" 1' : '"FILL" 0' }}
                >
                  star
                </span>
              </button>
              <span
                className="material-symbols-outlined text-[48px] text-[#3b82f6] opacity-90"
                style={{ fontVariationSettings: '"FILL" 1' }}
              >
                folder
              </span>
              <div className="mt-auto">
                <div className="text-[15px] font-[800] text-[#1d1d1f] tracking-[-0.01em] leading-[1.3]">
                  {folder.title}
                </div>
                <div className="text-[12px] text-[#aeaeb2] font-[500] mt-1">
                  {folder.date}
                </div>
              </div>
            </Link>
          ))}
          {/* File Card */}
          <Link
            href="/"
            className="bg-white rounded-3xl p-6 flex flex-col gap-3 cursor-pointer relative transition-all duration-200 hover:-translate-y-1 hover:shadow-[0_4px_12px_rgba(0,0,0,0.05)] aspect-square border border-gray-100"
          >
            <button className="absolute top-4 right-4 bg-transparent border-none cursor-pointer p-0 flex items-center justify-center text-[#d1d1d6] transition-colors">
              <span
                className="material-symbols-outlined text-[20px]"
                style={{ fontVariationSettings: '"FILL" 0' }}
              >
                star
              </span>
            </button>
            <span
              className="material-symbols-outlined text-[48px] text-[#1d1d1f] opacity-90"
              style={{ fontVariationSettings: '"FILL" 0' }}
            >
              description
            </span>
            <div className="mt-auto">
              <div className="text-[15px] font-[800] text-[#1d1d1f] tracking-[-0.01em] leading-[1.3]">
                주간 회의록.docx
              </div>
              <div className="text-[12px] text-[#aeaeb2] font-[500] mt-1">
                2026. 3. 10. 오전 10:15
              </div>
            </div>
          </Link>
        </div>
      </div>

      <div className="h-[140px] shrink-0"></div>

      {/* Floating Action Buttons */}
      <div className="fixed bottom-8 left-1/2 -translate-x-[50%] md:translate-x-[0%] md:left-[calc(50%-60px)] flex gap-3 z-50">
        <button className="flex items-center gap-2.5 px-7 py-3.5 rounded-full bg-[#1d1d1f] text-white text-[14px] font-[700] cursor-pointer transition-all duration-200 hover:opacity-90 hover:-translate-y-0.5 border-none shadow-none">
          <span className="material-symbols-outlined text-[20px]">
            create_new_folder
          </span>
          <span>새 폴더 생성</span>
        </button>
        <button className="flex items-center gap-2.5 px-7 py-3.5 rounded-full bg-[#1d1d1f] text-white text-[14px] font-[700] cursor-pointer transition-all duration-200 hover:opacity-90 hover:-translate-y-0.5 border-none shadow-none">
          <span className="material-symbols-outlined text-[20px]">
            description
          </span>
          <span>새 파일 생성</span>
        </button>
      </div>

      {/* AI Chat Logic */}
      <div
        className="fixed bottom-8 right-8 w-14 h-14 rounded-2xl bg-white text-[#373549] flex items-center justify-center cursor-pointer z-[60] shadow-[0_8px_24px_rgba(0,0,0,0.12)] transition-all duration-200 hover:scale-105 hover:bg-[#f9f9fb] border border-black/5"
        onClick={() => setIsAiOpen(!isAiOpen)}
      >
        <span
          className="material-symbols-outlined text-[28px]"
          style={{ fontVariationSettings: '"FILL" 1' }}
        >
          auto_awesome
        </span>
      </div>

      {/* ChatWindow */}
      <ChatWindow isOpen={isAiOpen} onClose={() => setIsAiOpen(false)} />
    </main>
  );
}
