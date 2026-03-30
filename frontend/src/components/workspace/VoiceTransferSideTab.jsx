import React, { useState } from 'react';

// ── WordPopover (단어 클릭 시 설명 팝오버) ──
const WORD_EXPLANATIONS = {
  "기초": { desc: "어떤 지식이나 기술 따위의 바탕이 되는 토대입니다.", source: "강의 교안 Chapter 1" },
  "네트워크": { desc: "여러 대의 컴퓨터나 통신기기를 통신망으로 연결하여 데이터를 주고받는 가상의 연결 체계입니다.", source: "IT 용어 대사전" },
  "OSI": { desc: "Open Systems Interconnection의 약자로, 국제표준화기구(ISO)에서 제정한 네트워크 통신 계층 모델입니다.", source: "네트워크 개론 p.42" },
  "전사": { desc: "음성이나 말소리를 텍스트 형태의 글자로 옮겨 적는 작업을 의미합니다.", source: "언어학 입문" },
  "백엔드": { desc: "사용자의 눈에 보이지 않는 서버 측의 로직, 데이터베이스 관리, API 등을 처리하는 영역입니다.", source: "풀스택 개발 가이드" },
  "데이터": { desc: "컴퓨터가 처리할 수 있는 문자, 숫자, 소리, 그림 따위의 가공되지 않은 정보의 단위입니다.", source: "데이터 정보학" },
  "테스트": { desc: "어떤 사물이나 기능이 정해진 목적에 잘 맞는지 확인하고 검사하는 과정입니다.", source: "소프트웨어 공학" },
  "샘플": { desc: "실제 제품이나 서비스의 상태를 미리 보여주기 위해 예본으로 만든 표본입니다.", source: "UI/UX 디자인 시스템" },
  "실시간": { desc: "데이터가 발생하는 즉시 또는 아주 짧은 지연 시간 내에 처리되는 방식을 의미합니다.", source: "운영체제론" },
};

function WordPopover({ visible, x, y, word, onClose, onAddToNote, onAskAi }) {
  if (!visible) return null;

  const data = WORD_EXPLANATIONS[word.replace(/[.,]/g, '')] || {
    desc: "해당 단어에 대한 상세 설명 정보가 아직 등록되지 않았습니다. AI를 사용하여 자동으로 검색하거나 노트를 추가할 수 있습니다.",
    source: "AI 분석 결과"
  };

  return (
    <div
      className="fixed z-[10000] bg-white/80 backdrop-blur-md rounded-[20px] p-5 shadow-[0_20px_50px_rgba(0,0,0,0.1)] border border-white/40 flex flex-col gap-3 min-w-[240px] max-w-[280px] transition-all animate-in fade-in zoom-in duration-200"
      style={{ left: x + 'px', top: y + 'px' }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></div>
          <span className="text-[15px] font-extrabold text-[#1d1d1f] tracking-tight">{word}</span>
        </div>
        <button onClick={onClose} className="p-1 rounded-full hover:bg-black/5 transition-colors">
          <span className="material-symbols-outlined text-[18px] text-[#8e8e93]">close</span>
        </button>
      </div>
      <div className="text-[13px] text-[#3a3a3c] leading-[1.6] font-medium tracking-tight">
        {data.desc}
      </div>

      {/* 근거 링크 섹션 */}
      <div className="flex items-center gap-1.5 mt-1 border-t border-black/5 pt-3">
        <span className="material-symbols-outlined text-[14px] text-[#8e8e93]">link</span>
        <span className="text-[11px] font-bold text-[#8e8e93] uppercase tracking-wider">Source:</span>
        <span className="text-[11px] font-bold text-blue-500 cursor-pointer hover:underline decoration-blue-500/50 underline-offset-2">{data.source}</span>
      </div>

      <div className="flex gap-2 mt-1">
        <button 
          className="flex-1 bg-blue-500 text-white border-none py-2 rounded-xl text-[12px] font-bold hover:bg-blue-600 transition-colors shadow-sm"
          onClick={() => { onAskAi(word); onClose(); }}
        >
          AI에게 질문
        </button>
        <button 
          className="flex-1 bg-[#f2f2f7] text-[#1d1d1f] border-none py-2 rounded-xl text-[12px] font-bold hover:bg-[#e5e5ea] transition-colors"
          onClick={() => { onAddToNote(data.desc, data.source); onClose(); }}
        >
          노트에 추가
        </button>
      </div>
    </div>
  );
}

export default function VoiceTransferSideTab({ 
  transcriptions = [],
  onAddToNote,
  onAskAi
}) {
  const [transSearch, setTransSearch] = useState('');
  const [wordPopover, setWordPopover] = useState({ visible: false, x: 0, y: 0, word: '' });

  const filteredTrans = transcriptions.filter(t => t.text.toLowerCase().includes(transSearch.toLowerCase()));

  return (
    <>
      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="sidebar-search-bg rounded-[14px] px-4 py-2 flex items-center gap-2.5 mb-6">
          <span className="material-symbols-outlined text-[#8e8e93] text-[20px]">search</span>
          <input
            className="bg-transparent border-none focus:ring-0 p-0 text-[14px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
            placeholder="전사 내용 검색"
            type="text"
            value={transSearch}
            onChange={(e) => setTransSearch(e.target.value)}
          />
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-4 pb-4">
          {filteredTrans.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full opacity-40 py-10">
              <span className="material-symbols-outlined text-[48px] mb-2 text-[#aeaeb2]">record_voice_over</span>
              <p className="text-[13px] font-medium text-[#8e8e93]">전사된 데이터가 없습니다.</p>
            </div>
          ) : (
            filteredTrans.map((t, idx) => (
              <div key={idx} className="flex flex-col gap-1.5 mt-2">
                <span className="text-[11px] font-bold text-[#aeaeb2] px-1.5">{t.time}</span>
                <div className="flex items-center gap-2 px-1.5 mb-1">
                  <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                    <span className="text-[10px] font-bold text-blue-600">나</span>
                  </div>
                  <span className="text-[11px] font-bold text-[#1d1d1f]">나</span>
                </div>
                <div className="message-bubble px-3.5 py-3 text-[13px] leading-[1.6]">
                  {t.text.split(' ').map((word, wIdx) => (
                    <span
                      key={wIdx}
                      className="clickable-word"
                      onClick={(e) => {
                        // 부모 요소로의 이벤트 전파를 막아 중복 처리를 방지
                        e.stopPropagation();
                        // 클릭된 단어(span)의 화면상 위치 정보를 가져옴
                        const rect = e.currentTarget.getBoundingClientRect();
                        // 해당 단어가 포함된 메시지 말풍선 요소의 위치 정보를 가져옴
                        const bubbleRect = e.currentTarget.closest('.message-bubble').getBoundingClientRect();
                        
                        // 단어 설명 팝오버(WordPopover) 상태를 활성화하고 위치 및 단어 설정
                        setWordPopover({
                          visible: true,
                          x: bubbleRect.right + 10, // 말풍선 오른쪽에 10px 간격을 두고 배치
                          y: rect.top - 20,         // 클릭된 단어의 상단 높이에 맞춰 배치
                          word: word                // 클릭된 단어 전달
                        });
                      }}
                    >
                      {word}{' '}
                    </span>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <WordPopover
        visible={wordPopover.visible}
        x={wordPopover.x}
        y={wordPopover.y}
        word={wordPopover.word}
        onClose={() => setWordPopover({ ...wordPopover, visible: false })}
        onAddToNote={onAddToNote}
        onAskAi={onAskAi}
      />

      {wordPopover.visible && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
          onClick={() => setWordPopover({ ...wordPopover, visible: false })}
        />
      )}
    </>
  );
}
