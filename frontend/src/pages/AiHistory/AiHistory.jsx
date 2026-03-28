import React from 'react';

export default function AiHistory({ onNavigateBack }) {
  const historyData = [
    {
      id: 1,
      title: "컴퓨터 네트워크 요약",
      date: "2026. 03. 27. 오후 02:15",
      prompt: "컴퓨터 네트워크 폴더의 최신 노트를 분석해서 OSI 7계층 핵심만 요약해줘.",
      aiResponse: "네, OSI 7계층 핵심 요약입니다: 1. 물리... 2. 데이터링크... 7. 응용 계층..."
    },
    {
      id: 2,
      title: "시험 문제 생성 (네트워크 보안)",
      date: "2026. 03. 26. 오전 11:30",
      prompt: "네트워크 보안 파트에서 나올법한 문제 5개만 만들어줘.",
      aiResponse: "생성된 문항입니다: 1. 대칭 키 암호화의 단점은? 2. SSL/TLS의 차이는?..."
    },
    {
      id: 3,
      title: "SQL 기초 정리",
      date: "2026. 03. 25. 오후 05:40",
      prompt: "SQL SELECT문 기본 문법이랑 예시 하나 알려줘.",
      aiResponse: "SELECT * FROM users WHERE age > 20; 와 같이 사용합니다..."
    }
  ];

  return (
    <div className="p-[12px] h-full flex flex-col bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
      <div className="bg-white rounded-[24px] p-8 flex flex-col h-full shadow-sm overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <button 
              onClick={onNavigateBack}
              className="p-2.5 rounded-xl hover:bg-[#f2f2f7] transition-colors"
            >
              <span className="material-symbols-outlined text-[24px]">arrow_back</span>
            </button>
            <h1 className="text-[24px] font-extrabold tracking-[-0.03em]">AI 명령 기록</h1>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2.5 rounded-xl hover:bg-[#f2f2f7] transition-colors">
              <span className="material-symbols-outlined text-[22px]">search</span>
            </button>
            <button className="p-2.5 rounded-xl hover:bg-[#f2f2f7] transition-colors">
              <span className="material-symbols-outlined text-[22px]">filter_list</span>
            </button>
          </div>
        </div>

        {/* History List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
          <div className="flex flex-col gap-4">
            {historyData.map(item => (
              <div 
                key={item.id}
                className="p-6 rounded-[20px] bg-[#f9f9fb] border border-[#e5e5ea] hover:shadow-md transition-all cursor-pointer group"
              >
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="w-10 h-10 bg-[#373549] rounded-xl flex items-center justify-center">
                      <span className="material-symbols-outlined text-white text-[20px]">auto_awesome</span>
                    </div>
                    <div>
                      <h3 className="font-bold text-[16px] tracking-[-0.01em]">{item.title}</h3>
                      <p className="text-[12px] text-[#8e8e93] font-medium">{item.date}</p>
                    </div>
                  </div>
                  <button className="text-[#8e8e93] opacity-0 group-hover:opacity-100 transition-opacity hover:text-[#ff3b30]">
                    <span className="material-symbols-outlined text-[20px]">delete</span>
                  </button>
                </div>
                
                <div className="bg-white/60 rounded-xl p-4 mt-2">
                  <div className="flex items-start gap-2 mb-2">
                    <span className="text-[12px] font-bold text-[#373549] shrink-0">Q.</span>
                    <p className="text-[14px] text-[#3a3a3c] leading-[1.5] line-clamp-1">{item.prompt}</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-[12px] font-bold text-[#4f8ef7] shrink-0">A.</span>
                    <p className="text-[14px] text-[#3a3a3c] leading-[1.5] line-clamp-2">{item.aiResponse}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
