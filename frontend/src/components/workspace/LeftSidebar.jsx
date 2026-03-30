import React, { useState, useRef, useEffect } from 'react';
import FolderSideTab from './FolderSideTab';
import VoiceTransferSideTab from './VoiceTransferSideTab';

export default function LeftSidebar({ 
  onNavigateHome, 
  fileTree, 
  setFileTree, 
  favorites, 
  setFavorites, 
  onFileSelect,
  transcriptions = [],
  onAddToNote,
  onAskAi
}) {
  const [activeTab, setActiveTab] = useState('folders'); // 'folders' or 'voice'
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [width, setWidth] = useState(280);
  const [toast, setToast] = useState('');

  // ── 리사이징 로직 ──
  const isResizingRef = useRef(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
      const newWidth = e.clientX - 12;
      if (newWidth > 160 && newWidth < 600) {
        setWidth(newWidth);
      }
    };
    const handleMouseUp = () => {
      if (!isResizingRef.current) return;
      isResizingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.body.classList.remove('is-resizing');
    };
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const handleResizerMouseDown = () => {
    isResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.body.classList.add('is-resizing');
  };

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 2000);
  };

  return (
    <>
      <aside
        className={`${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}
        id="sidebar"
        style={{ width: isSidebarCollapsed ? '0px' : width + 'px', flexShrink: 0 }}
      >
        <div className="card h-full bg-white flex flex-col p-5 overflow-hidden">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2 font-extrabold tracking-tight">
              <div className="w-8 h-8 bg-[#1d1d1f] rounded-[10px] flex items-center justify-center">
                <span className="material-symbols-outlined text-[20px] text-white">menu_book</span>
              </div>
              <span className="text-[20px]">LectoAI</span>
            </div>
            <div className="flex gap-1">
              <button
                className="btn-ghost-icon p-1.5 rounded-[10px]"
                title="사이드바 접기"
                onClick={() => setIsSidebarCollapsed(true)}
              >
                <span className="material-symbols-outlined text-[22px] text-[#8e8e93]">menu_open</span>
              </button>
            </div>
          </div>

          {/* 탭 전환 버튼 (원본 디자인 복원) */}
          <div className="bg-gray-100/50 p-1 rounded-lg flex gap-1 mb-4">
            <button
              className={`flex-1 py-1.5 rounded-md text-[12px] font-bold transition-all ${activeTab === 'folders' ? 'bg-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-black' : 'text-gray-500 hover:text-gray-700'}`}
              onClick={() => setActiveTab('folders')}
            >
              폴더
            </button>
            <button
              className={`flex-1 py-1.5 rounded-md text-[12px] font-bold transition-all ${activeTab === 'voice' ? 'bg-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] text-black' : 'text-gray-500 hover:text-gray-700'}`}
              onClick={() => setActiveTab('voice')}
            >
              전사 내용
            </button>
          </div>

          {/* 탭 컨텐츠 */}
          {activeTab === 'folders' ? (
            <FolderSideTab 
              fileTree={fileTree}
              setFileTree={setFileTree}
              favorites={favorites}
              setFavorites={setFavorites}
              onFileSelect={onFileSelect}
              showToast={showToast}
            />
          ) : (
            <VoiceTransferSideTab 
              transcriptions={transcriptions}
              onAddToNote={onAddToNote}
              onAskAi={onAskAi}
            />
          )}

          {/* 홈 버튼 (원본 디자인 복원: 구분선 제거) */}
          <footer className="mt-auto pt-4 flex items-center justify-center">
            <button
              className="btn-ghost-icon p-2.5 rounded-xl cursor-pointer flex items-center text-[#aeaeb2] justify-center"
              onClick={onNavigateHome}
            >
              <span className="material-symbols-outlined text-[24px]" style={{ fontVariationSettings: '"FILL" 1' }}>home</span>
            </button>
          </footer>
        </div>
      </aside>

      {/* 리사이저 (원본 디자인 복원) */}
      <div
        className="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
        id="resizer-left"
        style={{ display: isSidebarCollapsed ? 'none' : undefined }}
        onMouseDown={handleResizerMouseDown}
      >
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
      </div>

      {/* 펼치기 플로팅 버튼 (사이드바 숨김 시) */}
      {isSidebarCollapsed && (
        <button
          className="fixed left-4 top-5 z-[50] w-10 h-10 bg-white shadow-lg rounded-full flex items-center justify-center border border-black/5 hover:scale-110 transition-transform active:scale-95"
          onClick={() => setIsSidebarCollapsed(false)}
        >
          <span className="material-symbols-outlined text-[20px] text-[#1d1d1f]">dock_to_left</span>
        </button>
      )}

      {/* 토스트 */}
      <div className={`toast${toast ? ' show' : ''}`} id="toast">{toast}</div>
    </>
  );
}
