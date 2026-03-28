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
  transcriptions = []
}) {
  const [activeTab, setActiveTab] = useState('folders'); // 'folders' or 'voice'
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [width, setWidth] = useState(280);
  const [toast, setToast] = useState('');

  // ── 리사이징 로직 ──
  const isResizingRef = useRef(false);

  const handleResizerMouseDown = (e) => {
    e.preventDefault();
    isResizingRef.current = true;
    document.body.classList.add('is-resizing');
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (e) => {
    if (!isResizingRef.current) return;
    const newWidth = e.clientX;
    if (newWidth > 200 && newWidth < 600) {
      setWidth(newWidth);
    }
  };

  const handleMouseUp = () => {
    isResizingRef.current = false;
    document.body.classList.remove('is-resizing');
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 3000);
  };

  return (
    <>
      <aside
        className={`${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}
        id="sidebar"
        style={{ width: isSidebarCollapsed ? '0px' : width + 'px', flexShrink: 0 }}
      >
        <div className="sidebar-main-card h-full bg-white flex flex-col p-5 overflow-hidden">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2 font-extrabold tracking-tight">
              <div className="w-7 h-7 bg-[#1d1d1f] rounded-lg flex items-center justify-center">
                <span className="material-symbols-outlined text-[17px] text-white" style={{ transform: 'scaleX(-1)' }}>side_navigation</span>
              </div>
              <span className="text-[18px]">Toyo</span>
            </div>
            <div className="flex gap-1">
              <button
                className="btn-ghost-icon p-1.5 rounded-lg"
                title="사이드바 접기"
                onClick={() => setIsSidebarCollapsed(true)}
              >
                <span className="material-symbols-outlined text-[20px] text-[#8e8e93]">dock_to_left</span>
              </button>
            </div>
          </div>

          {/* 탭 전환 버튼 */}
          <div className="bg-[#f2f2f7] p-1 rounded-xl flex gap-1 mb-6">
            <button
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-[13px] font-bold transition-all ${activeTab === 'folders' ? 'bg-white shadow-sm text-[#1d1d1f]' : 'text-[#8e8e93] hover:text-[#3a3a3c]'}`}
              onClick={() => setActiveTab('folders')}
            >
              <span className="material-symbols-outlined text-[18px]">folder</span>폴더
            </button>
            <button
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-[13px] font-bold transition-all ${activeTab === 'voice' ? 'bg-white shadow-sm text-[#1d1d1f]' : 'text-[#8e8e93] hover:text-[#3a3a3c]'}`}
              onClick={() => setActiveTab('voice')}
            >
              <span className="material-symbols-outlined text-[18px]">record_voice_over</span>전사 내용
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
            />
          )}

          {/* 홈 버튼 */}
          <footer className="mt-auto pt-4 flex items-center justify-center border-t border-black/5">
            <button
              className="btn-ghost-icon p-2.5 rounded-xl cursor-pointer flex items-center text-[#aeaeb2] justify-center hover:text-[#1d1d1f] transition-colors"
              onClick={onNavigateHome}
            >
              <span className="material-symbols-outlined text-[24px]" style={{ fontVariationSettings: '"FILL" 1' }}>home</span>
            </button>
          </footer>
        </div>
      </aside>

      {/* 리사이저 */}
      <div
        className="w-1.5 hover:bg-blue-500/20 transition-colors cursor-col-resize flex items-center justify-center group active:bg-blue-500/40 mx-[-6px] z-20"
        id="resizer-left"
        style={{ display: isSidebarCollapsed ? 'none' : undefined }}
        onMouseDown={handleResizerMouseDown}
      >
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-blue-400 group-active:bg-blue-500 transition-colors"></div>
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
