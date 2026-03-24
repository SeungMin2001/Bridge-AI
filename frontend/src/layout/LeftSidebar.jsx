import React, { useState } from 'react';

export default function LeftSidebar({ onNavigateHome }) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  return (
    <aside 
      id="sidebar" 
      className={`w-[280px] flex flex-col h-full shrink-0 overflow-hidden transition-[width] duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}
    >
      <div className="sidebar-main-card card">
        <div className="sidebar-header transition-all">
          <div className="sidebar-logo-section">
            <div className="sidebar-logo-box">
              <span className="material-symbols-outlined text-white text-[20px]">menu_book</span>
            </div>
            <span className="collapsible-content sidebar-logo-text">LectoAI</span>
          </div>
          <div className="sidebar-btn-group">
            <button 
              className="sidebar-icon-btn" 
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            >
              <span className="material-symbols-outlined">
                {isSidebarCollapsed ? 'menu' : 'side_navigation'}
              </span>
            </button>
            <button className="sidebar-icon-btn collapsible-content" title="새 노트">
              <span className="material-symbols-outlined">edit_note</span>
            </button>
          </div>
        </div>

        {/* Search (검색) */}
        <div className="sidebar-search-container collapsible-content">
          <span className="material-symbols-outlined">search</span>
          <input
            className="bg-transparent border-none focus:ring-0 p-0 sidebar-search-text w-full placeholder-[#aeaeb2]"
            placeholder="제목으로 검색" type="text"
          />
        </div>

        {/* Sections (섹션들) */}
        <div className="sidebar-content collapsible-content custom-scrollbar flex flex-col gap-6">
          <section>
            <div className="sidebar-section-title">활성화한 파일</div>
            <div id="active-file-display"></div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="sidebar-section-title !mb-0 !px-0">구조</span>
              <div className="flex gap-1">
                <button className="sidebar-icon-btn !p-0.5" title="새 폴더">
                  <span className="material-symbols-outlined !text-[18px]">create_new_folder</span>
                </button>
                <button className="sidebar-icon-btn !p-0.5" title="새 파일">
                  <span className="material-symbols-outlined !text-[18px]">note_add</span>
                </button>
              </div>
            </div>
            <div id="file-tree"></div>
          </section>

          <section>
            <div className="sidebar-section-title">즐겨찾기</div>
            <div id="favorites-list" className="flex flex-col gap-0.5"></div>
          </section>

          <section>
            <div className="sidebar-section-title">리스트</div>
            <div id="all-list" className="flex flex-col gap-0.5 pb-4"></div>
          </section>
        </div>

        {/* Home Button (홈 버튼) */}
        <div className="mt-auto pt-5 flex justify-center">
          <button 
            className="sidebar-icon-btn !p-3 rounded-full text-[#1d1d1f]"
            onClick={onNavigateHome}
          >
            <span className="material-symbols-outlined !text-[24px]" style={{ fontVariationSettings: "'FILL' 1" }}>home</span>
          </button>
        </div>
      </div>
      </aside>
  );
}
