import React, { useState } from 'react';

export default function LeftSidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <aside className={`w-[280px] flex flex-col h-full shrink-0 overflow-hidden ${isCollapsed ? 'sidebar-collapsed' : ''}`} id="sidebar">
      <div className="card flex-1 flex flex-col p-5 pb-5 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between mb-5 sidebar-header-row">
          <div className="flex items-center gap-2 font-heavy-heading text-[18px]">
            <div className="w-[28px] h-[28px] bg-[#1d1d1f] rounded-[8px] flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-white text-[17px] scale-x-[-1]">menu_book</span>
            </div>
            <span className="collapsible-content">LectoAI</span>
          </div>
          <div className="flex gap-1 sidebar-header-btns">
            <button 
              className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]" 
              onClick={() => setIsCollapsed(!isCollapsed)}
            >
              <span className="material-symbols-outlined text-[20px]">{isCollapsed ? 'menu' : 'grid_view'}</span>
            </button>
            <button className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93] collapsible-content" title="새 노트">
              <span className="material-symbols-outlined text-[20px]">edit_note</span>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="sidebar-search-bg rounded-[12px] px-3.5 py-2 flex items-center gap-2 mb-5 collapsible-content">
          <span className="material-symbols-outlined text-[#8e8e93] text-[18px]">search</span>
          <input
            className="bg-transparent border-none focus:ring-0 p-0 text-[13px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
            placeholder="제목으로 검색" type="text"
          />
        </div>

        {/* Sections */}
        <div className="collapsible-content flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">활성화한 파일</span>
            </div>
            <div id="active-file-display"></div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">구조</span>
              <div className="flex gap-1">
                <button className="btn-ghost-icon p-0.5 rounded-lg" title="새 폴더">
                  <span className="material-symbols-outlined text-[18px]" style={{ color: '#8e8e93' }}>create_new_folder</span>
                </button>
                <button className="btn-ghost-icon p-0.5 rounded-lg" title="새 파일">
                  <span className="material-symbols-outlined text-[18px]" style={{ color: '#8e8e93' }}>note_add</span>
                </button>
              </div>
            </div>
            <div id="file-tree"></div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">즐겨찾기</span>
            </div>
            <div id="favorites-list" className="flex flex-col gap-0.5"></div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">리스트</span>
            </div>
            <div id="all-list" className="flex flex-col gap-0.5 pb-4"></div>
          </section>
        </div>

        {/* Footer */}
        <div className="mt-auto pt-4 flex items-center justify-center">
          <a className="btn-ghost-icon p-2.5 rounded-xl cursor-pointer flex items-center text-[#aeaeb2] justify-center" href="home.html">
            <span className="material-symbols-outlined text-[24px]" style={{ fontVariationSettings: '"FILL" 1' }}>home</span>
          </a>
        </div>
      </div>
    </aside>
  );
}
