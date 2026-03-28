import React from 'react';

export default function HomeSidebar({
  isCollapsed,
  setIsCollapsed,
  onNavigate,
  fileTree,
  favorites
}) {
  return (
    <aside
      id="sidebar"
      className={`w-[280px] flex flex-col h-full shrink-0 overflow-hidden transition-[width] duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] ${isCollapsed ? 'sidebar-collapsed' : ''}`}
    >
      <div className="sidebar-main-card card">
        <div className="sidebar-header transition-all">
          <div className="sidebar-logo-section">
            <div className="sidebar-logo-box">
              <span className="material-symbols-outlined text-white text-[20px]">menu_book</span>
            </div>
            <span className="collapsible-content sidebar-logo-text font-extrabold">LectoAI</span>
          </div>
          <div className="sidebar-btn-group">
            <button className="sidebar-icon-btn" onClick={() => setIsCollapsed(!isCollapsed)}>
              <span className="material-symbols-outlined">
                {isCollapsed ? 'menu' : 'side_navigation'}
              </span>
            </button>
            <button
              className="sidebar-icon-btn collapsible-content"
              onClick={() => onNavigate('workspace')}
            >
              <span className="material-symbols-outlined">edit_note</span>
            </button>
          </div>
        </div>
        
        <div className="sidebar-search-container collapsible-content">
          <span className="material-symbols-outlined">search</span>
          <span className="sidebar-search-text">제목으로 검색</span>
        </div>

        <div className="sidebar-content collapsible-content custom-scrollbar">
          <div className="sidebar-section-title">즐겨찾기</div>
          <div id="favorites-list" className="flex flex-col gap-1">
            {fileTree.filter(item => favorites.has(item.id)).map(fav => (
              <div key={`fav-${fav.id}`} className="sidebar-nav-item" onClick={() => onNavigate('workspace')}>
                <span className="material-symbols-outlined nav-icon" style={{ color: fav.color, fontVariationSettings: `'FILL' ${fav.type === 'folder' ? 1 : 0}` }}>
                  {fav.type === 'folder' ? 'folder' : 'description'}
                </span>
                <span className="nav-text truncate">{fav.name}</span>
                {fav.type === 'folder' && (
                  <span className="material-symbols-outlined text-[#8e8e93] text-[18px]">expand_more</span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-auto pt-5 flex justify-center">
          <button className="sidebar-icon-btn !p-3 rounded-full text-[#1d1d1f]" onClick={() => onNavigate('home')}>
            <span className="material-symbols-outlined !text-[24px]" style={{ fontVariationSettings: "'FILL' 1" }}>home</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
