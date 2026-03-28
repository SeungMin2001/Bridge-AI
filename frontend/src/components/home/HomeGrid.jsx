import React from 'react';

const FOLDER_COLORS = {
  '#3b82f6': { body: 'fc-blue', tab: 'fc-blue-tab' },
  '#2dd4bf': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#ef4444': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f87171': { body: 'fc-coral', tab: 'fc-coral-tab' },
  '#f59e0b': { body: 'fc-amber', tab: 'fc-amber-tab' },
  '#10b981': { body: 'fc-teal', tab: 'fc-teal-tab' },
  '#8b5cf6': { body: 'fc-purple', tab: 'fc-purple-tab' },
  '#a78bfa': { body: 'fc-purple', tab: 'fc-purple-tab' },
};

export default function HomeGrid({
  items,
  favorites,
  toggleStar,
  onEnterFolder,
  onNavigate,
  navigationStack,
  onGoBack,
  title,
  onOpenFolderModal,
  onOpenFileModal
}) {
  return (
    <div className="flex-1 mt-6"> 
      <div className="shrink-0">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            {navigationStack.length > 0 && (
              <button 
                onClick={onGoBack}
                className="flex items-center justify-center p-2 bg-white border-none text-[#1d1d1f] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
              >
                <span className="material-symbols-outlined text-[20px]">arrow_back</span>
              </button>
            )}
            <span className="section-title !m-0 transition-all duration-300">{title}</span>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => onNavigate('ai-history')}
              className="flex items-center justify-center p-2 bg-white border-none text-[#3a3a3c] cursor-pointer rounded-xl hover:bg-[#f2f2f7] transition-colors shadow-sm"
              title="AI 명령 기록"
            >
              <span className="material-symbols-outlined text-[20px]">history</span>
            </button>
            <button className="flex items-center gap-1 px-3 py-1.5 bg-white border-none text-[14px] font-bold text-[#3b82f6] cursor-pointer rounded-xl hover:bg-blue-50 transition-colors shadow-sm">
              전체보기
              <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
            </button>
          </div>
        </div>
        
        <div className="folder-grid">
          {items.map(item => {
            const isStarred = favorites.has(item.id);

            if (item.type === 'folder') {
              const colors = FOLDER_COLORS[item.color] || { body: 'fc-blue', tab: 'fc-blue-tab' };
              return (
                <div key={item.id} className="folder-card" onClick={(e) => onEnterFolder(e, item)}>
                  <div className={`folder-tab ${colors.tab}`} style={{ width: '45%' }}></div>
                  <div className={`folder-body ${colors.body}`}>
                    <button
                      className={`star-btn ${isStarred ? 'starred' : ''}`}
                      onClick={(e) => toggleStar(e, item.id)}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '16px', fontVariationSettings: `'FILL' ${isStarred ? 1 : 0}` }}>star</span>
                    </button>
                    <div className="folder-icon-area">
                      <span className="material-symbols-outlined" style={{ fontSize: '22px', color: '#fff', fontVariationSettings: "'FILL' 1" }}>folder</span>
                    </div>
                    <div className="folder-card-name">{item.name}</div>
                    <div className="folder-card-date">{item.date || ''}</div>
                  </div>
                </div>
              );
            } else {
              return (
                <div key={item.id} className="folder-card file-card" onClick={() => onNavigate('workspace')} style={{ display: 'flex', flexDirection: 'column', height: '160px' }}>
                  <div style={{ height: '10px', flexShrink: 0 }}></div>
                  <div style={{
                    background: '#fff',
                    borderRadius: '14px',
                    padding: 0,
                    flex: 1,
                    position: 'relative',
                    overflow: 'hidden',
                    boxShadow: '2px 3px 0px #e0e0e8',
                    border: '1.5px solid #e5e5ea',
                    display: 'flex',
                    flexDirection: 'column',
                  }}>
                    <div style={{ height: '6px', background: 'linear-gradient(90deg, #6366f1, #a78bfa)', borderRadius: '12px 12px 0 0' }}></div>
                    <div style={{
                      position: 'absolute', top: '30px', left: 0, right: 0, bottom: 0,
                      backgroundImage: 'repeating-linear-gradient(transparent, transparent 22px, #f0f0f5 22px, #f0f0f5 23px)',
                      opacity: 0.6
                    }}></div>
                    <div style={{ position: 'relative', zIndex: 1, padding: '14px', display: 'flex', flexDirection: 'column', flex: 1 }}>
                      <button
                        className={`star-btn ${isStarred ? 'starred' : ''}`}
                        onClick={(e) => toggleStar(e, item.id)}
                        style={{ position: 'absolute', top: '14px', right: '10px', background: 'rgba(0,0,0,0.04)', color: '#d1d1d6' }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: '16px', fontVariationSettings: `'FILL' ${isStarred ? 1 : 0}` }}>star</span>
                      </button>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <div style={{ width: '36px', height: '36px', background: '#ede9fe', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <span className="material-symbols-outlined" style={{ fontSize: '20px', color: '#6366f1', fontVariationSettings: "'FILL' 1" }}>article</span>
                        </div>
                        <span style={{ fontSize: '10px', fontWeight: 700, color: '#6366f1', background: '#ede9fe', padding: '2px 8px', borderRadius: '100px', letterSpacing: '0.04em' }}>FILE</span>
                      </div>
                      <div style={{ marginTop: 'auto' }}>
                        <div className="folder-card-name" style={{ color: '#1d1d1f', fontSize: '13px' }}>{item.name}</div>
                        <div className="folder-card-date" style={{ color: '#8e8e93' }}>{item.date || ''}</div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            }
          })}
        </div>
      </div>

      <div className="h-[140px] shrink-0"></div>

      <div className="fab-group z-50">
        <button className="fab-btn group" onClick={onOpenFolderModal}>
          <span className="material-symbols-outlined group-hover:scale-110 transition-transform">create_new_folder</span>
          <span>새 폴더 생성</span>
        </button>
        <button className="fab-btn group" onClick={onOpenFileModal}>
          <span className="material-symbols-outlined group-hover:scale-110 transition-transform">description</span>
          <span>새 파일 생성</span>
        </button>
      </div>
    </div>
  );
}
