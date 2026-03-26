import React, { useState, useEffect, useRef } from 'react';
import './Home.css';

export default function Home({ onNavigate }) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const [isFolderModalOpen, setIsFolderModalOpen] = useState(false);
  const [isFileModalOpen, setIsFileModalOpen] = useState(false);
  const [selectedColor, setSelectedColor] = useState('#3b82f6');

  const aiWinRef = useRef(null);
  const aiBtnRef = useRef(null);
  
  // Folders state
  const [folders, setFolders] = useState([
    { id: 'folder-26', name: '26년도 폴더', date: '2026. 3. 9. 오후 2:27', color: '#3b82f6', isFile: false, isStarred: true },
    { id: 'folder-sqld', name: 'SQLD', date: '2026. 3. 9. 오전 11:15', color: '#3b82f6', isFile: false, isStarred: false },
    { id: 'folder-25', name: '25년 1학기', date: '2026. 3. 8. 오후 6:40', color: '#3b82f6', isFile: false, isStarred: true },
    { id: 'folder-network', name: '컴퓨터 네트워크', date: '2026. 3. 7. 오후 1:12', color: '#3b82f6', isFile: false, isStarred: false },
    { id: 'folder-note', name: '2학기 필기폴더', date: '2026. 3. 5. 오전 9:45', color: '#3b82f6', isFile: false, isStarred: true },
    { id: 'folder-lecture', name: '2학기 강의 폴더', date: '2026. 3. 4. 오후 10:20', color: '#3b82f6', isFile: false, isStarred: false },
    { id: 'file-minutes', name: '주간 회의록.docx', date: '2026. 3. 10. 오전 10:15', color: '#1d1d1f', isFile: true, isStarred: false },
  ]);

  const [newFolderName, setNewFolderName] = useState('');
  const [newFileName, setNewFileName] = useState('');

  useEffect(() => {
    const handleOutsideClick = (e) => {
      // Modals
      if (e.target.classList.contains('modal-overlay')) {
        setIsFolderModalOpen(false);
        setIsFileModalOpen(false);
      }
      
      // AI Chat
      if (
        aiWinRef.current && !aiWinRef.current.contains(e.target) &&
        aiBtnRef.current && !aiBtnRef.current.contains(e.target)
      ) {
        setIsAiChatOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const toggleStar = (e, targetId) => {
    e.stopPropagation();
    setFolders(prev => prev.map(f => f.id === targetId ? { ...f, isStarred: !f.isStarred } : f));
  };

  const handleCreateFolder = () => {
    if (!newFolderName.trim()) return;
    const now = new Date();
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`;
    setFolders([{
      id: 'folder-' + Date.now(),
      name: newFolderName,
      date: timeStr,
      color: selectedColor,
      isFile: false,
      isStarred: false
    }, ...folders]);
    setIsFolderModalOpen(false);
    setNewFolderName('');
  };

  const handleCreateFile = () => {
    if (!newFileName.trim()) return;
    const now = new Date();
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`;
    setFolders([{
      id: 'file-' + Date.now(),
      name: newFileName,
      date: timeStr,
      color: '#1d1d1f',
      isFile: true,
      isStarred: false
    }, ...folders]);
    setIsFileModalOpen(false);
    setNewFileName('');
  };

  const favorites = folders.filter(f => f.isStarred);

  return (
    <div className="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
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
              <button className="sidebar-icon-btn" onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}>
                <span className="material-symbols-outlined">
                  {isSidebarCollapsed ? 'menu' : 'side_navigation'}
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
              {favorites.map(fav => (
                <div key={`fav-${fav.id}`} className="sidebar-nav-item" onClick={() => onNavigate('workspace')}>
                  <span className="material-symbols-outlined nav-icon" style={{ color: fav.color, fontVariationSettings: `"FILL" ${fav.isFile ? 0 : 1}` }}>
                    {fav.isFile ? 'description' : 'folder'}
                  </span>
                  <span className="nav-text truncate">{fav.name}</span>
                  {!fav.isFile && (
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

      <main id="home-main-content" className="custom-scrollbar"> 
        <div className="home-banner shrink-0">
          <div className="absolute w-[140px] h-[140px] bg-[#2d2b3e] rounded-full top-[10px] left-[40%]"></div>
          <div className="absolute w-[220px] h-[220px] bg-[#2d2b3e] rounded-full bottom-[-60px] left-[10%]"></div>
          
          <div className="relative z-10 max-w-[480px] flex flex-col gap-3 items-start">
            <div className="text-[36px] font-bold text-white tracking-[-0.03em] leading-[1.1]">하이</div>
            <div className="text-[15px] text-white/80 leading-[1.6] font-medium">
              Learn fun anywhere and anytime without any time limit just through the application.
            </div>
            <button className="bg-white text-[#373549] border-none rounded-full px-7 py-3.5 text-[14px] font-bold cursor-pointer transition-opacity hover:opacity-90">
              Get Started
            </button>
          </div>
          
          <div className="relative z-10 shrink-0 w-[240px] h-[140px] flex items-end justify-center">
            <svg fill="none" height="180" style={{ position: 'absolute', bottom: '-20px', right: '-20px' }} viewBox="0 0 240 180" width="240" xmlns="http://www.w3.org/2000/svg">
              <path d="M40 50 h12 v-12 h8 v12 h12 v8 h-12 v12 h-8 v-12 h-12 z" fill="#fff" opacity="0.9" transform="rotate(-15 50 50)"></path>
              <path d="M50 100 h20 v6 h-20 z" fill="#fff" opacity="0.9" transform="rotate(10 60 100)"></path>
              <g transform="translate(180, 20) rotate(15)">
                <path d="M15 0 C6.7 0 0 6.7 0 15 C0 20.3 2.7 25 6.7 27.8 L6.7 33.3 C6.7 34.2 7.5 35 8.3 35 L21.7 35 C22.6 35 23.3 34.2 23.3 33.3 L23.3 27.8 C27.3 25 30 20.3 30 15 Z" fill="none" stroke="#fff" strokeWidth="2.5"></path>
                <path d="M10 40 h10 M12 45 h6" stroke="#fff" strokeLinecap="round" strokeWidth="2.5"></path>
                <path d="M15 15 v10" stroke="#fff" strokeLinecap="round" strokeWidth="2.5"></path>
              </g>
              <path d="M130 90 C130 65 170 65 170 90 C170 105 160 115 150 115 C140 115 130 105 130 90 Z" fill="#fff"></path>
              <path d="M125 60 C140 45 165 45 175 60 C185 75 160 80 150 70 C140 80 115 75 125 60 Z" fill="#2d2b3e"></path>
              <circle cx="140" cy="85" fill="#2d2b3e" r="2.5"></circle>
              <circle cx="160" cy="85" fill="#2d2b3e" r="2.5"></circle>
              <path d="M145 95 Q150 100 155 95" fill="none" stroke="#2d2b3e" strokeLinecap="round" strokeWidth="2"></path>
              <path d="M110 180 C110 130 190 130 190 180 Z" fill="#2d2b3e"></path>
              <path d="M90 160 C110 145 125 155 135 165" fill="none" stroke="#fff" strokeLinecap="round" strokeWidth="12"></path>
              <path d="M210 160 C190 145 175 155 165 165" fill="none" stroke="#fff" strokeLinecap="round" strokeWidth="12"></path>
              <path d="M85 140 L150 165 L150 200 L85 175 Z" fill="#fff" stroke="#2d2b3e" strokeLinejoin="round" strokeWidth="2"></path>
              <path d="M215 140 L150 165 L150 200 L215 175 Z" fill="#f4f4f5" stroke="#2d2b3e" strokeLinejoin="round" strokeWidth="2"></path>
              <path d="M95 150 L140 168 M95 158 L140 176" stroke="#2d2b3e" strokeLinecap="round" strokeWidth="2"></path>
              <path d="M205 150 L160 168 M205 158 L160 176" stroke="#2d2b3e" strokeLinecap="round" strokeWidth="2"></path>
            </svg>
          </div>
        </div>

        <div className="shrink-0">
          <div className="flex items-center justify-end mb-4">
            <button className="flex items-center gap-1 px-2 py-1 bg-transparent border-none text-[14px] font-bold text-[#3b82f6] cursor-pointer rounded-lg hover:bg-blue-50 transition-colors">
              전체보기
              <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
            </button>
          </div>
          
          <div className="folder-grid">
            {folders.map(folder => (
              <div key={folder.id} className="folder-card" onClick={() => onNavigate('workspace')}>
                <button 
                  className={`star-btn ${folder.isStarred ? 'starred' : ''}`} 
                  onClick={(e) => toggleStar(e, folder.id)}
                >
                  <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: `"FILL" ${folder.isStarred ? 1 : 0}` }}>star</span>
                </button>
                <span className="material-symbols-outlined text-[48px] opacity-90" style={{ color: folder.color, fontVariationSettings: `"FILL" ${folder.isFile ? 0 : 1}` }}>
                  {folder.isFile ? 'description' : 'folder'}
                </span>
                <div className="mt-auto">
                  <div className="text-[15px] font-bold text-[#1d1d1f] tracking-[-0.01em] leading-[1.3] truncate">{folder.name}</div>
                  <div className="text-[12px] text-[#aeaeb2] font-medium mt-1 truncate">{folder.date}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="h-[140px] shrink-0"></div>
        
        <div className="fab-group z-50">
          <button className="fab-btn group" onClick={() => setIsFolderModalOpen(true)}>
            <span className="material-symbols-outlined group-hover:scale-110 transition-transform">create_new_folder</span>
            <span>새 폴더 생성</span>
          </button>
          <button className="fab-btn group" onClick={() => setIsFileModalOpen(true)}>
            <span className="material-symbols-outlined group-hover:scale-110 transition-transform">description</span>
            <span>새 파일 생성</span>
          </button>
        </div>
      </main>

      {/* AI Assistant Floating Button */}
      <div className="home-ai-fab z-50" title="Ask AI Assistant" ref={aiBtnRef} onClick={() => setIsAiChatOpen(!isAiChatOpen)}>
        <span className="material-symbols-outlined text-[28px]" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
      </div>

      {/* AI Chat Window Overlay */}
      <div ref={aiWinRef} className={`home-ai-chat-window z-[60] ${isAiChatOpen ? 'open' : ''}`}>
        <div className="chat-header">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-[#373549] rounded-lg flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-[18px]">auto_awesome</span>
            </div>
            <span className="text-[16px] font-bold tracking-[-0.01em]">Lecto AI Assistant</span>
          </div>
          <button className="btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93]" onClick={() => setIsAiChatOpen(false)}>
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>
        <div className="chat-content custom-scrollbar">
          <div className="chat-bubble bubble-ai">
            안녕하세요! 어떤 것을 도와드릴까요? <br />강의 노트 요약이나 시험 문제 생성 등을 도와드릴 수 있습니다.
          </div>
          <div className="chat-bubble bubble-user">
            지난 '컴퓨터 네트워크' 수업 내용을 요약해줘.
          </div>
          <div className="chat-bubble bubble-ai">
            네, '컴퓨터 네트워크' 폴더의 최신 노트를 분석하여 요약해 드릴게요. <br /><br />
            1. OSI 7계층의 구조와 각 계층의 역할<br />
            2. TCP/IP 4계층 모델과의 차이점<br />
            3. 데이터 캡슐화와 비캡슐화 과정...
          </div>
        </div>
        <div className="chat-footer">
          <div className="chat-input-container">
            <input className="chat-input" placeholder="AI에게 질문해보세요..." type="text" />
            <button className="btn-ghost-icon p-1 text-[#373549]">
              <span className="material-symbols-outlined text-[20px]">send</span>
            </button>
          </div>
        </div>
      </div>

      {/* Folder Creation Modal */}
      <div className={`modal-overlay ${isFolderModalOpen ? 'open' : ''}`}>
        <div className="modal-card" onClick={(e) => e.stopPropagation()}>
          <h2 className="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 폴더 생성</h2>
          <p className="text-[14px] text-[#8e8e93] mb-6">이름과 색상을 지정해주세요.</p>
          <input 
            className="modal-input mb-5" 
            placeholder="폴더 이름 입력" 
            autoFocus
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            type="text" 
          />
          <p className="text-[14px] font-bold text-[#3a3a3c] mb-3">테마 색상</p>
          <div className="color-picker-container">
            {['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'].map(color => (
              <div 
                key={color}
                className={`color-circle ${selectedColor === color ? 'selected' : ''}`} 
                style={{ backgroundColor: color }}
                onClick={() => setSelectedColor(color)}
              ></div>
            ))}
          </div>
          <div className="flex justify-end gap-3">
            <button className="modal-btn-secondary" onClick={() => { setIsFolderModalOpen(false); setNewFolderName(''); }}>취소</button>
            <button className="modal-btn-primary" onClick={handleCreateFolder}>추가</button>
          </div>
        </div>
      </div>

      {/* File Creation Modal */}
      <div className={`modal-overlay ${isFileModalOpen ? 'open' : ''}`}>
        <div className="modal-card" onClick={(e) => e.stopPropagation()}>
          <h2 className="text-[20px] font-bold tracking-[-0.02em] mb-1.5">새 파일 생성</h2>
          <p className="text-[14px] text-[#8e8e93] mb-6">노트의 제목을 입력해주세요.</p>
          <input 
            className="modal-input mb-8" 
            placeholder="파일 이름 입력" 
            autoFocus
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
            type="text" 
          />
          <div className="flex justify-end gap-3">
            <button className="modal-btn-secondary" onClick={() => { setIsFileModalOpen(false); setNewFileName(''); }}>취소</button>
            <button className="modal-btn-primary" onClick={handleCreateFile}>생성</button>
          </div>
        </div>
      </div>
    </div>
  );
}
