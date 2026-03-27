import React, { useState, useRef, useCallback, useEffect } from 'react';

// ── 초기 파일 트리 데이터 (file1.html과 동일) ──
const INITIAL_FILE_TREE = [
  {
    id: 'f1', type: 'folder', name: '컴퓨터 네트워크', color: '#3b82f6', expanded: true,
    children: [
      {
        id: 'f1-1', type: 'folder', name: '강의요점', color: '#3b82f6', expanded: true,
        children: [
          { id: 'f1-1-1', type: 'file', name: '강의1', content: '' },
          { id: 'f1-1-2', type: 'file', name: '강의2', content: '' },
        ]
      }
    ]
  },
  { id: 'f2', type: 'folder', name: 'SQLD', color: '#3b82f6', expanded: false, children: [] },
  { id: 'f3', type: 'folder', name: '25년 1학기', color: '#5856d6', expanded: false, children: [] },
  { id: 'f4', type: 'folder', name: '26년도 폴더', color: '#ff9500', expanded: false, children: [] },
  { id: 'f5', type: 'file', name: '주간 회의록.docx', content: '' },
];

const FOLDER_COLORS = ['#3b82f6', '#5856d6', '#ff9500', '#34c759', '#ff3b30', '#af52de'];

// ── 유틸리티 함수들 ──
function findNode(id, nodes) {
  for (const n of nodes) {
    if (n.id === id) return n;
    if (n.children) {
      const found = findNode(id, n.children);
      if (found) return found;
    }
  }
  return null;
}

function deleteNode(id, nodes) {
  const idx = nodes.findIndex(n => n.id === id);
  if (idx !== -1) { nodes.splice(idx, 1); return true; }
  for (const n of nodes) {
    if (n.children && deleteNode(id, n.children)) return true;
  }
  return false;
}

function flattenAll(nodes) {
  const result = [];
  for (const n of nodes) {
    result.push(n);
    if (n.children) result.push(...flattenAll(n.children));
  }
  return result;
}

function genId() { return 'n' + Date.now() + Math.random().toString(36).slice(2, 6); }

// ── TreeItem 컴포넌트 (재귀) ──
function TreeItem({ node, depth, activeFileId, onSelectFile, onToggleFolder, onShowContextMenu }) {
  const isFile = node.type === 'file';
  const isSelected = node.id === activeFileId;

  const handleClick = (e) => {
    e.stopPropagation();
    if (isFile) {
      onSelectFile(node.id);
    } else {
      onToggleFolder(node.id);
    }
  };

  const handleDotsClick = (e) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    onShowContextMenu(node.id, rect.right + 4, rect.top);
  };

  return (
    <div>
      <div
        className={`tree-item${isSelected ? ' is-selected' : ''}`}
        style={{ paddingLeft: (8 + depth * 12) + 'px' }}
        onClick={handleClick}
      >
        {/* 아이콘 */}
        <span
          className="material-symbols-outlined shrink-0"
          style={{
            fontSize: '18px',
            color: isFile ? (isSelected ? '#1d1d1f' : '#8e8e93') : (node.color || '#3b82f6'),
            fontVariationSettings: isFile ? undefined : '"FILL" 1',
          }}
        >
          {isFile ? 'description' : (node.expanded ? 'folder_open' : 'folder')}
        </span>

        {/* 이름 */}
        <span className={`text-[13px] flex-1 truncate ${isSelected ? 'font-semibold text-[#1d1d1f]' : 'font-medium text-[#3a3a3c]'}`}>
          {node.name}
        </span>

        {/* 쉐브론 (폴더만) */}
        {!isFile && (
          <span
            className={`material-symbols-outlined chevron${node.expanded ? ' open' : ''}`}
            style={{ fontSize: '16px' }}
          >
            chevron_right
          </span>
        )}

        {/* 더보기 버튼 */}
        <button className="dots-btn" title="더 보기" onClick={handleDotsClick}>
          <span className="material-symbols-outlined" style={{ fontSize: '16px', pointerEvents: 'none' }}>more_horiz</span>
        </button>
      </div>

      {/* 하위 항목 (폴더인 경우) */}
      {!isFile && node.children && (
        <div
          className={`tree-children ${node.expanded ? 'expanded' : 'collapsed'}`}
          style={{ maxHeight: node.expanded ? '9999px' : '0' }}
        >
          {node.children.map(child => (
            <TreeItem
              key={child.id}
              node={child}
              depth={depth + 1}
              activeFileId={activeFileId}
              onSelectFile={onSelectFile}
              onToggleFolder={onToggleFolder}
              onShowContextMenu={onShowContextMenu}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── ContextMenu (드롭다운 메뉴) ──
function ContextMenu({ visible, x, y, targetNode, isFavorite, onClose, onAction }) {
  if (!visible) return null;

  return (
    <div className="dropdown-menu" style={{ display: 'block', left: x + 'px', top: y + 'px' }}>
      <div className="dropdown-item" onClick={() => onAction('open')}>
        <span className="material-symbols-outlined text-[16px] text-[#3b82f6]">open_in_new</span>열기
      </div>
      <div className="dropdown-item" onClick={() => onAction('rename')}>
        <span className="material-symbols-outlined text-[16px] text-[#8e8e93]">edit</span>이름 변경
      </div>
      {targetNode?.type === 'folder' && (
        <>
          <div className="dropdown-item" onClick={() => onAction('new-file')}>
            <span className="material-symbols-outlined text-[16px] text-[#8e8e93]">note_add</span>새 파일
          </div>
          <div className="dropdown-item" onClick={() => onAction('new-folder')}>
            <span className="material-symbols-outlined text-[16px] text-[#8e8e93]">create_new_folder</span>새 폴더
          </div>
        </>
      )}
      <div className="dropdown-item" onClick={() => onAction('favorite')}>
        <span className="material-symbols-outlined text-[16px] text-[#ff9500]">
          {isFavorite ? 'star_border' : 'star'}
        </span>
        {isFavorite ? '즐겨찾기 제거' : '즐겨찾기 추가'}
      </div>
      <div className="dropdown-divider"></div>
      <div className="dropdown-item danger" onClick={() => onAction('delete')}>
        <span className="material-symbols-outlined text-[16px]">delete</span>삭제
      </div>
    </div>
  );
}

// ── 메인 LeftSidebar 컴포넌트 ──
export default function LeftSidebar({ onNavigateHome, transcriptions = [], activeFileId: externalActiveFileId, onFileSelect }) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [sidebarMode, setSidebarMode] = useState('transcription'); // 'folder' | 'transcription'
  const [fileTree, setFileTree] = useState(INITIAL_FILE_TREE);
  const [activeFileId, setActiveFileId] = useState(externalActiveFileId || 'f1-1-1');
  const [favorites, setFavorites] = useState(new Set(['f1', 'f3', 'f4']));
  const [searchQuery, setSearchQuery] = useState('');
  const [toast, setToast] = useState('');
  const [width, setWidth] = useState(280);

  // ── 리사이징 로직 ──
  const isResizingRef = useRef(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
      // 좌측 패딩 12px를 빼서 정확한 사이드바 너비 계산
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

  // Context Menu
  const [ctxMenu, setCtxMenu] = useState({ visible: false, x: 0, y: 0, targetId: null });
  const colorIdxRef = useRef(0);

  const showToast = useCallback((msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 2000);
  }, []);

  const handleSelectFile = useCallback((id) => {
    setActiveFileId(id);
    if (onFileSelect) onFileSelect(id, findNode(id, fileTree));
  }, [fileTree, onFileSelect]);

  const handleToggleFolder = useCallback((id) => {
    setFileTree(prev => {
      const copy = JSON.parse(JSON.stringify(prev));
      const node = findNode(id, copy);
      if (node) node.expanded = !node.expanded;
      return copy;
    });
  }, []);

  const handleShowContextMenu = useCallback((id, x, y) => {
    setCtxMenu({ visible: true, x, y, targetId: id });
  }, []);

  const handleCloseContextMenu = useCallback(() => {
    setCtxMenu({ visible: false, x: 0, y: 0, targetId: null });
  }, []);

  const handleContextAction = useCallback((action) => {
    const targetId = ctxMenu.targetId;
    setCtxMenu({ visible: false, x: 0, y: 0, targetId: null });

    setFileTree(prev => {
      const copy = JSON.parse(JSON.stringify(prev));
      const node = findNode(targetId, copy);
      if (!node) return prev;

      switch (action) {
        case 'open':
          if (node.type === 'file') {
            setActiveFileId(targetId);
            if (onFileSelect) onFileSelect(targetId, node);
          } else {
            node.expanded = true;
          }
          break;
        case 'rename': {
          const newName = prompt('새 이름을 입력하세요:', node.name);
          if (newName && newName.trim()) {
            node.name = newName.trim();
            showToast(`"${newName.trim()}"으로 이름 변경됨`);
          }
          break;
        }
        case 'new-file': {
          if (node.type !== 'folder') break;
          const newFile = { id: genId(), type: 'file', name: '새 파일', content: '' };
          node.children.push(newFile);
          node.expanded = true;
          showToast('새 파일이 추가되었습니다');
          break;
        }
        case 'new-folder': {
          if (node.type !== 'folder') break;
          const color = FOLDER_COLORS[colorIdxRef.current++ % FOLDER_COLORS.length];
          const newFolder = { id: genId(), type: 'folder', name: '새 폴더', color, expanded: false, children: [] };
          node.children.push(newFolder);
          node.expanded = true;
          showToast('새 폴더가 추가되었습니다');
          break;
        }
        case 'favorite':
          setFavorites(prev => {
            const next = new Set(prev);
            if (next.has(targetId)) { next.delete(targetId); showToast('즐겨찾기에서 제거됨'); }
            else { next.add(targetId); showToast('즐겨찾기에 추가됨'); }
            return next;
          });
          break;
        case 'delete':
          deleteNode(targetId, copy);
          showToast(`"${node.name}" 삭제됨`);
          if (activeFileId === targetId) setActiveFileId('f1-1-1');
          break;
        default: break;
      }
      return copy;
    });
  }, [ctxMenu, activeFileId, onFileSelect, showToast]);

  // 상단 새 파일/폴더 버튼
  const handleNewFile = useCallback(() => {
    const newFile = { id: genId(), type: 'file', name: '새 파일', content: '' };
    setFileTree(prev => [...prev, newFile]);
    showToast('새 파일이 추가되었습니다');
  }, [showToast]);

  const handleNewFolder = useCallback(() => {
    const color = FOLDER_COLORS[colorIdxRef.current++ % FOLDER_COLORS.length];
    const newFolder = { id: genId(), type: 'folder', name: '새 폴더', color, expanded: false, children: [] };
    setFileTree(prev => [...prev, newFolder]);
    showToast('새 폴더가 추가되었습니다');
  }, [showToast]);

  // 사이드바 모드 전환 (file1.html 로직과 동일)
  const handleFolderMode = () => {
    if (isSidebarCollapsed) {
      setIsSidebarCollapsed(false);
      setSidebarMode('folder');
    } else if (sidebarMode === 'folder') {
      setIsSidebarCollapsed(true);
    } else {
      setSidebarMode('folder');
    }
  };

  const handleTransMode = () => {
    if (isSidebarCollapsed) {
      setIsSidebarCollapsed(false);
      setSidebarMode('transcription');
    } else if (sidebarMode === 'transcription') {
      setIsSidebarCollapsed(true);
    } else {
      setSidebarMode('transcription');
    }
  };

  // 검색 필터
  const allFlat = flattenAll(fileTree);
  const searchResults = searchQuery.trim()
    ? allFlat.filter(n => n.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : null;

  // 즐겨찾기 목록
  const favoriteNodes = allFlat.filter(n => favorites.has(n.id));

  // 활성 파일 노드
  const activeNode = findNode(activeFileId, fileTree);

  // 우클릭 타겟 노드
  const ctxTargetNode = ctxMenu.targetId ? findNode(ctxMenu.targetId, fileTree) : null;

  return (
    <>
      <aside
        id="sidebar"
        className={`flex flex-col h-full shrink-0 overflow-hidden transition-[width] duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]`}
        style={isSidebarCollapsed
          ? { width: '0px', minWidth: '0px', overflow: 'hidden', opacity: 0, pointerEvents: 'none' }
          : { width: `${width}px`, minWidth: `${width}px` }
        }
      >
        <div className="card flex-1 flex flex-col p-5 pb-5 overflow-hidden">
          {/* 헤더 */}
          <div className="flex items-center justify-between mb-5 sidebar-header-row">
            <div className="flex items-center gap-2 font-heavy-heading text-[18px]">
              <div className="w-[28px] h-[28px] bg-[#1d1d1f] rounded-[8px] flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-white text-[17px] scale-x-[-1]">menu_book</span>
              </div>
              <span className="collapsible-content">LectoAI</span>
            </div>
            <div className="flex gap-1 sidebar-header-btns">
              <button
                className={`btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93] sidebar-mode-btn ${sidebarMode === 'folder' ? 'active' : ''}`}
                onClick={handleFolderMode}
                title="폴더 구조"
              >
                <span className="material-symbols-outlined text-[20px]">folder</span>
              </button>
              <button
                className={`btn-ghost-icon p-1.5 rounded-lg text-[#8e8e93] sidebar-mode-btn ${sidebarMode === 'transcription' ? 'active' : ''}`}
                onClick={handleTransMode}
                title="전사 내용"
              >
                <span className="material-symbols-outlined text-[20px]">transcribe</span>
              </button>
            </div>
          </div>

          {/* ── 폴더 뷰 ── */}
          <div className={`flex flex-col flex-1 overflow-hidden ${sidebarMode === 'folder' ? '' : 'hidden'}`}>
            {/* 검색 */}
            <div className="sidebar-search-bg rounded-[12px] px-3.5 py-2 flex items-center gap-2 mb-5 collapsible-content">
              <span className="material-symbols-outlined text-[#8e8e93] text-[18px]">search</span>
              <input
                className="bg-transparent border-none focus:ring-0 p-0 text-[13px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
                placeholder="제목으로 검색"
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {/* 스크롤 컨텐츠 */}
            <div className="collapsible-content flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
              {/* 활성화한 파일 */}
              <section>
                <div className="flex items-center justify-between px-1 mb-2">
                  <span className="text-[13px] font-bold text-[#3a3a3c]">활성화한 파일</span>
                </div>
                <div id="active-file-display">
                  {activeNode && (
                    <div className="tree-item" style={{ backgroundColor: '#ffffff' }}>
                      <span className="material-symbols-outlined text-[#1d1d1f]" style={{ fontSize: '18px' }}>description</span>
                      <span className="text-[13px] font-semibold text-[#1d1d1f] flex-1 truncate">{activeNode.name}</span>
                    </div>
                  )}
                </div>
              </section>

              {/* 구조 (파일 트리) */}
              <section>
                <div className="flex items-center justify-between px-1 mb-2">
                  <span className="text-[13px] font-bold text-[#3a3a3c]">구조</span>
                  <div className="flex gap-1">
                    <button className="btn-ghost-icon p-0.5 rounded-lg" title="새 폴더" onClick={handleNewFolder}>
                      <span className="material-symbols-outlined text-[18px]" style={{ color: '#8e8e93' }}>create_new_folder</span>
                    </button>
                    <button className="btn-ghost-icon p-0.5 rounded-lg" title="새 파일" onClick={handleNewFile}>
                      <span className="material-symbols-outlined text-[18px]" style={{ color: '#8e8e93' }}>note_add</span>
                    </button>
                  </div>
                </div>
                <div id="file-tree">
                  {searchResults ? (
                    searchResults.length > 0 ? searchResults.map(n => (
                      <div
                        key={n.id}
                        className={`tree-item${n.id === activeFileId ? ' is-selected' : ''}`}
                        onClick={() => n.type === 'file' && handleSelectFile(n.id)}
                      >
                        <span
                          className="material-symbols-outlined"
                          style={{
                            color: n.color || '#8e8e93',
                            fontSize: '18px',
                            fontVariationSettings: n.type === 'folder' ? '"FILL" 1' : undefined,
                          }}
                        >
                          {n.type === 'folder' ? 'folder' : 'description'}
                        </span>
                        <span className="text-[13px] flex-1 truncate">{n.name}</span>
                      </div>
                    )) : (
                      <div className="text-[12px] text-[#aeaeb2] px-2 py-2">검색 결과 없음</div>
                    )
                  ) : (
                    fileTree.map(node => (
                      <TreeItem
                        key={node.id}
                        node={node}
                        depth={0}
                        activeFileId={activeFileId}
                        onSelectFile={handleSelectFile}
                        onToggleFolder={handleToggleFolder}
                        onShowContextMenu={handleShowContextMenu}
                      />
                    ))
                  )}
                </div>
              </section>

              {/* 즐겨찾기 */}
              <section>
                <div className="flex items-center justify-between px-1 mb-2">
                  <span className="text-[13px] font-bold text-[#3a3a3c]">즐겨찾기</span>
                </div>
                <div className="flex flex-col gap-0.5" id="favorites-list">
                  {favoriteNodes.map(n => (
                    <div key={n.id} className="tree-item" onClick={() => n.type === 'file' ? handleSelectFile(n.id) : handleToggleFolder(n.id)}>
                      <span
                        className="material-symbols-outlined"
                        style={{
                          color: n.color || '#8e8e93',
                          fontSize: '18px',
                          fontVariationSettings: n.type === 'folder' ? '"FILL" 1' : undefined,
                        }}
                      >
                        {n.type === 'folder' ? 'folder' : 'description'}
                      </span>
                      <span className="text-[13px] font-medium text-[#3a3a3c] flex-1 truncate">{n.name}</span>
                      {n.type === 'folder' && (
                        <span className="material-symbols-outlined text-[16px] text-[#8e8e93]">chevron_right</span>
                      )}
                    </div>
                  ))}
                </div>
              </section>

              {/* 리스트 (전체 평탄화) */}
              <section>
                <div className="flex items-center justify-between px-1 mb-2">
                  <span className="text-[13px] font-bold text-[#3a3a3c]">리스트</span>
                </div>
                <div className="flex flex-col gap-0.5 pb-4" id="all-list">
                  {allFlat.map(n => (
                    <div
                      key={n.id}
                      className="tree-item"
                      onClick={() => n.type === 'file' && handleSelectFile(n.id)}
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{
                          color: n.color || '#8e8e93',
                          fontSize: '18px',
                          fontVariationSettings: n.type === 'folder' ? '"FILL" 1' : undefined,
                        }}
                      >
                        {n.type === 'folder' ? 'folder' : 'description'}
                      </span>
                      <span className="text-[13px] text-[#3a3a3c] flex-1 truncate">{n.name}</span>
                      {n.type === 'folder' && (
                        <span className="material-symbols-outlined text-[16px] text-[#8e8e93]">chevron_right</span>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>

          {/* ── 전사 뷰 ── */}
          <div className={`flex flex-col flex-1 overflow-hidden ${sidebarMode === 'transcription' ? '' : 'hidden'}`}>
            {/* 검색 */}
            <div className="sidebar-search-bg rounded-[12px] px-3 py-1.5 flex items-center gap-2 mb-5">
              <span className="material-symbols-outlined text-[#8e8e93] text-[16px]">search</span>
              <input
                className="bg-transparent border-none focus:ring-0 p-0 text-[12px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
                placeholder="전사 내용 검색"
                type="text"
              />
            </div>
            {/* 전사 내용 또는 빈 상태 */}
            {(!transcriptions || transcriptions.length === 0) ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center px-4 opacity-40">
                <div className="w-16 h-16 rounded-full bg-[#f2f2f7] flex items-center justify-center mb-4">
                  <span className="material-symbols-outlined text-[32px] text-[#8e8e93]">mic_none</span>
                </div>
                <p className="text-[13px] text-[#1d1d1f] font-medium leading-relaxed">
                  음성 녹음을 시작하면<br />실시간 전사가 여기에 표시됩니다
                </p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-4 pr-1">
                {transcriptions.map((t, idx) => (
                  <div key={idx} className="flex flex-col gap-1.5 mt-2">
                    <span className="text-[11px] font-bold text-[#aeaeb2] px-1.5">{t.time}</span>
                    <div className="flex items-center gap-2 px-1.5 mb-1">
                      <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                        <span className="text-[10px] font-bold text-blue-600">나</span>
                      </div>
                      <span className="text-[11px] font-bold text-[#1d1d1f]">나</span>
                    </div>
                    <div className="message-bubble px-3.5 py-3 text-[13px] leading-[1.6]">
                      {t.text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 홈 버튼 */}
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

      {/* 좌측 리사이저 — collapsed 시 숨김 */}
      <div
        className="w-1.5 hover:bg-[#d1d1d6] transition-colors cursor-col-resize flex items-center justify-center group active:bg-[#aeaeb2] mx-[-6px] z-20"
        id="resizer-left"
        style={{ display: isSidebarCollapsed ? 'none' : undefined }}
        onMouseDown={handleResizerMouseDown}
      >
        <div className="w-0.5 h-8 bg-[#d1d1d6] rounded-full group-hover:bg-[#8e8e93]"></div>
      </div>

      {/* 드롭다운 메뉴 */}
      <ContextMenu
        visible={ctxMenu.visible}
        x={ctxMenu.x}
        y={ctxMenu.y}
        targetNode={ctxTargetNode}
        isFavorite={ctxMenu.targetId ? favorites.has(ctxMenu.targetId) : false}
        onClose={handleCloseContextMenu}
        onAction={handleContextAction}
      />

      {/* 토스트 */}
      <div className={`toast${toast ? ' show' : ''}`} id="toast">{toast}</div>

      {/* 외부 클릭으로 메뉴 닫기 */}
      {ctxMenu.visible && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
          onClick={handleCloseContextMenu}
        />
      )}
    </>
  );
}
