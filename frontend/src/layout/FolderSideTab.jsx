import React, { useState, useRef, useCallback } from 'react';

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

const FOLDER_COLORS = ['#3b82f6', '#5856d6', '#ff9500', '#34c759', '#ff3b30', '#af52de'];

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
        <span className={`text-[13px] flex-1 truncate ${isSelected ? 'font-semibold text-[#1d1d1f]' : 'font-medium text-[#3a3a3c]'}`}>
          {node.name}
        </span>
        {!isFile && (
          <span
            className={`material-symbols-outlined chevron${node.expanded ? ' open' : ''}`}
            style={{ fontSize: '16px' }}
          >
            chevron_right
          </span>
        )}
        <button className="dots-btn" title="더 보기" onClick={handleDotsClick}>
          <span className="material-symbols-outlined" style={{ fontSize: '16px', pointerEvents: 'none' }}>more_horiz</span>
        </button>
      </div>
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

// ── ContextMenu 컴포넌트 ──
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

export default function FolderSideTab({ 
  fileTree, 
  setFileTree, 
  favorites, 
  setFavorites, 
  onFileSelect,
  showToast 
}) {
  const [activeFileId, setActiveFileId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [ctxMenu, setCtxMenu] = useState({ visible: false, x: 0, y: 0, targetId: null });
  const colorIdxRef = useRef(0);

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
  }, [setFileTree]);

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
          if (node.type === 'file') handleSelectFile(targetId);
          else node.expanded = true;
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
          break;
      }
      return copy;
    });
  }, [ctxMenu, handleSelectFile, setFileTree, setFavorites, showToast]);

  const handleNewFile = () => {
    const newFile = { id: genId(), type: 'file', name: '새 파일', content: '' };
    setFileTree(prev => [...prev, newFile]);
    showToast('새 파일이 추가되었습니다');
  };

  const handleNewFolder = () => {
    const color = FOLDER_COLORS[colorIdxRef.current++ % FOLDER_COLORS.length];
    const newFolder = { id: genId(), type: 'folder', name: '새 폴더', color, expanded: false, children: [] };
    setFileTree(prev => [...prev, newFolder]);
    showToast('새 폴더가 추가되었습니다');
  };

  const allFlat = flattenAll(fileTree);
  const searchResults = searchQuery.trim() ? allFlat.filter(n => n.name.toLowerCase().includes(searchQuery.toLowerCase())) : null;
  const favoriteNodes = allFlat.filter(n => favorites.has(n.id));
  const activeNode = findNode(activeFileId, fileTree);
  const ctxTargetNode = ctxMenu.targetId ? findNode(ctxMenu.targetId, fileTree) : null;

  return (
    <>
      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="sidebar-search-bg rounded-[12px] px-3.5 py-2 flex items-center gap-2 mb-5">
          <span className="material-symbols-outlined text-[#8e8e93] text-[18px]">search</span>
          <input
            className="bg-transparent border-none focus:ring-0 p-0 text-[13px] text-[#1d1d1f] placeholder-[#aeaeb2] w-full"
            placeholder="제목으로 검색"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">활성화한 파일</span>
            </div>
            {activeNode && (
              <div className="tree-item" style={{ backgroundColor: '#ffffff' }}>
                <span className="material-symbols-outlined text-[#1d1d1f]" style={{ fontSize: '18px' }}>description</span>
                <span className="text-[13px] font-semibold text-[#1d1d1f] flex-1 truncate">{activeNode.name}</span>
              </div>
            )}
          </section>

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
                  <div key={n.id} className={`tree-item${n.id === activeFileId ? ' is-selected' : ''}`} onClick={() => n.type === 'file' && handleSelectFile(n.id)}>
                    <span className="material-symbols-outlined" style={{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? "'FILL' 1" : "'FILL' 0" }}>
                      {n.type === 'folder' ? 'folder' : 'description'}
                    </span>
                    <span className="text-[13px] flex-1 truncate">{n.name}</span>
                  </div>
                )) : <div className="text-[12px] text-[#aeaeb2] px-2 py-2">검색 결과 없음</div>
              ) : fileTree.map(node => (
                <TreeItem key={node.id} node={node} depth={0} activeFileId={activeFileId} onSelectFile={handleSelectFile} onToggleFolder={handleToggleFolder} onShowContextMenu={handleShowContextMenu} />
              ))}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">즐겨찾기</span>
            </div>
            <div className="flex flex-col gap-0.5">
              {favoriteNodes.map(n => (
                <div key={n.id} className="tree-item" onClick={() => n.type === 'file' ? handleSelectFile(n.id) : handleToggleFolder(n.id)}>
                  <span className="material-symbols-outlined" style={{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? "'FILL' 1" : "'FILL' 0" }}>
                    {n.type === 'folder' ? 'folder' : 'description'}
                  </span>
                  <span className="text-[13px] font-medium text-[#3a3a3c] flex-1 truncate">{n.name}</span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between px-1 mb-2">
              <span className="text-[13px] font-bold text-[#3a3a3c]">리스트</span>
            </div>
            <div className="flex flex-col gap-0.5 pb-4">
              {allFlat.map(n => (
                <div key={n.id} className="tree-item" onClick={() => n.type === 'file' && handleSelectFile(n.id)}>
                  <span className="material-symbols-outlined" style={{ color: n.color || '#8e8e93', fontSize: '18px', fontVariationSettings: n.type === 'folder' ? "'FILL' 1" : "'FILL' 0" }}>
                    {n.type === 'folder' ? 'folder' : 'description'}
                  </span>
                  <span className="text-[13px] text-[#3a3a3c] flex-1 truncate">{n.name}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>

      <ContextMenu visible={ctxMenu.visible} x={ctxMenu.x} y={ctxMenu.y} targetNode={ctxTargetNode} isFavorite={ctxMenu.targetId ? favorites.has(ctxMenu.targetId) : false} onClose={handleCloseContextMenu} onAction={handleContextAction} />
      {ctxMenu.visible && <div style={{ position: 'fixed', inset: 0, zIndex: 9998 }} onClick={handleCloseContextMenu} />}
    </>
  );
}
