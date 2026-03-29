import { useState, useEffect, useRef } from 'react';

export default function useHome({ fileTree, setFileTree, favorites, setFavorites }) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isAiChatOpen, setIsAiChatOpen] = useState(false);
  const [isFolderModalOpen, setIsFolderModalOpen] = useState(false);
  const [isFileModalOpen, setIsFileModalOpen] = useState(false);
  const [selectedColor, setSelectedColor] = useState('#3b82f6');
  const [navigationStack, setNavigationStack] = useState([]); // [{id, name}]

  const aiWinRef = useRef(null);
  const aiBtnRef = useRef(null);

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
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(targetId)) next.delete(targetId);
      else next.add(targetId);
      return next;
    });
  };

  const addItemToTree = (nodes, parentId, newItem) => {
    if (!parentId) return [newItem, ...nodes];
    return nodes.map(node => {
      if (node.id === parentId) {
        return { ...node, children: [newItem, ...(node.children || [])] };
      }
      if (node.children) {
        return { ...node, children: addItemToTree(node.children, parentId, newItem) };
      }
      return node;
    });
  };

  const handleCreateFolder = () => {
    if (!newFolderName.trim()) return;
    const now = new Date();
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`;

    const newFolder = {
      id: 'f' + Date.now(),
      type: 'folder',
      name: newFolderName,
      date: timeStr,
      color: selectedColor,
      expanded: false,
      children: []
    };

    const currentFolderId = navigationStack.length > 0 ? navigationStack[navigationStack.length - 1].id : null;
    setFileTree(addItemToTree(fileTree, currentFolderId, newFolder));
    setIsFolderModalOpen(false);
    setNewFolderName('');
  };

  const handleCreateFile = () => {
    if (!newFileName.trim()) return;
    const now = new Date();
    const timeStr = `${now.getFullYear()}. ${now.getMonth() + 1}. ${now.getDate()}. ${now.getHours() >= 12 ? '오후' : '오전'} ${now.getHours() % 12 || 12}:${now.getMinutes().toString().padStart(2, '0')}`;

    const newFile = {
      id: 'file-' + Date.now(),
      type: 'file',
      name: newFileName,
      date: timeStr,
      color: '#1d1d1f'
    };

    const currentFolderId = navigationStack.length > 0 ? navigationStack[navigationStack.length - 1].id : null;
    setFileTree(addItemToTree(fileTree, currentFolderId, newFile));
    setIsFileModalOpen(false);
    setNewFileName('');
  };

  const getCurrentItems = () => {
    if (navigationStack.length === 0) return fileTree;
    
    const currentFolderId = navigationStack[navigationStack.length - 1].id;
    const findFolder = (nodes, id) => {
      for (const node of nodes) {
        if (node.id === id) return node;
        if (node.children) {
          const found = findFolder(node.children, id);
          if (found) return found;
        }
      }
      return null;
    };
    const folder = findFolder(fileTree, currentFolderId);
    return folder ? (folder.children || []) : [];
  };

  const handleEnterFolder = (e, item) => {
    e.stopPropagation();
    setNavigationStack([...navigationStack, { id: item.id, name: item.name }]);
  };

  const handleGoBack = () => {
    setNavigationStack(navigationStack.slice(0, -1));
  };

  const currentItems = getCurrentItems();
  const currentTitle = navigationStack.length > 0 
    ? navigationStack[navigationStack.length - 1].name 
    : '내 폴더';

  return {
    isSidebarCollapsed, setIsSidebarCollapsed,
    isAiChatOpen, setIsAiChatOpen,
    isFolderModalOpen, setIsFolderModalOpen,
    isFileModalOpen, setIsFileModalOpen,
    selectedColor, setSelectedColor,
    navigationStack, setNavigationStack,
    aiWinRef, aiBtnRef,
    newFolderName, setNewFolderName,
    newFileName, setNewFileName,
    toggleStar,
    handleCreateFolder,
    handleCreateFile,
    handleEnterFolder,
    handleGoBack,
    currentItems,
    currentTitle
  };
}
