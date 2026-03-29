import React from 'react';

export default function HomeModals({
  isFolderOpen,
  setIsFolderOpen,
  isFileOpen,
  setIsFileOpen,
  newFolderName,
  setNewFolderName,
  newFileName,
  setNewFileName,
  selectedColor,
  setSelectedColor,
  onCreateFolder,
  onCreateFile
}) {
  return (
    <>
      {/* Folder Creation Modal */}
      <div className={`modal-overlay ${isFolderOpen ? 'open' : ''}`}>
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
            <button className="modal-btn-secondary" onClick={() => { setIsFolderOpen(false); setNewFolderName(''); }}>취소</button>
            <button className="modal-btn-primary" onClick={onCreateFolder}>추가</button>
          </div>
        </div>
      </div>

      {/* File Creation Modal */}
      <div className={`modal-overlay ${isFileOpen ? 'open' : ''}`}>
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
            <button className="modal-btn-secondary" onClick={() => { setIsFileOpen(false); setNewFileName(''); }}>취소</button>
            <button className="modal-btn-primary" onClick={onCreateFile}>생성</button>
          </div>
        </div>
      </div>
    </>
  );
}
