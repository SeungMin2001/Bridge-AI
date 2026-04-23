import React from 'react';
import './Home.css';
import HomeSidebar from '../../components/home/HomeSidebar';
import HomeBanner from '../../components/home/HomeBanner';
import HomeGrid from '../../components/home/HomeGrid';
import AiAssistant from '../../components/home/AiAssistant';
import HomeModals from '../../components/home/HomeModals';
import useHome from './hooks/useHome';

export default function Home({ onNavigate, fileTree, setFileTree, favorites, setFavorites }) {
  const home = useHome({ fileTree, setFileTree, favorites, setFavorites });

  return (
    <div className="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
      <HomeSidebar
        isCollapsed={home.isSidebarCollapsed}
        setIsCollapsed={home.setIsSidebarCollapsed}
        onNavigate={onNavigate}
        fileTree={fileTree}
        favorites={favorites}
      />

      <main id="home-main-content" className="custom-scrollbar flex-1 overflow-y-auto">
        <HomeBanner />
        
        <HomeGrid
          items={home.currentItems}
          favorites={favorites}
          toggleStar={home.toggleStar}
          onEnterFolder={home.handleEnterFolder}
          onNavigate={onNavigate}
          navigationStack={home.navigationStack}
          onGoBack={home.handleGoBack}
          title={home.currentTitle}
          onOpenFolderModal={() => home.setIsFolderModalOpen(true)}
          onOpenFileModal={() => home.setIsFileModalOpen(true)}
        />
      </main>

      <AiAssistant
        isOpen={home.isAiChatOpen}
        setIsOpen={home.setIsAiChatOpen}
        aiWinRef={home.aiWinRef}
        aiBtnRef={home.aiBtnRef}
      />

      <HomeModals
        isFolderOpen={home.isFolderModalOpen}
        setIsFolderOpen={home.setIsFolderModalOpen}
        isFileOpen={home.isFileModalOpen}
        setIsFileOpen={home.setIsFileModalOpen}
        newFolderName={home.newFolderName}
        setNewFolderName={home.setNewFolderName}
        newFileName={home.newFileName}
        setNewFileName={home.setNewFileName}
        selectedColor={home.selectedColor}
        setSelectedColor={home.setSelectedColor}
        onCreateFolder={home.handleCreateFolder}
        onCreateFile={home.handleCreateFile}
      />
    </div>
  );
}
