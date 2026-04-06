import LeftSidebar from '../../components/workspace/LeftSidebar';
import MainContent from '../../components/workspace/MainContent';
import RightSidebar from '../../components/workspace/RightSidebar';

export default function Workspace({
  onNavigateHome,
  transcriptions,
  onFileSelect,
  fileTree,
  setFileTree,
  favorites,
  setFavorites,
  isRecording,
  recordingTimeText,
  startRecording,
  stopRecording,
  activeFileName,
  onRightSidebarToggle,
  isRightSidebarVisible
}) {
  return (
    <div className="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
      <LeftSidebar
        onNavigateHome={onNavigateHome}
        transcriptions={transcriptions}
        onFileSelect={onFileSelect}
        fileTree={fileTree}
        setFileTree={setFileTree}
        favorites={favorites}
        setFavorites={setFavorites}
      />
      <MainContent
        isRecording={isRecording}
        recordingTimeText={recordingTimeText}
        startRecording={startRecording}
        stopRecording={stopRecording}
        activeFileName={activeFileName}
        onRightSidebarToggle={onRightSidebarToggle}
      />
      <RightSidebar visible={isRightSidebarVisible} />
    </div>
  );
}
