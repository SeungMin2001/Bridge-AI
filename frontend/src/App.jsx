import React, { useState, useEffect, useRef, useCallback } from 'react';
import Workspace from './pages/Workspace/Workspace';
import Home from './pages/Home/Home';
import AiHistory from './pages/AiHistory/AiHistory';

export default function App() {
  // --- 상태 관리 (State) ---
  const [currentView, setCurrentView] = useState('home');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcriptions, setTranscriptions] = useState([]);
  const [activeFileName, setActiveFileName] = useState('강의1');
  const [isRightSidebarVisible, setIsRightSidebarVisible] = useState(true);

  // --- 통합 데이터 상태 (Home & Sidebar 공유) ---
  const [fileTree, setFileTree] = useState(() => {
    const saved = localStorage.getItem('lecto_file_tree');
    return saved ? JSON.parse(saved) : [];
  });
  const [favorites, setFavorites] = useState(() => {
    const saved = localStorage.getItem('lecto_favorites');
    return saved ? new Set(JSON.parse(saved)) : new Set();
  });

  // 데이터 자동 저장
  useEffect(() => {
    localStorage.setItem('lecto_file_tree', JSON.stringify(fileTree));
  }, [fileTree]);

  useEffect(() => {
    localStorage.setItem('lecto_favorites', JSON.stringify(Array.from(favorites)));
  }, [favorites]);

  // --- 참조 관리 (Refs) ---
  const timerRef = useRef(null);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const lastBubbleTimeRef = useRef(0);
  const mockTimersRef = useRef([]);

  const float32ToInt16 = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return int16Array;
  };

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    clearInterval(timerRef.current);

    // 모의 타이머 정리
    mockTimersRef.current.forEach(t => clearTimeout(t));
    mockTimersRef.current = [];

    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (sourceRef.current) { sourceRef.current.disconnect(); sourceRef.current = null; }
    if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
  }, []);

  const addTranscriptionBubble = useCallback((text) => {
    const now = new Date();
    const timeSpan = now.getTime() - lastBubbleTimeRef.current;

    setTranscriptions(prev => {
      const newItems = [...prev];
      if (newItems.length === 0 || timeSpan >= 3000) {
        newItems.push({
          time: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
          text: text
        });
      } else {
        newItems[newItems.length - 1].text += " " + text;
      }
      return newItems;
    });

    lastBubbleTimeRef.current = now.getTime();
  }, []);

  const startRecording = useCallback(async () => {
    setIsRecording(true);
    setRecordingSeconds(0);

    timerRef.current = setInterval(() => {
      setRecordingSeconds(s => s + 1);
    }, 1000);

    // --- Mock Data 설정 --- 
    // 나중에 백엔드 서버를 사용할 때는 이 값을 false로 바꾸면 됩니다.
    const USE_MOCK_DATA = false;

    if (USE_MOCK_DATA) {
      const t1 = setTimeout(() => {
        addTranscriptionBubble("안녕하세요, 실시간 음성 전사 테스트 중입니다.");
      }, 3000);
      const t2 = setTimeout(() => {
        addTranscriptionBubble("현재는 백엔드 연결 없이 샘플 데이터가 출력되고 있습니다.");
      }, 7000);
      mockTimersRef.current = [t1, t2];
      return;
    }

    // --- 실시간 백엔드 연결 (WebSocket) ---
    wsRef.current = new WebSocket("ws://100.104.164.84:8000/ws");

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.text && data.text.trim() !== "") {
          addTranscriptionBubble(data.text);
        }
      } catch (e) {
        console.log("Error parsing JSON:", e);
      }
    };

    wsRef.current.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        await audioContext.audioWorklet.addModule("/pcm-worklet.js");

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        const processor = new AudioWorkletNode(audioContext, "pcm-worklet");
        processorRef.current = processor;

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.port.onmessage = (event) => {
          const data = float32ToInt16(event.data);
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(data.buffer);
          }
        };
      } catch (error) {
        console.error("Error accessing microphone:", error);
        stopRecording();
      }
    };
  }, [addTranscriptionBubble, stopRecording]);

  useEffect(() => {
    return () => { stopRecording(); };
  }, [stopRecording]);

  const minutes = Math.floor(recordingSeconds / 60);
  const seconds = recordingSeconds % 60;
  const recordingTimeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;

  // 파일 선택 핸들러 (LeftSidebar -> MainContent 연결)
  const handleFileSelect = useCallback((id, node) => {
    if (node) setActiveFileName(node.name);
  }, []);

  // 우측 사이드바 토글
  const handleRightSidebarToggle = useCallback(() => {
    setIsRightSidebarVisible(prev => !prev);
  }, []);

  if (currentView === 'ai-history') {
    return (
      <AiHistory onNavigateBack={() => setCurrentView('home')} />
    );
  }

  if (currentView === 'home') {
    return (
      <Home
        onNavigate={(view) => setCurrentView(view)}
        fileTree={fileTree}
        setFileTree={setFileTree}
        favorites={favorites}
        setFavorites={setFavorites}
      />
    );
  }

  return (
    <Workspace
      onNavigateHome={() => setCurrentView('home')}
      transcriptions={transcriptions}
      onFileSelect={handleFileSelect}
      fileTree={fileTree}
      setFileTree={setFileTree}
      favorites={favorites}
      setFavorites={setFavorites}
      isRecording={isRecording}
      recordingTimeText={recordingTimeText}
      startRecording={startRecording}
      stopRecording={stopRecording}
      activeFileName={activeFileName}
      onRightSidebarToggle={handleRightSidebarToggle}
      isRightSidebarVisible={isRightSidebarVisible}
    />
  );
}
