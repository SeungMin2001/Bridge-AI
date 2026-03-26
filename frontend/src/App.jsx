import React, { useState, useEffect, useRef, useCallback } from 'react';
import LeftSidebar from './layout/LeftSidebar';
import MainContent from './layout/MainContent';
import RightSidebar from './layout/RightSidebar';
import Home from './pages/Home';

export default function App() {
  // --- 상태 관리 (State) ---
  const [currentView, setCurrentView] = useState('home'); // 현재 화면 보기 상태 ('home' 또는 'workspace')
  const [isRecording, setIsRecording] = useState(false); // 녹음 진행 여부
  const [recordingSeconds, setRecordingSeconds] = useState(0); // 녹음 진행 시간 (초)
  const [transcriptions, setTranscriptions] = useState([]); // 실시간으로 받아온 변환 텍스트 목록

  // --- 참조 관리 (Refs) ---
  const timerRef = useRef(null); // 타이머 인터벌 참조
  const wsRef = useRef(null); // 웹소켓 연결 참조
  const audioContextRef = useRef(null); // Web Audio API 컨텍스트
  const streamRef = useRef(null); // 미디어 스트림 (마이크 입력)
  const processorRef = useRef(null); // AudioWorklet 노드
  const sourceRef = useRef(null); // 오디오 소스 노드
  const lastBubbleTimeRef = useRef(0); // 마지막으로 텍스트 버블이 생성된 시간

  /**
   * 오디오 데이터 형식을 Float32에서 Int16으로 변환하는 함수
   * 서버(STT 엔진)에서 요구하는 데이터 형식에 맞추기 위해 사용됨
   */
  const float32ToInt16 = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return int16Array;
  };

  /**
   * 녹음을 중단하고 동기화된 모든 리소스를 정리하는 함수
   */
  const stopRecording = useCallback(() => {
    setIsRecording(false);
    clearInterval(timerRef.current);

    // 오디오 처리 노드 및 컨텍스트 정리
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    // 웹소켓 연결 종료
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  /**
   * 실시간 변환된 텍스트를 화면에 추가하는 함수
   * 3초 이내에 들어오는 텍스트는 기존 버블에 이어 붙이고, 그 이상은 새로운 버블 생성
   */
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

  /**
   * 녹음을 시작하고 웹소켓을 연결하여 오디오 데이터를 스트리밍하는 함수
   */
  const startRecording = useCallback(async () => {
    setIsRecording(true);
    setRecordingSeconds(0);

    // 녹음 시간 카운터 시작
    timerRef.current = setInterval(() => {
      setRecordingSeconds(s => s + 1);
    }, 1000);

    // 서버와 웹소켓 연결
    wsRef.current = new WebSocket("ws://100.104.164.84:8000/ws");

    // 메시지 수신 핸들러 (서버에서 변환된 텍스트 수신)
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

    // 웹소켓 연결 완료 후 마이크 권한 요청 및 오디오 스트리밍 설정
    wsRef.current.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        // PCM 오디오 처리를 위한 AudioWorklet 등록
        await audioContext.audioWorklet.addModule("/pcm-worklet.js");

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        const processor = new AudioWorkletNode(audioContext, "pcm-worklet");
        processorRef.current = processor;

        source.connect(processor);
        processor.connect(audioContext.destination);

        // AudioWorklet에서 넘어온 오디오 조각을 서버로 전송
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

  // 컴포넌트 언마운트 시 녹음 정리
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  // 초 단위를 MM:SS 형식의 텍스트로 변환
  const minutes = Math.floor(recordingSeconds / 60);
  const seconds = recordingSeconds % 60;
  const recordingTimeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;

  // 초기 홈 화면 렌더링
  if (currentView === 'home') {
    return <Home onNavigate={(view) => setCurrentView(view)} />;
  }

  // 메인 워크스페이스 대시보드 구조 렌더링
  return (
    <div className="p-[12px] flex gap-[12px] relative h-full w-full bg-[#ebebf0] text-[#1d1d1f] overflow-hidden">
      <LeftSidebar onNavigateHome={() => setCurrentView('home')} />
      <MainContent
        isRecording={isRecording}
        recordingTimeText={recordingTimeText}
        startRecording={startRecording}
        stopRecording={stopRecording}
      />
      <RightSidebar transcriptions={transcriptions} />
    </div>
  );
}
