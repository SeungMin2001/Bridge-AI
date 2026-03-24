"use client";

import { useState, useEffect, useRef } from "react";

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcriptions, setTranscriptions] = useState<{ id: string; text: string; timestamp: Date }[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const float32ToInt16 = (float32Array: Float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return int16Array;
  };

  const startRecording = async () => {
    setIsRecording(true);
    setRecordingSeconds(0);
    setTranscriptions([]);

    timerRef.current = setInterval(() => {
      setRecordingSeconds((prev) => prev + 1);
    }, 1000);

    const wsUrl = "ws://100.104.164.84:8000/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.text && data.text.trim() !== "") {
          setTranscriptions((prev) => {
            const now = new Date();
            // Group messages if they arrive within 3 seconds of each other
            if (prev.length > 0) {
              const last = prev[prev.length - 1];
              if (now.getTime() - last.timestamp.getTime() < 3000) {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...last,
                  text: last.text + " " + data.text,
                  timestamp: now,
                };
                return updated;
              }
            }
            return [
              ...prev,
              {
                id: Math.random().toString(36).substring(7),
                text: data.text,
                timestamp: now,
              },
            ];
          });
        }
      } catch (e) {
        console.error("Error parsing JSON:", e);
      }
    };

    ws.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const AudioContextConstructor = window.AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContextConstructor();
        audioContextRef.current = audioContext;

        await audioContext.audioWorklet.addModule("/pcm-worklet.js");

        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        const processor = new AudioWorkletNode(audioContext, "pcm-worklet");
        processorRef.current = processor;

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.port.onmessage = (event) => {
          const channelData: Float32Array = event.data;
          const int16Data = float32ToInt16(channelData);
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(int16Data.buffer);
          }
        };
      } catch (error) {
        console.error("Error accessing microphone:", error);
        stopRecording();
      }
    };
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);

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
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  return { isRecording, recordingSeconds, transcriptions, startRecording, stopRecording };
}
