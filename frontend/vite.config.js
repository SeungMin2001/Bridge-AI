import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import net from 'net'

// 1. 127.0.0.1:8000이 살아있는지 확인하는 함수 (Ping 테스트)
const checkLocalBackend = () => {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    socket.setTimeout(500); // 0.5초 대기
    socket.on('connect', () => {
      socket.destroy();
      resolve(true); // 연결 성공 (로컬 백엔드 켜져있음)
    });
    socket.on('timeout', () => {
      socket.destroy();
      resolve(false); // 타임아웃
    });
    socket.on('error', () => {
      resolve(false); // 연결 실패 (로컬 백엔드가 꺼져있거나 Docker 내부임)
    });
    socket.connect(8000, '127.0.0.1');
  });
};

// 2. defineConfig를 비동기(async)로 변경하여 시작 시 검사
export default defineConfig(async () => {
  // 로컬 백엔드 상태 확인
  const isLocalAlive = await checkLocalBackend();

  // 원격 백엔드 fallback이 필요할 때 아래 코드를 다시 활성화하세요.
  // const checkRemote = () => new Promise(res => {
  //   const s = new net.Socket();
  //   s.setTimeout(500).on('connect', () => { s.destroy(); res(true); })
  //     .on('error', () => res(false)).on('timeout', () => res(false)).connect(8000, '100.104.164.84');
  // });

  // 127.0.0.1이 켜져있으면 우선 사용, 안 되면 환경변수(Docker) 사용
  const backendUrl = isLocalAlive
    ? 'http://127.0.0.1:8000'
    : (process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000');

  // 원격 백엔드 fallback 포함 버전:
  // const backendUrl = isLocalAlive
  //   ? 'http://127.0.0.1:8000'
  //   : process.env.VITE_BACKEND_URL
  //     ? process.env.VITE_BACKEND_URL
  //     : (await checkRemote())
  //       ? 'http://100.104.164.84:8000'
  //       : 'http://127.0.0.1:8000';

  const backendWsUrl = backendUrl.replace(/^http/, 'ws');

  // 터미널에서 현재 어떤 주소로 연결되었는지 확인하기 쉽게 로그 출력
  console.log(`\n🚀 [Vite Proxy] Backend URL automatically set to: ${backendUrl}\n`);

  return {
    plugins: [vue()],
    server: {
      host: true,
      proxy: {
        '/chat': {
          target: backendUrl,
          changeOrigin: true,
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
                proxyRes.headers['cache-control'] = 'no-cache'
                proxyRes.headers['x-accel-buffering'] = 'no'
              }
            })
          },
        },
        '/workspace': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/schedule': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/summary': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/quiz': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/ws': {
          target: backendWsUrl,
          ws: true,
        },
      },
    },
  };
})
