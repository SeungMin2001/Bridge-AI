import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const BACKEND_URL = 'http://100.104.164.84:8000'

export default defineConfig(() => {
  const backendUrl = BACKEND_URL

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
