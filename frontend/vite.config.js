import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

const BACKEND_URL = 'http://100.104.164.84:8000'
const FRONTEND_ROOT = fileURLToPath(new URL('.', import.meta.url))
const FRONTEND_HOST = '127.0.0.1'
const FRONTEND_PORT = 5173

function proxyConfig(backendUrl, backendWsUrl) {
  return {
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
  }
}

export default defineConfig(() => {
  const backendUrl = BACKEND_URL

  const backendWsUrl = backendUrl.replace(/^http/, 'ws');

  // 터미널에서 현재 어떤 주소로 연결되었는지 확인하기 쉽게 로그 출력
  console.log(`\n🚀 [Vite Proxy] Backend URL automatically set to: ${backendUrl}\n`);

  return {
    root: FRONTEND_ROOT,
    cacheDir: 'node_modules/.vite-cache',
    plugins: [vue()],
    optimizeDeps: {
      entries: ['index.html'],
      include: [
        'vue',
        'marked',
        'chart.js',
        'jszip',
        'lottie-web/build/player/lottie_light',
      ],
    },
    server: {
      host: FRONTEND_HOST,
      port: FRONTEND_PORT,
      strictPort: true,
      hmr: false,
      watch: {
        ignored: [
          '**/*',
        ],
      },
      proxy: proxyConfig(backendUrl, backendWsUrl),
    },
    preview: {
      host: FRONTEND_HOST,
      port: FRONTEND_PORT,
      strictPort: true,
      proxy: proxyConfig(backendUrl, backendWsUrl),
    },
  };
})
