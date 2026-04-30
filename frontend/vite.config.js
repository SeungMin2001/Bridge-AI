import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,
    proxy: {
      '/chat': {
        target: 'http://100.104.164.84:8000',
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
        target: 'http://127.0.0.1:8001', // 8001 -> main.py : 8000
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8001', // 8001 -> main.py : 8000
        ws: true,
      },
    },
  },
})
