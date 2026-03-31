<template>
  <div class="home-banner shrink-0">
    <div class="absolute w-[140px] h-[140px] bg-[#2d2b3e] rounded-full top-[10px] left-[40%]"></div>
    <div class="absolute w-[220px] h-[220px] bg-[#2d2b3e] rounded-full bottom-[-60px] left-[10%]"></div>
    
    <div class="relative z-10 max-w-[480px] flex flex-col gap-3 items-start">
      <div class="text-[36px] font-bold text-white tracking-[-0.03em] leading-[1.1]">하이</div>
      <div class="text-[15px] text-white/80 leading-[1.6] font-medium">
        Learn fun anywhere and anytime without any time limit just through the application.
      </div>
    </div>

    <!-- Centered Button -->
    <div class="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
      <button class="bg-white text-[#373549] border-none rounded-full px-7 py-3.5 text-[14px] font-bold cursor-pointer transition-opacity hover:opacity-90 pointer-events-auto shadow-lg">
        Get Started
      </button>
    </div>
    
    <div class="relative z-10 shrink-0 w-[240px] h-[140px] flex items-end justify-center">
      <svg fill="none" height="180" style="position: absolute; bottom: -20px; right: -20px" viewBox="0 0 240 180" width="240" xmlns="http://www.w3.org/2000/svg">
        <path d="M40 50 h12 v-12 h8 v12 h12 v8 h-12 v12 h-8 v-12 h-12 z" fill="#fff" opacity="0.9" transform="rotate(-15 50 50)"></path>
        <path d="M50 100 h20 v6 h-20 z" fill="#fff" opacity="0.9" transform="rotate(10 60 100)"></path>
        <g transform="translate(180, 20) rotate(15)">
          <path d="M15 0 C6.7 0 0 6.7 0 15 C0 20.3 2.7 25 6.7 27.8 L6.7 33.3 C6.7 34.2 7.5 35 8.3 35 L21.7 35 C22.6 35 23.3 34.2 23.3 33.3 L23.3 27.8 C27.3 25 30 20.3 30 15 Z" fill="none" stroke="#fff" stroke-width="2.5"></path>
          <path d="M10 40 h10 M12 45 h6" stroke="#fff" stroke-linecap="round" stroke-width="2.5"></path>
          <path d="M15 15 v10" stroke="#fff" stroke-linecap="round" stroke-width="2.5"></path>
        </g>
        <circle cx="140" cy="85" fill="#2d2b3e" r="2.5"></circle>
        <circle cx="160" cy="85" fill="#2d2b3e" r="2.5"></circle>
        <path d="M145 95 Q150 100 155 95" fill="none" stroke="#2d2b3e" stroke-linecap="round" stroke-width="2"></path>
        <path d="M110 180 C110 130 190 130 190 180 Z" fill="#2d2b3e"></path>
        <path d="M90 160 C110 145 125 155 135 165" fill="none" stroke="#fff" stroke-linecap="round" stroke-width="12"></path>
        <path d="M210 160 C190 145 175 155 165 165" fill="none" stroke="#fff" stroke-linecap="round" stroke-width="12"></path>
        <path d="M85 140 L150 165 L150 200 L85 175 Z" fill="#fff" stroke="#2d2b3e" stroke-linejoin="round" stroke-width="2"></path>
        <path d="M215 140 L150 165 L150 200 L215 175 Z" fill="#f4f4f5" stroke="#2d2b3e" stroke-linejoin="round" stroke-width="2"></path>
        <path d="M95 150 L140 168 M95 158 L140 176" stroke="#2d2b3e" stroke-linecap="round" stroke-width="2"></path>
        <path d="M205 150 L160 168 M205 158 L160 176" stroke="#2d2b3e" stroke-linecap="round" stroke-width="2"></path>
      </svg>
    </div>
  </div>
</template>
