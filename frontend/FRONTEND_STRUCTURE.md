# Frontend Structure Guide

이 문서는 `frontend/` 폴더 안의 주요 폴더와 파일이 어떤 역할을 하는지 빠르게 파악하기 위한 정리 문서입니다.

## 1. 전체 개요

이 프론트엔드는 다음 흐름으로 동작합니다.

1. 브라우저가 `frontend/index.html`을 연다.
2. `frontend/src/main.js`가 Vue 앱을 시작한다.
3. `frontend/src/App.vue`가 루트 컴포넌트로 올라간다.
4. `usePageNavigation.js`가 현재 해시 주소를 읽어서 어떤 페이지를 보여줄지 결정한다.
5. `pages/` 아래의 페이지 컴포넌트가 `components/`와 `composables/`를 조합해 실제 화면을 렌더링한다.

## 2. 폴더 구조 요약

```text
frontend/
  index.html
  package.json
  vite.config.js
  public/
  src/
    App.vue
    main.js
    index.css
    pages/
    components/
    composables/
    assets/styles/
    test_features/
```

## 3. 루트 파일

### 실행/설정 파일

- `index.html`
  - Vite가 브라우저에 처음 제공하는 HTML 엔트리 파일입니다.
  - `#root` DOM을 만들고 `/src/main.js`를 불러옵니다.
  - Tailwind CDN과 Material Symbols 폰트를 로드합니다.

- `package.json`
  - 프론트엔드 프로젝트의 패키지 정보와 실행 스크립트가 들어 있습니다.
  - `dev`, `build`, `preview`, `lint` 명령의 기준 파일입니다.

- `package-lock.json`
  - 설치된 npm 의존성 버전을 고정하는 잠금 파일입니다.

- `vite.config.js`
  - Vite 설정 파일입니다.
  - Vue 플러그인을 등록하고, `/chat` 요청을 백엔드로 프록시하도록 설정합니다.

### 루트의 보조/중복 파일

- `main.js`
  - `src/main.js`와는 별개의 루트 파일입니다.
  - 현재 Vite 실행 엔트리는 `src/main.js`이므로 운영 기준 핵심 엔트리로는 보이지 않습니다.

- `pcm-worklet.js`
  - 오디오 처리용 워크릿 스크립트입니다.
  - 실제 앱에서는 `public/pcm-worklet.js` 경로로도 사용됩니다.

- `file1.html`, `file2.html`
  - 실험용 또는 임시 산출물로 보이는 HTML 파일입니다.
  - 현재 메인 앱 구동 흐름에는 직접 연결되어 있지 않습니다.

- `file1 2.html`, `file2 2.html`, `main 2.js`, `pcm-worklet 2.js`
  - 이름상 복사본 또는 백업본으로 보이는 파일들입니다.
  - 운영 중인 메인 경로와 직접 연결되지는 않습니다.

- `.DS_Store`
  - macOS가 만든 시스템 파일입니다.

## 4. public 폴더

`public/`은 번들링 과정 없이 정적 경로 그대로 제공되는 자원을 둡니다.

- `public/pcm-worklet.js`
  - 녹음/오디오 처리 시 브라우저에서 직접 로드하는 오디오 워크릿 파일입니다.

- `public/images/banner_illust.png`
  - 홈 화면 또는 소개성 UI에서 사용하는 이미지 자원입니다.

## 5. src 폴더

`src/`는 실제 Vue 앱의 핵심 코드가 들어 있는 메인 영역입니다.

### 핵심 엔트리

- `src/main.js`
  - `createApp(App).mount('#root')`를 통해 Vue 앱을 시작합니다.
  - `src/index.css`도 여기서 전역 로드합니다.

- `src/App.vue`
  - 앱의 루트 컴포넌트입니다.
  - `Home`, `Workspace`, `Workfolder`, `AiHistory` 페이지를 상황에 따라 렌더링합니다.
  - `useAppState()`로 전역 성격의 앱 상태를 받고, `usePageNavigation()`으로 현재 페이지를 판단합니다.

- `src/index.css`
  - 전역 CSS 진입 파일입니다.
  - 공통 스타일, 애니메이션, 사이드바, 홈/워크스페이스 전용 CSS를 `@import`로 불러옵니다.

## 6. pages 폴더

`pages/`는 화면 단위의 상위 컴포넌트 모음입니다.

### `src/pages/AiHistory/`

- `AiHistory.vue`
  - 과거 AI 요청 기록을 보여주는 히스토리 페이지입니다.

### `src/pages/Home/`

- `Home.vue`
  - 메인 홈 대시보드 페이지입니다.
  - `HomeSidebar`, `HomeBanner`, `HomeRightSidebar`, `InfiniteGrid`를 조합합니다.

- `Home.css`
  - 홈 페이지 관련 스타일 파일입니다.
  - 현재는 전역/컴포넌트 클래스 중심 구조와 병행되는 페이지 스타일 리소스입니다.

### `src/pages/Workfolder/`

- `Workfolder.vue`
  - 폴더/파일 탐색 전용 페이지입니다.
  - `useHome()` composable과 `HomeGrid`, `HomeModals`를 사용합니다.

- `Workfolder.css`
  - 워크폴더 화면 전용 스타일 파일입니다.
  - `Workfolder.vue`에서 `style src="./Workfolder.css"` 방식으로 불러옵니다.

### `src/pages/Workspace/`

- `Workspace.vue`
  - 녹음, 실시간 전사, 자료 미리보기, AI 보조 기능이 모인 작업실 페이지입니다.
  - `LeftSidebar`, `MainContent`, `RightSidebar`를 조합합니다.

## 7. components 폴더

`components/`는 페이지를 이루는 재사용 가능한 화면 조각들입니다.

### `src/components/home/`

- `AiAssistant.vue`
  - 홈 계열에서 쓰이는 AI 비서 팝업 컴포넌트입니다.

- `HomeBanner.vue`
  - 홈 중앙의 대표 배너/환영 메시지/메인 인터랙션 영역입니다.
  - `MultimodalInput`을 포함합니다.

- `HomeGrid.vue`
  - 워크폴더 화면에서 폴더/파일 카드를 그리드로 보여줍니다.

- `HomeModals.vue`
  - 폴더 생성, 파일 생성 같은 모달 창들을 담당합니다.

- `HomeRightSidebar.vue`
  - 홈 화면 오른쪽 참고 패널입니다.
  - 자료나 AI 분석 결과를 보여주는 용도입니다.

- `HomeSidebar.vue`
  - 홈 왼쪽 사이드바입니다.
  - 메뉴, 캘린더, 네비게이션 역할을 합니다.

- `InfiniteGrid.vue`
  - 배경용 그리드 효과를 그리는 장식용 컴포넌트입니다.

- `MultimodalInput.vue`
  - 텍스트/이미지 등 복합 입력을 받는 입력 UI입니다.

### `src/components/ui/`

- `AnimatedTabs.vue`
  - 탭 전환 애니메이션을 제공하는 공통 UI 컴포넌트입니다.

### `src/components/workspace/`

- `ChatHistoryModal.vue`
  - 워크스페이스 내 AI 대화 기록 확인용 모달입니다.

- `FolderSideTab.vue`
  - 왼쪽 사이드바 안에서 파일 트리를 보여주는 탭입니다.

- `LeftSidebar.vue`
  - 워크스페이스 왼쪽 사이드바 본체입니다.
  - `FolderSideTab`, `VoiceTransferSideTab` 전환을 관리합니다.

- `MainContent.vue`
  - 워크스페이스 중앙 메인 영역입니다.
  - 헤더, 자료 리스트, 미리보기, 단어 카드 등 여러 하위 컴포넌트를 조합합니다.

- `RightSidebar.vue`
  - 워크스페이스 오른쪽 AI 채팅 패널입니다.

- `VoiceTransferSideTab.vue`
  - 실시간 전사 결과를 보여주고 AI 질문/노트 추가를 연결하는 탭입니다.

### `src/components/workspace/MainContent/`

- `LectureMaterialList.vue`
  - 업로드된 강의 자료 목록을 표시합니다.

- `LecturePreviewPanel.vue`
  - PDF/PPT 강의 자료 미리보기 패널입니다.

- `PptPreviewToolbar.vue`
  - PPT 미리보기에서 슬라이드 이동을 담당하는 툴바입니다.

- `WorkspaceFloatingTabs.vue`
  - 워크스페이스 중앙 하단의 플로팅 탭 전환 바입니다.

- `WorkspaceHeader.vue`
  - 녹음 제어, 자료 업로드, 사이드바 토글을 담당하는 헤더입니다.

- `WorkspaceWordCard.vue`
  - 선택된 단어 설명과 빠른 액션을 보여주는 카드입니다.

### 기타

- `src/components/common/`
  - 현재 기준 눈에 띄는 파일은 없고, 공통 컴포넌트용 폴더로 보입니다.

- `src/components/.DS_Store`
  - macOS 시스템 파일입니다.

## 8. composables 폴더

`composables/`는 Vue Composition API 기반의 상태/로직 재사용 함수들을 둡니다.

### 루트 composables

- `useAppState.js`
  - 앱 상태를 묶어서 반환하는 상위 조립용 composable입니다.
  - 내부적으로 `appState/` 아래 모듈들을 조합합니다.

- `useChat.js`
  - AI 메시지, 근거 팝오버, 단어 선택 정보 등 채팅 관련 상태를 관리합니다.

- `useHome.js`
  - 워크폴더 탐색, 폴더/파일 생성, 별표 처리, 이동 스택 등 홈/폴더 관련 로직을 담당합니다.

- `usePageNavigation.js`
  - 해시 라우팅 기반 페이지 이동을 관리합니다.
  - `#/`, `#/workspace`, `#/workfolder`, `#/ai-history` 경로를 해석합니다.

### `src/composables/appState/`

- `aiState.js`
  - AI 입력값, 요약 노트, 오른쪽 사이드바 토글을 관리합니다.

- `fileTreeState.js`
  - 파일 트리, 즐겨찾기, 선택 파일, localStorage 저장/복원을 담당합니다.

- `materialsState.js`
  - 강의 자료 업로드, 현재 미리보기 자료, 저장된 자료 열기/삭제를 담당합니다.

- `recordingState.js`
  - 녹음 상태, 전사 데이터, WebSocket/오디오 워크릿 처리 로직을 담당합니다.

## 9. assets/styles 폴더

`src/assets/styles/`는 CSS를 목적별로 나눈 전역 스타일 모음입니다.

### `src/assets/styles/common/`

- `animations.css`
  - 전사 등장, 팝오버, 사이드바, 탭, 녹음 파동 등 공통 애니메이션을 모아둔 파일입니다.

- `sidebar.css`
  - 공통 사이드바 레이아웃, 검색, 네비게이션, 트리 스타일을 정의합니다.

- `ui-elements.css`
  - 카드, 버튼, 스크롤바, 드롭다운, 토스트 등 공통 UI 스타일을 정의합니다.

### `src/assets/styles/home/`

- `home-sidebar.css`
  - 홈 화면 사이드바 전용 스타일입니다.

### `src/assets/styles/workspace/`

- `workspace-layout.css`
  - 워크스페이스 메인 레이아웃, 메시지 버블, 미리보기, 채팅 입력창 등의 스타일을 정의합니다.

## 10. test_features 폴더

`src/test_features/`는 실험용 기능이나 임시 시도 코드로 보이는 영역입니다.

- `src/test_features/components/AiAssistant.vue`
  - 테스트/실험용 AI 어시스턴트 컴포넌트입니다.

- `src/test_features/pages/`
  - 현재 눈에 띄는 파일은 없지만 테스트 페이지용 폴더로 보입니다.

## 11. 빌드 산출물/의존성 폴더

아래 폴더는 운영 코드 문서화 대상에서는 보통 제외합니다.

- `node_modules/`
  - 설치된 라이브러리 폴더입니다.

- `dist/`
  - `npm run build` 결과물 폴더입니다.

## 12. 현재 구조를 한 줄로 정리하면

이 프론트엔드는 `App.vue`가 페이지를 선택하고, `pages/`가 큰 화면을 조립하며, `components/`가 UI 조각을 담당하고, `composables/`가 상태와 동작 로직을 담당하는 구조입니다.

스타일은 `index.css`를 시작점으로 `assets/styles/` 아래 공통 CSS와 페이지별 CSS를 함께 사용하는 혼합형 구조입니다.
