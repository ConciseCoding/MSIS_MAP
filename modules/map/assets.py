# modules/map/assets.py

CUSTOM_CSS = """
#hidden_proxies { display: none !important; }

/* 맵 뷰어 스타일 (기존 유지) */
#map_viewer, #mapping_viewer { 
    height: 75vh !important; 
    background: #ffffff00; 
    overflow: hidden; 
    border-radius: 15px; 
    border: 1px solid #4b5563;
    margin-bottom: 20px !important;
}

.image-container { 
    width: 100%; 
    height: 100%; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    background-color: #969696; 
    overflow: hidden !important; 
    cursor: crosshair; 
    position: relative;
}

.image-container .icon-button-wrapper{
    visibility: hidden !important;
}

/* 컨트롤 패널 */
.control-panel-card { 
    background-color: #f9fafb !important; 
    border: 1px solid #e5e7eb !important; 
    border-radius: 12px !important; 
    padding: 20px !important; 
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important; 
}

/* -----------------------------------------------------------
   [수정됨] 네비게이션 라디오 버튼 스타일
   - 라디오 버튼 그룹을 가로로 강제 정렬
----------------------------------------------------------- */
.nav-radio-group {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 10px !important;
}

/* 라디오 버튼 컨테이너 (Gradio 내부 구조 대응) */
.nav-radio-group .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 5px !important;
    width: 100% !important;
}

/* 개별 라디오 버튼 */
.nav-radio-group .wrap label {
    flex: 1 1 0 !important; /* 균등 분할 */
    justify-content: center !important;
    text-align: center !important;
    padding: 8px 2px !important; /* 내부 여백 */
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    background: white !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    min-width: 0 !important; /* 텍스트 넘침 방지 */
}

/* 선택된 라디오 버튼 */
.nav-radio-group .wrap label.selected {
    background-color: #dbeafe !important; /* 연한 파랑 */
    border-color: #3b82f6 !important;
    color: #1e40af !important;
    font-weight: bold !important;
}

/* 라디오 버튼 내부 텍스트 */
.nav-radio-group .wrap label span {
    font-size: 0.8rem !important; /* 글자 크기 조정 */
    white-space: nowrap !important; /* 줄바꿈 방지 */
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* 기본 라디오 원형 버튼 숨기기 (선택적) */
.nav-radio-group input[type="radio"] {
    /* display: none !important; */ /* 원형 버튼 숨기려면 주석 해제 */
    margin-right: 4px !important;
}


/* 액션 버튼 (Go, Follow) 스타일 */
.action-btn { 
    min-height: 60px !important; 
    font-size: 16px !important; 
    font-weight: bold !important; 
    border-radius: 8px !important; 
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4) !important;
    height: 100% !important;
}

/* STOP 버튼 스타일 */
.stop-btn { 
    background-color: #ef4444 !important; 
    color: white !important; 
    border: none !important; 
    min-height: 60px !important; 
    font-size: 16px !important; 
    font-weight: bold !important; 
    border-radius: 8px !important; 
    margin: 0 !important; 
    box-shadow: 0 4px 6px rgba(239, 68, 68, 0.4) !important;
    height: 100% !important;
}
.stop-btn:active { transform: scale(0.98); }

/* --- 도구 모음 스타일 (Obstacle) --- */
.tool-row {
    gap: 5px !important;
    margin-bottom: 8px !important;
    justify-content: space-between !important; /* 버튼 간격 균등 */
}

/* 작은 아이콘 버튼 스타일 */
.tool-row button {
    padding: 2px 5px !important;
    min-height: 35px !important;
    font-size: 14px !important;
    flex-grow: 1 !important; /* 버튼 너비 채우기 */
}

/* 컬러 피커 스타일 */
.tool-row input[type="color"] {
    height: 36px !important;
    width: 100% !important;
    border: none !important;
    padding: 0 !important;
    background: transparent;
    cursor: pointer;
}

/* 텍스트 입력창 스타일 */
.tool-row textarea, .tool-row input[type="text"] {
    border: 1px solid #d1d5db !important;
    border-radius: 4px !important;
    background: white !important;
    padding: 4px 8px !important;
}

/* JSON 리스트 스타일 */
.json-scroll-list {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    max-height: 160px !important;
    overflow-y: auto !important;
    padding: 8px !important;
}
.json-scroll-list label {
    background: #f3f4f6;
    border-radius: 15px;
    padding: 4px 10px !important;
    margin: 2px !important;
    border: 1px solid #d1d5db;
    font-size: 0.85rem !important;
    display: inline-flex !important;
    align-items: center;
    transition: background 0.2s;
}
.json-scroll-list label:hover {
    background: #e5e7eb;
}
.json-scroll-list label.selected {
    background: #dbeafe;
    border-color: #3b82f6;
}

/* Hide Toolbar & Header Adjustments */
#map_viewer .image-button-container, #map_viewer .actions, #map_viewer .toolbar,
#mapping_viewer .image-button-container, #mapping_viewer .actions, #mapping_viewer .toolbar { display: none !important; }
#msis-header { display: flex; align-items: center; gap: 12px; background: transparent !important; padding: 0; margin-bottom: 12px; }
"""

PAN_ZOOM_JS = """
if (!window.mapZoomState) {
    window.mapZoomState = { scale: 1.0, pX: 0.0, pY: 0.0 };
}

function attachZoomListener(wrapper) {
    if (!wrapper) return;
    const img = wrapper.querySelector('img');
    if (!img) return;
    if (img.dataset.zoomAttached === 'true') return;
    img.dataset.zoomAttached = 'true';

    img.parentElement.style.overflow = 'hidden';
    img.parentElement.style.cursor = 'grab';
    img.style.transformOrigin = '0 0';
    img.style.transform = `translate(${window.mapZoomState.pX}px, ${window.mapZoomState.pY}px) scale(${window.mapZoomState.scale})`;

    wrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomStep = 0.15; 
        const delta = -Math.sign(e.deltaY); 
        const prevScale = window.mapZoomState.scale;
        let newScale = prevScale * (1 + delta * zoomStep);
        newScale = Math.min(Math.max(0.1, newScale), 10.0);

        const rect = img.parentElement.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const scaleRatio = newScale / prevScale;
        const newX = mouseX - (mouseX - window.mapZoomState.pX) * scaleRatio;
        const newY = mouseY - (mouseY - window.mapZoomState.pY) * scaleRatio;

        window.mapZoomState = { scale: newScale, pX: newX, pY: newY };
        img.style.transform = `translate(${newX}px, ${newY}px) scale(${newScale})`;
    }, { passive: false });

    let isDragging = false;
    let startX, startY, startPX, startPY;

    wrapper.addEventListener('mousedown', (e) => {
        if(e.button !== 0) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startPX = window.mapZoomState.pX;
        startPY = window.mapZoomState.pY;
        img.parentElement.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        e.preventDefault();
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        window.mapZoomState.pX = startPX + dx;
        window.mapZoomState.pY = startPY + dy;
        img.style.transform = `translate(${window.mapZoomState.pX}px, ${window.mapZoomState.pY}px) scale(${window.mapZoomState.scale})`;
    });

    window.addEventListener('mouseup', () => {
        if(isDragging) {
            isDragging = false;
            if(img.parentElement) img.parentElement.style.cursor = 'grab';
        }
    });
}

function setup_pan_zoom() {
    const targets = ['#map_viewer', '#mapping_viewer'];
    targets.forEach(selector => {
        const wrapper = document.querySelector(selector);
        if (!wrapper) return;
        attachZoomListener(wrapper);
        const observer = new MutationObserver(() => {
            attachZoomListener(wrapper);
        });
        observer.observe(wrapper, { childList: true, subtree: true });
    });
}
"""
SVG_INTERACTION_JS = """
<script>
function setupSVGInteraction() {
    const svg = document.getElementById('map-svg');
    if (!svg) return;

    // Pan & Zoom 변수
    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let panning = false;
    let startX = 0;
    let startY = 0;

    // 1. 클릭 이벤트 -> 좌표를 Hidden Textbox로 전송
    svg.addEventListener('click', (e) => {
        // SVG 내부 좌표 계산
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
        
        const coords = JSON.stringify({x: svgP.x, y: svgP.y});
        
        // Gradio의 숨겨진 Textbox 찾기 (id='click_coords'로 지정 예정)
        const textarea = document.querySelector('#click_coords textarea');
        if (textarea) {
            textarea.value = coords;
            // React 이벤트 트리거 (Gradio가 변경을 감지하게 함)
            const event = new Event('input', { bubbles: true });
            textarea.dispatchEvent(event);
        }
    });

    // 2. 줌 (Wheel)
    svg.addEventListener('wheel', (e) => {
        e.preventDefault();
        const xs = (e.clientX - pointX) / scale;
        const ys = (e.clientY - pointY) / scale;
        const delta = (e.wheelDelta ? e.wheelDelta : -e.deltaY);
        (delta > 0) ? (scale *= 1.2) : (scale /= 1.2);
        pointX = e.clientX - xs * scale;
        pointY = e.clientY - ys * scale;
        setTransform();
    });

    // 3. 팬 (Drag)
    svg.onmousedown = function (e) {
        if (e.button !== 0) return; // 좌클릭만
        e.preventDefault();
        startX = e.clientX - pointX;
        startY = e.clientY - pointY;
        panning = true;
    }
    svg.onmouseup = function (e) { panning = false; }
    svg.onmousemove = function (e) {
        e.preventDefault();
        if (!panning) return;
        pointX = e.clientX - startX;
        pointY = e.clientY - startY;
        setTransform();
    }

    function setTransform() {
        // SVG 내부 그룹이나 viewBox 조작 대신 CSS transform 사용 예시
        // 실제로는 viewBox를 조작하는 것이 더 깔끔함
        svg.style.transform = "translate(" + pointX + "px, " + pointY + "px) scale(" + scale + ")";
    }
}

// DOM 로드 시 실행 (또는 버튼 클릭 시 실행)
setTimeout(setupSVGInteraction, 1000);
</script>
"""

COMBINED_JS = PAN_ZOOM_JS + SVG_INTERACTION_JS + "\nsetup_pan_zoom();"