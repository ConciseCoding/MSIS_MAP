import os
import cv2
import numpy as np
import struct
import json
import math
import modules.map.config as cfg
import modules.map.utils as utils


def fetch_explore_map_data(robot):
    if robot is None: return None, None
    
    # api.http_get 대신 robot._get 사용
    r = robot._get("/api/core/slam/v1/maps/explore", timeout=1.0)
    
    if not r or r.status_code != 200: return None, None
    data = r.content
    if len(data) < 20: return None, None
    ox, oy = struct.unpack("<ff", data[0:8])
    nx, ny = struct.unpack("<II", data[8:16])
    res = struct.unpack("<f", data[16:20])[0]
    expected = nx * ny
    # 버퍼 안전 처리
    buf = data[36 : 36 + expected] if len(data) >= 36 + expected else data[-expected:]
    if len(buf) != expected: return None, None
    
    grid = np.frombuffer(buf, dtype=np.uint8).reshape((ny, nx))
    return grid, {"x": ox, "y": oy, "w": nx, "h": ny, "res": res}

def explore_to_slamtec_editor_style(raw):
    # Numpy 조건부 인덱싱으로 압축
    img = np.full((*raw.shape, 3), 255, dtype=np.uint8) # 기본 흰색
    img[raw == 0] = (150, 150, 150) # 미탐사 회색
    img[raw == 255] = (0, 0, 0)     # 벽 검은색
    return img

def enhance_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 80)
    # Dilation으로 벽 두껍게
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    out = img.copy()
    out[edges > 0] = (0, 0, 0)
    return out

def load_static_viewer_map():
    if not os.path.exists(cfg.VIEWER_IMG_PATH): return None, None
    try:
        img = cv2.imread(cfg.VIEWER_IMG_PATH)
        info = {"x":0, "y":0, "w":img.shape[1], "h":img.shape[0], "res":0.05}
        if os.path.exists(cfg.VIEWER_INFO_PATH):
            with open(cfg.VIEWER_INFO_PATH, "r") as f: info.update(json.load(f))
        return img, info
    except: return None, None

def save_viewer_map(img, info):
    cv2.imwrite(cfg.VIEWER_IMG_PATH, img)
    with open(cfg.VIEWER_INFO_PATH, "w") as f: json.dump(info, f, indent=4)

# [수정] view_state를 인자로 받도록 수정
def draw_overlays(img, pose, scan, show_axis, show_lidar, map_info, 
                  nav_target=None, jsons=None, canvas_scale=1.0, ignore_crop=False,
                  virtual_track=None, is_active=True, view_state=None):
    
    if img is None or map_info is None: return img
    canvas = img.copy()
    if pose is None: return canvas
    
    rx, ry, yaw = pose

    # 1. 배율 계산
    base_width = 1200.0 * canvas_scale
    vis_scale = max(1.0, float(canvas.shape[1]) / base_width)
    vis_scale = min(2.5, vis_scale)

    # 2. 좌표 변환 헬퍼
    if view_state is None:
        view_state = {"active_crop": False, "scale": 1.0, "offset": (0,0)}

    do_transform = (not ignore_crop) and view_state["active_crop"]
    
    def get_view_pos(wx, wy):
        rpx, rpy = utils.world_to_pixel_with_info(wx, wy, map_info)
        if do_transform:
            vx, vy = utils.transform_pixel_to_view(rpx, rpy, view_state)
            return int(vx * canvas_scale), int(vy * canvas_scale)
        return int(rpx * canvas_scale), int(rpy * canvas_scale)

    # 색상 설정
    if is_active:
        col_robot = (0, 0, 255)      
        col_lidar = (255, 0, 0)      
        col_track = (0, 255, 255)    
        col_target = (128, 0, 128)   
    else:
        col_robot = (100, 100, 100) 
        col_lidar = (200, 200, 200) 
        col_track = (100, 150, 150) 
        col_target = (100, 100, 100)

    # 1. Virtual Track
    if virtual_track:
        pts = np.array([get_view_pos(p[0], p[1]) for p in virtual_track], np.int32)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, col_track, max(2, int(3 * vis_scale)), cv2.LINE_AA)
        if is_active:
            pt_rad = max(3, int(4 * vis_scale))
            for pt in pts:
                cv2.circle(canvas, tuple(pt), pt_rad, col_track, -1)

    # 2. Nav Target
    if nav_target:
        t_pos = get_view_pos(nav_target[0], nav_target[1])
        cv2.circle(canvas, t_pos, max(6, int(7 * vis_scale)), col_target, -1)
        if is_active:
            font_s, font_t = 0.7 * vis_scale, max(2, int(2 * vis_scale))
            cv2.putText(canvas, "Target", (t_pos[0] + int(5*vis_scale), t_pos[1] - int(5*vis_scale)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_s, col_target, font_t, cv2.LINE_AA)

    # 3. Lidar Points
    if show_lidar and scan:
        lidar_r = max(2, int(2 * vis_scale))
        for p in scan:
            if not p.get("valid"): continue
            d, a = p["distance"], p["angle"]
            if d <= 0: continue # 거리 0인 데이터 무시
            
            # [수정] 오타 수정: bx, by -> lx, ly
            lx, ly = d * math.cos(a), d * math.sin(a)
            wx = rx + (lx * math.cos(yaw) - ly * math.sin(yaw))
            wy = ry + (lx * math.sin(yaw) + ly * math.cos(yaw))
            
            px, py = get_view_pos(wx, wy)
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), lidar_r, col_lidar, -1)

    # 4. Robot Pose
    RW, RL = 0.45, 0.75
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    def local_to_world(lx, ly):
        return (rx + (lx*cos_y - ly*sin_y), ry + (lx*sin_y + ly*cos_y))

    p1 = local_to_world(RL/2, 0)
    p2 = local_to_world(-RL/2, RW/2)
    p3 = local_to_world(-RL/2, -RW/2)
    robot_pts = np.array([get_view_pos(*p1), get_view_pos(*p2), get_view_pos(*p3)], np.int32)
    cv2.fillPoly(canvas, [robot_pts], col_robot)
    
    if not is_active:
         center = get_view_pos(rx, ry)
         cv2.putText(canvas, "Robot", (center[0], center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_robot, 1)

    # 5. Axis & JSON
    if is_active:
        if show_axis:
            th = max(2, int(2 * vis_scale))
            origin = get_view_pos(0, 0)
            cv2.arrowedLine(canvas, origin, get_view_pos(1, 0), (0, 0, 255), th, tipLength=0.1)
            cv2.arrowedLine(canvas, origin, get_view_pos(0, 1), (0, 255, 0), th, tipLength=0.1)
        
        if jsons:
            for jf in jsons:
                path = os.path.join(cfg.DATA_DIR, jf)
                if not os.path.exists(path): continue
                try:
                    with open(path, "r") as f: data = json.load(f)
                    for pos in data.get("positions", []):
                        hc = pos.get("color", "#00FF00").lstrip("#")
                        bgr = tuple(int(hc[i:i+2], 16) for i in (0,2,4))[::-1]

                        pt = get_view_pos(pos["x"], pos["y"])
                        is_sel = False
                        # 선택 로직 등... (생략하거나 cfg 참조)
                        # 여기서는 화면 표시만 중요하므로 간단히 그림

                        rad = max(5, int(7 * vis_scale))
                        fs = max(0.5, 0.6 * vis_scale)
                        ft = max(2, int(2 * vis_scale))
                        ot = max(3, int(3 * vis_scale))

                        cv2.circle(canvas, pt, rad, bgr, -1)
                        cv2.circle(canvas, pt, rad, (0,0,0), max(2, int(2*vis_scale)))

                        label = pos.get("name", "?")
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
                        tx, ty = pt[0] - tw//2, pt[1] - rad - int(5*vis_scale)

                        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), ot, cv2.LINE_AA)
                        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), ft, cv2.LINE_AA)
                except: pass

    return canvas