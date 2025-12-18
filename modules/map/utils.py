import math

# 좌표 변환 로직
def world_to_pixel_with_info(wx, wy, info):
    if info is None or info.get("res", 0) == 0: return 0, 0
    dx = wx - info["x"]
    dy = wy - info["y"]
    px_float = dx / info["res"]
    py_float = dy / info["res"]
    
    # Slamtec Map Flip (X축 반전)
    px = info["w"] - 1 - px_float
    py = py_float 
    return int(px), int(py)

def pixel_to_world_with_info(px, py, info):
    if info is None or info.get("res", 0) == 0: return 0.0, 0.0
    real_px = info["w"] - 1 - px
    wx = (real_px * info["res"]) + info["x"]
    wy = (py * info["res"]) + info["y"]
    return float(wx), float(wy)


# [NEW] 뷰 변환 함수 (Crop 및 확대 적용된 좌표 계산)
def transform_pixel_to_view(px, py, view_state):
    """
    원본 맵의 픽셀 좌표(px, py)를 현재 보고 있는 화면(Crop/Zoom)의 좌표로 변환
    """
    if view_state["active_crop"] and view_state["crop_rect"]:
        rx, ry, _, _ = view_state["crop_rect"]
        scale = view_state["scale"]
        off_x, off_y = view_state["offset"]
        
        # (원본 - Crop시작점) * 배율 + 중앙정렬오프셋
        npx = int((px - rx) * scale + off_x)
        npy = int((py - ry) * scale + off_y)
        return npx, npy
    
    # Crop 아닐 때는 원본 좌표 그대로 (혹은 전체 화면 스케일링이 있다면 적용)
    return int(px), int(py)

def normalize_angle(angle):
    """각도를 -pi ~ pi 사이로 정규화"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def get_angle_diff(angle1, angle2):
    """두 각도 사이의 최소 회전각 계산 (절댓값)"""
    diff = normalize_angle(angle1 - angle2)
    return abs(diff)