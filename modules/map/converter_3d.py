import numpy as np
import os
from modules.map import config as cfg

def generate_3d_ply(map_grid, map_info):
    """
    2D 맵 데이터를 3D Point Cloud (.ply) 파일로 변환합니다.
    **핵심 수정**: 좌표를 (0,0,0) 중심으로 이동시켜 뷰어에서 바로 보이게 함.
    """
    if map_grid is None or map_info is None:
        return None

    # map_grid: (H, W) numpy array
    # 값 의미: 255=벽, 0=미탐사, 0<x<255=빈공간(확률)
    h, w = map_grid.shape
    res = map_info['res']
    ox = map_info['x']
    oy = map_info['y']

    # 1. 좌표 격자 생성
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    
    # 픽셀 -> 월드 좌표 변환 (아직 오프셋 적용 전)
    # 이미지상 y는 아래로 증가하므로, 월드 좌표계(일반적으로 y 위로)에 맞게 반전 고려
    # 여기서는 단순히 시각화가 목적이므로 이미지 좌표계 그대로 쓰되, 스케일만 맞춤
    wx = (xv * res)
    wy = (yv * res) # 상하 반전 없이 그대로 사용 (뷰어에서 돌려보면 됨)

    # 2. 유효 데이터 추출
    # (A) 벽 (Wall) = 255
    wall_mask = (map_grid == 255)
    
    # (B) 바닥 (Floor) = 탐사된 영역 (0이 아니고 255도 아닌 값, 보통 128 근처)
    # 너무 어두운 색(장애물 확률 높음)은 바닥에서 제외하고 싶다면 범위를 조절
    floor_mask = (map_grid > 0) & (map_grid < 255)

    # 3. 포인트 생성
    points = []
    colors = []

    # --- [벽 생성] ---
    if np.any(wall_mask):
        w_x = wx[wall_mask]
        w_y = wy[wall_mask]
        
        # 벽은 높이감을 주기 위해 Z축으로 쌓음 (0.0m ~ 0.5m)
        for z in np.linspace(0, 0.8, 8): # 8개 층으로 쌓아서 벽처럼 보이게 함
            n = len(w_x)
            z_layer = np.full(n, z)
            
            pts = np.column_stack((w_x, w_y, z_layer))
            points.append(pts)
            
            # 벽 색상: 빨간색 (R, G, B)
            col = np.full((n, 3), [255, 50, 50])
            colors.append(col)

    # --- [바닥 생성] ---
    if np.any(floor_mask):
        f_x = wx[floor_mask]
        f_y = wy[floor_mask]
        n_f = len(f_x)
        
        # 바닥은 z=0
        z_layer = np.full(n_f, 0.0)
        pts_f = np.column_stack((f_x, f_y, z_layer))
        points.append(pts_f)
        
        # 바닥 색상: 연한 회색/흰색
        col_f = np.full((n_f, 3), [200, 200, 200])
        colors.append(col_f)

    if not points:
        return None

    # 데이터 병합
    all_points = np.vstack(points)
    all_colors = np.vstack(colors).astype(np.uint8)

    # ---------------------------------------------------------
    # [핵심] 좌표 중앙 정렬 (Centering)
    # ---------------------------------------------------------
    # 전체 포인트의 평균 지점을 구해서 0,0,0으로 이동시킴
    # 그래야 Gradio 3D 뷰어 카메라가 물체를 바로 비춤
    center_x = np.mean(all_points[:, 0])
    center_y = np.mean(all_points[:, 1])
    center_z = np.mean(all_points[:, 2])

    all_points[:, 0] -= center_x
    all_points[:, 1] -= center_y
    all_points[:, 2] -= center_z

    # 4. PLY 파일 저장
    save_path = os.path.join(cfg.DATA_DIR, "map_3d.ply")
    
    header = f"""ply
format ascii 1.0
element vertex {len(all_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property blue
end_header
"""
    with open(save_path, "w") as f:
        f.write(header)
        for p, c in zip(all_points, all_colors):
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")

    return save_path