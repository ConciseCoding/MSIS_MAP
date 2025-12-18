import numpy as np
from modules.map import config as cfg
# 도구 객체들도 상태의 일부이므로 여기서 임포트하여 인스턴스화
from modules.map.obstacle import MSIS_Obstacle
from modules.map.map_editor import MSIS_mapEditor

class RobotState:
    def __init__(self):
        # 1. Robot Status
        self.latest_pose = None  # (x, y, yaw)
        self.latest_scan = []
        self.battery_status = "Unknown"
        
        # 2. Navigation
        self.nav_target = None   # (x, y)
        self.nav_target_yaw = None
        self.target_name = None
        self.trajectory = []
        self.virtual_track = []
        self.is_tracking = False # 스레드 제어용 플래그

        # 3. Map Data
        self.last_rendered_map = None # 현재 보여지는 이미지
        self.viewer_map_info = {"x": 0.0, "y": 0.0, "w": 100, "h": 100, "res": 0.05}
        self.mapping_map_info = {"x": 0.0, "y": 0.0, "w": 100, "h": 100, "res": 0.05}
        self.is_loaded_mode = False # 파일 로드 모드 여부

        # 4. View State (화면 확대/Crop 상태)
        self.view_state = {
            "active_crop": False,
            "crop_rect": None,
            "crop_poly": None,
            "scale": 1.0,
            "offset": (0, 0)
        }
        
        # 5. Tools (각 로봇마다 별도의 장애물/에디터 관리)
        # 맵 이미지가 초기화되지 않았으므로 None으로 시작하거나 기본 이미지 할당
        default_map = np.full((cfg.IMG_H, cfg.IMG_W, 3), 150, dtype=np.uint8)
        
        self.obstacle_tool = MSIS_Obstacle(default_map)
        self.editor_tool = MSIS_mapEditor(default_map)
        
        self.selected_marker = None
        self.current_view_image = None