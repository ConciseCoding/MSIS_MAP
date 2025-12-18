import requests
import threading
import time
import json
import math
from modules.map.states import RobotState

class RobotClient:
    def __init__(self, name, ip, port):
        self.name = name
        self.base_url = f"http://{ip}:{port}"
        
        # [핵심] 이 로봇만의 상태(State) 인스턴스 생성
        self.state = RobotState()
        
        # 백그라운드 스레드 제어
        self.connected = True
        self._start_threads()

    def _start_threads(self):
        threading.Thread(target=self._pose_updater, daemon=True).start()
        threading.Thread(target=self._lidar_updater, daemon=True).start()

    def disconnect(self):
        """연결 종료 시 호출"""
        self.connected = False

    # --- HTTP Helpers ---
    def _get(self, path, **kw):
        try: return requests.get(f"{self.base_url}{path}", **kw)
        except: return None
    
    def _post(self, path, json=None, **kw):
        try: return requests.post(f"{self.base_url}{path}", json=json, **kw)
        except: return None

    def _put(self, path, json=None, **kw):
        try: return requests.put(f"{self.base_url}{path}", json=json, **kw)
        except: return None
        
    def _delete(self, path, **kw):
        try: return requests.delete(f"{self.base_url}{path}", **kw)
        except: return None

    # --- Background Tasks ---
    def _pose_updater(self):
        while self.connected:
            r = self._get("/api/core/slam/v1/localization/pose", timeout=0.25)
            if r and r.status_code == 200:
                d = r.json()
                self.state.latest_pose = (float(d.get("x",0)), float(d.get("y",0)), float(d.get("yaw",0)))
            time.sleep(0.1)

    def _lidar_updater(self):
        while self.connected:
            r = self._get("/api/core/system/v1/laserscan", timeout=0.5)
            if r and r.status_code == 200:
                self.state.latest_scan = r.json().get("laser_points", [])
            time.sleep(0.08)

    # --- Actions (기존 함수들을 메서드로 변환) ---
    def get_laserscan(self):
        return self.state.latest_scan

    def send_move_to(self, x, y, yaw=None, speed=0.6):
        self.state.trajectory = [] 
        # self.state.nav_target = (x, y) # 핸들러에서 설정하므로 여기선 생략 가능
        
        target = {"x": float(x), "y": float(y), "z": 0.0}
        opts = {"mode": 0, "acceptable_precision": 0.01, "speed_ratio": speed, "flags": ["precise"]}
        
        if yaw is not None:
            opts["yaw"] = math.radians(float(yaw))
            opts["flags"].append("with_yaw")
            
        payload = {"action_name": "slamtec.agent.actions.MoveToAction", "options": {"target": target, "move_options": opts}}
        return self._post("/api/core/motion/v1/actions", json=payload, timeout=2)

    def follow_path(self, points, speed_ratio=1.0):
        formatted_points = [{"x": float(p[0]), "y": float(p[1])} for p in points]
        payload = {
            "action_name": "follow_path_points",
            "options": {
                "path_points": formatted_points,
                "move_options": {"mode": 0, "flags": ["find_path_ignoring_dynamic_obstacles"], "speed_ratio": speed_ratio}
            }
        }
        r = self._post("/api/core/motion/v1/actions", json=payload, timeout=2.0)
        return r and r.status_code == 200, "Path Started" if r and r.status_code==200 else "Failed"

    def stop_now(self):
        self.state.nav_target = None
        self.state.trajectory = []
        self.state.is_tracking = False
        return self._delete("/api/core/motion/v1/actions/:current", timeout=0.4)

    def go_charge(self):
        payload = {"action_name": "slamtec.agent.actions.GoHomeAction", "options": {"flags": ["dock"], "back_to_landing": True}}
        self._post("/api/core/motion/v1/actions", json=payload, timeout=2)

    # --- Map & System ---
    def enable_mapping(self, enable=True):
        self._put("/api/core/slam/v1/mapping/:enable", json={"enable": enable})
    
    def reset_map(self):
        return self._delete("/api/core/slam/v1/maps")

    def sync_map_to_robot(self):
        r = self._post("/api/core/slam/v1/maps", json={})
        return r.status_code in [200, 201], "Synced"

    # [NEW] Artifacts (state가 아닌 API 호출은 그대로 유지하되 인스턴스 메서드로)
    def get_pois(self):
        r = self._get("/api/core/artifact/v1/pois", timeout=0.5)
        return r.json() if r and r.status_code == 200 else []
        
    def get_lines(self, usage="walls"):
        r = self._get(f"/api/core/artifact/v1/lines/{usage}", timeout=0.5)
        return r.json() if r and r.status_code == 200 else []