from modules.map import config as cfg
from modules.map.api import RobotClient

class RobotManager:
    def __init__(self):
        self.robots = {} # {name: RobotClient}
        self.current_robot_name = None

     # [NEW] 연결된 모든 로봇 상태 문자열 반환
    def get_connection_status_string(self):
        if not self.robots:
            return "Disconnected"
        
        status_parts = []
        for name in self.robots:
            if name == self.current_robot_name:
                # 현재 제어 중인 로봇 (파란색 원)
                status_parts.append(f"🔵 {name} (Active)")
            else:
                # 연결은 되어 있지만 백그라운드인 로봇 (초록색 원)
                status_parts.append(f"🟢 {name}")
        
        # 예: "🔵 AMR_01 (Active) | 🟢 AMR_02"
        return " | ".join(status_parts)

    def add_robot(self, name, ip, port):
        if name in self.robots: return False, "Exists"
        # 연결 시도 및 인스턴스 생성
        client = RobotClient(name, ip, port)
        self.robots[name] = client
        
        # 설정 파일에도 저장
        cfg.ROBOT_LIST[name] = {"ip": ip, "port": port}
        cfg.save_robot_list(cfg.ROBOT_LIST)
        
        if self.current_robot_name is None:
            self.current_robot_name = name
            
        return True, "Added & Connected"

    def select_robot(self, name):
        if name in self.robots:
            self.current_robot_name = name
            return True
        return False

    def get_current_robot(self):
        if self.current_robot_name and self.current_robot_name in self.robots:
            return self.robots[self.current_robot_name]
        return None
    
    def delete_robot(self, name):
        if name in self.robots:
            self.robots[name].disconnect() # 스레드 종료
            del self.robots[name]
            
            if name in cfg.ROBOT_LIST:
                del cfg.ROBOT_LIST[name]
                cfg.save_robot_list(cfg.ROBOT_LIST)
            
            if self.current_robot_name == name:
                self.current_robot_name = next(iter(self.robots)) if self.robots else None
            return True
        return False

    # 초기화 시 저장된 로봇 자동 연결
    def auto_connect_saved(self):
        for name, info in cfg.ROBOT_LIST.items():
            self.add_robot(name, info['ip'], info['port'])

manager = RobotManager()
manager.auto_connect_saved()