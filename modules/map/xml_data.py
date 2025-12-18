import xml.etree.ElementTree as ET
import os

class XMLMapData:
    def __init__(self):
        self.root = ET.Element("Map")
        self.width = 1000  # Default World Size (not pixels)
        self.height = 1000
        self.obstacles = [] # [{'id': 1, 'type': 'rect', 'x': 10, 'y': 20, ...}]

    def load_from_file(self, path):
        if os.path.exists(path):
            tree = ET.parse(path)
            self.root = tree.getroot()
            self._parse_obstacles()

    def _parse_obstacles(self):
        self.obstacles = []
        for obj in self.root.findall("Obstacle"):
            self.obstacles.append(obj.attrib)

    def add_obstacle(self, obs_type, x, y, w=10, h=10):
        # XML 요소 추가
        new_obs = ET.SubElement(self.root, "Obstacle")
        new_obs.set("type", obs_type)
        new_obs.set("x", str(x))
        new_obs.set("y", str(y))
        if obs_type == 'rect':
            new_obs.set("w", str(w))
            new_obs.set("h", str(h))
        self._parse_obstacles() # 리스트 갱신

    def get_svg_content(self, robot_pose=None):
        """XML 데이터를 SVG 문자열로 변환"""
        svg_elements = []
        
        # 1. 맵 배경
        svg_elements.append(f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#eee" stroke="#999" />')

        # 2. 장애물 렌더링
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                svg_elements.append(
                    f'<rect x="{obs["x"]}" y="{obs["y"]}" width="{obs["w"]}" height="{obs["h"]}" fill="black" />'
                )
            elif obs['type'] == 'circle':
                 svg_elements.append(
                    f'<circle cx="{obs["x"]}" cy="{obs["y"]}" r="{obs.get("r", 10)}" fill="black" />'
                )

        # 3. 로봇 렌더링 (동적 요소)
        if robot_pose:
            rx, ry, ryaw = robot_pose
            # 로봇 (파란 삼각형)
            svg_elements.append(
                f'<polygon points="{rx},{ry-10} {rx-8},{ry+10} {rx+8},{ry+10}" fill="blue" transform="rotate({ryaw} {rx} {ry})" />'
            )

        # SVG 태그 조합
        svg_body = "\n".join(svg_elements)
        return f"""
        <svg id="map-svg" viewBox="0 0 {self.width} {self.height}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:100%; cursor:crosshair;">
            {svg_body}
        </svg>
        """

xml_manager = XMLMapData()