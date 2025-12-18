import cv2
import numpy as np
import math

class MSIS_Obstacle:
    def __init__(self, map_image=None):
        self.map_image = map_image
        self.obstacles = []  # List of dicts
        self.selected_index = -1 
        self.line_start = None 
        self.current_free = []
        self.preview_text = None 
        self.rect_start = None # 직사각형 시작점

    def update_map_image(self, image):
        self.map_image = image

    # -------------------------------------------------------------
    # 도형 추가 메서드
    # -------------------------------------------------------------
    def add_shape(self, shape_type, x, y, size=5, label=""):
        self.obstacles.append({
            'type': shape_type, 'x': x, 'y': y, 
            'size': size, 'angle': 0, 
            'points': [], 'label': label 
        })
        self.selected_index = len(self.obstacles) - 1

    def add_brush(self, x, y, color_code, size=5, angle=0, label=""):
        colors = {'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (150, 150, 150)}
        self.obstacles.append({
            'type': 'brush', 'x': x, 'y': y, 'size': size, 'angle': angle,
            'color': colors.get(color_code, (0,0,0)), 'color_name': color_code, 'label': label
        })
        return f"Painted {color_code}"

    # [수정] 직사각형 추가 (size 키 추가)
    def add_rect_point(self, x, y):
        if self.rect_start is None:
            self.rect_start = (x, y)
            return "Click Opposite Corner"
        else:
            sx, sy = self.rect_start
            w = abs(x - sx)
            h = abs(y - sy)
            cx = (x + sx) // 2
            cy = (y + sy) // 2
            
            self.obstacles.append({
                'type': 'rectangle',
                'x': cx, 'y': cy,
                'w': w, 'h': h, 
                'angle': 0, 'label': "",
                'size': 5 # [중요] 기본 size 추가 (에러 방지)
            })
            self.rect_start = None
            self.selected_index = len(self.obstacles) - 1
            return "Rectangle Created"

    def add_line_point(self, x, y):
        if self.line_start is None:
            self.line_start = (x, y); return "Click End Point"
        else:
            sx, sy = self.line_start
            self.obstacles.append({
                'type': 'line', 'p1': (sx, sy), 'p2': (x, y),
                'x': (sx+x)//2, 'y': (sy+y)//2, 'size': 2, 'angle': 0, 'label': ""
            })
            self.line_start = None; self.selected_index = len(self.obstacles) - 1
            return "Line Created"

    def add_free_point(self, x, y):
        if len(self.current_free) >= 3:
            sx, sy = self.current_free[0]
            if math.hypot(x - sx, y - sy) < 10:
                self.obstacles.append({
                    'type': 'free', 'x': 0, 'y': 0, 'size': 1, 'angle': 0,
                    'points': list(self.current_free), 'label': ""
                })
                self.current_free = []; return "Free Polygon Closed"
        self.current_free.append((x, y))
        return "Point Added"

    # -------------------------------------------------------------
    # 텍스트 관련
    # -------------------------------------------------------------
    def set_preview_text(self, x, y, text, size, color_code):
        rgb = self._get_color(color_code)
        self.preview_text = {'type': 'text', 'x': x, 'y': y, 'text': text, 'size': size, 'color': rgb, 'color_name': color_code}
        return f"Text Preview"

    def apply_text(self):
        if self.preview_text:
            self.obstacles.append(self.preview_text); self.preview_text = None
            self.selected_index = len(self.obstacles) - 1
            return "Text Applied"
        return "No text"

    def update_preview_props(self, text, size, color_code):
        if self.preview_text:
            self.preview_text['text'] = text; self.preview_text['size'] = int(size)
            self.preview_text['color'] = self._get_color(color_code)

    def _get_color(self, code):
        if code.startswith("#"): return self._hex_to_bgr(code)
        colors = {'black':(0,0,0), 'white':(255,255,255), 'red':(0,0,255), 'blue':(255,0,0), 'green':(0,255,0), 'yellow':(0,255,255)}
        return colors.get(code, (0,0,0))

    def _hex_to_bgr(self, hex_col):
        try:
            hex_col = hex_col.lstrip('#')
            return tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))[::-1] 
        except: return (0, 0, 0)

    # -------------------------------------------------------------
    # 충돌 감지 및 선택
    # -------------------------------------------------------------
    def is_point_inside(self, x, y):
        pt = (x, y)
        for obs in self.obstacles:
            if obs['type'] == 'text': continue 
            if obs['type'] in ['brush', 'square']:
                rect = ((obs['x'], obs['y']), (obs['size']*2, obs['size']*2), obs['angle'])
                if cv2.pointPolygonTest(np.intp(cv2.boxPoints(rect)), pt, False) > 0: return True
            
            # [NEW] 직사각형 충돌 체크
            elif obs['type'] == 'rectangle':
                rect = ((obs['x'], obs['y']), (obs['w'], obs['h']), obs['angle'])
                if cv2.pointPolygonTest(np.intp(cv2.boxPoints(rect)), pt, False) > 0: return True

            elif obs['type'] in ['circle', 'semicircle']:
                if math.hypot(x - obs['x'], y - obs['y']) < obs['size']: return True
            elif obs['type'] in ['triangle', 'pentagon']:
                sides = 3 if obs['type'] == 'triangle' else 5
                pts = self._get_poly_points(obs['x'], obs['y'], obs['size'], sides, obs['angle'])
                if cv2.pointPolygonTest(pts, pt, False) > 0: return True
            elif obs['type'] == 'free':
                if cv2.pointPolygonTest(np.array(obs['points'], np.int32), pt, False) > 0: return True
            elif obs['type'] == 'line':
                p1, p2, p3 = np.array(obs['p1']), np.array(obs['p2']), np.array([x, y])
                l2 = np.sum((p1-p2)**2)
                if l2 == 0: dist = np.linalg.norm(p3-p1)
                else: t = max(0, min(1, np.dot(p3-p1, p2-p1)/l2)); dist = np.linalg.norm(p3 - (p1 + t*(p2-p1)))
                if dist < max(1, int(obs.get('size', 2)/6)): return True
        return False

    def select_object(self, x, y):
        for i in range(len(self.obstacles) - 1, -1, -1):
            obs = self.obstacles[i]
            if obs['type'] == 'brush': continue
            
            if obs['type'] == 'text':
                if abs(x - obs['x']) < obs['size'] and abs(y - obs['y']) < obs['size']:
                    self.selected_index = i; return True
                continue

            # [수정] 직사각형 선택
            if obs['type'] == 'rectangle':
                # 회전 고려 없이 간단한 바운딩 박스로 체크 (편의상)
                # w, h는 전체 너비/높이이므로 중심에서 절반
                if abs(x - obs['x']) < obs['w']/2 + 10 and abs(y - obs['y']) < obs['h']/2 + 10:
                    self.selected_index = i; return True

            if obs['type'] == 'free':
                if cv2.pointPolygonTest(np.array(obs['points']), (x, y), False) >= 0: self.selected_index = i; return True
            elif obs['type'] == 'line':
                p1, p2, p3 = np.array(obs['p1']), np.array(obs['p2']), np.array([x, y])
                l2 = np.sum((p1-p2)**2)
                if l2 == 0: dist = np.linalg.norm(p3-p1)
                else: t = max(0, min(1, np.dot(p3-p1, p2-p1)/l2)); dist = np.linalg.norm(p3 - (p1 + t*(p2-p1)))
                if dist < 10: self.selected_index = i; return True
            else:
                if math.hypot(x - obs['x'], y - obs['y']) <= obs['size'] * 1.5: self.selected_index = i; return True
        self.selected_index = -1
        return False

    def move_selected(self, x, y):
        if self.selected_index != -1:
            obs = self.obstacles[self.selected_index]
            if obs['type'] == 'line':
                dx, dy = x - obs['x'], y - obs['y']
                obs['p1'] = (obs['p1'][0]+dx, obs['p1'][1]+dy)
                obs['p2'] = (obs['p2'][0]+dx, obs['p2'][1]+dy)
            obs['x'], obs['y'] = x, y

    def delete_selected(self):
        if self.preview_text: self.preview_text = None; return "Cancelled"
        if len(self.obstacles) > 0:
            if self.selected_index != -1:
                self.obstacles.pop(self.selected_index); self.selected_index = -1
                return "Deleted Selected"
            else:
                self.obstacles.pop(); return "Deleted Last"
        return "Nothing"

    def update_selected_property(self, size=None, angle=None, label=None):
        if self.selected_index != -1:
            obs = self.obstacles[self.selected_index]
            
            # [수정] 직사각형 크기 변경 로직
            if size is not None: 
                if obs['type'] == 'rectangle':
                    # 비율 유지하며 크기 변경
                    ratio = obs['h'] / obs['w'] if obs['w'] > 0 else 1.0
                    new_w = int(size) * 4 # 슬라이더 값이 작으므로 적당히 키움
                    obs['w'] = new_w
                    obs['h'] = int(new_w * ratio)
                    obs['size'] = int(size) # size 값도 동기화
                else:
                    obs['size'] = int(size)
            
            if angle is not None: obs['angle'] = int(angle)
            if label is not None: 
                if obs['type'] == 'text': obs['text'] = str(label)
                else: obs['label'] = str(label)

    def get_selected_info(self):
        if self.selected_index != -1:
            obs = self.obstacles[self.selected_index]
            txt = obs['text'] if obs['type'] == 'text' else obs.get('label', "")
            return obs.get('size', 10), obs.get('angle', 0), txt
        return 10, 0, ""

    def _get_poly_points(self, x, y, radius, sides, angle_deg):
        points = []
        offset = -90 + angle_deg
        for i in range(sides):
            rad = math.radians(offset + (360 / sides) * i)
            points.append([x + radius * math.cos(rad), y + radius * math.sin(rad)])
        return np.array(points, np.int32)

    def undo(self):
        if self.preview_text: self.preview_text = None; return "Undo"
        if self.line_start: self.line_start = None; return "Undo"
        if self.rect_start: self.rect_start = None; return "Undo Rect"
        if self.current_free: self.current_free.pop(); return "Undo"
        if self.obstacles: self.obstacles.pop(); self.selected_index = -1; return "Undo"
        return "Nothing"
    
    def clear(self):
        self.obstacles = []; self.current_free = []; self.line_start = None; self.rect_start = None; self.selected_index = -1; self.preview_text = None

    # -------------------------------------------------------------
    # [수정] 그리기 (에러 방지: .get('size') 사용)
    # -------------------------------------------------------------
    def draw_obstacles(self, canvas_scale=1.0):
        if self.map_image is None: return

        current_map_width = float(self.map_image.shape[1])
        base_width = 1600.0 * canvas_scale
        draw_ratio = max(1.0, current_map_width / base_width)
        draw_ratio = min(1.5, draw_ratio) 

        for i, obs in enumerate(self.obstacles):
            if obs['type'] == 'brush': color = obs['color']
            elif obs['type'] == 'text': color = obs['color']
            else: color = (0, 255, 255) if i == self.selected_index else (0, 0, 0)
            
            thickness = -1
            
            cx = int(obs['x'] * canvas_scale)
            cy = int(obs['y'] * canvas_scale)
            # [수정] size가 없는 경우(직사각형 등) 대비 안전하게 가져옴
            scaled_size = int(obs.get('size', 2) * canvas_scale)

            # --- 도형 그리기 ---
            if obs['type'] == 'brush' or obs['type'] == 'square':
                rect = ((cx, cy), (scaled_size*2, scaled_size*2), obs['angle'])
                box = cv2.boxPoints(rect); box = np.intp(box)
                cv2.drawContours(self.map_image, [box], 0, color, thickness)

            elif obs['type'] == 'rectangle':
                w, h = int(obs['w'] * canvas_scale), int(obs['h'] * canvas_scale)
                rect = ((cx, cy), (w, h), obs['angle'])
                box = cv2.boxPoints(rect); box = np.intp(box)
                cv2.drawContours(self.map_image, [box], 0, color, -1)

            elif obs['type'] == 'circle':
                cv2.circle(self.map_image, (cx, cy), scaled_size, color, thickness)
            
            elif obs['type'] in ['triangle', 'pentagon']:
                sides = 3 if obs['type'] == 'triangle' else 5
                pts = self._get_poly_points(cx, cy, scaled_size, sides, obs['angle'])
                cv2.fillPoly(self.map_image, [pts], color)

            elif obs['type'] == 'semicircle':
                cv2.ellipse(self.map_image, (cx, cy), (scaled_size, scaled_size), obs['angle'], 180, 360, color, thickness)

            elif obs['type'] == 'line':
                p1_x, p1_y = int(obs['p1'][0] * canvas_scale), int(obs['p1'][1] * canvas_scale)
                p2_x, p2_y = int(obs['p2'][0] * canvas_scale), int(obs['p2'][1] * canvas_scale)
                line_thick = max(1, int(obs.get('size', 2)/3 * canvas_scale))
                cv2.line(self.map_image, (p1_x, p1_y), (p2_x, p2_y), color, line_thick, cv2.LINE_AA)
                if i == self.selected_index:
                    rad = int(3 * canvas_scale)
                    cv2.circle(self.map_image, (p1_x, p1_y), rad, color, -1)
                    cv2.circle(self.map_image, (p2_x, p2_y), rad, color, -1)

            elif obs['type'] == 'free':
                pts = np.array(obs['points'], dtype=np.float32) * canvas_scale
                pts = pts.astype(np.int32)
                cv2.fillPoly(self.map_image, [pts], color)

            # --- 텍스트 그리기 ---
            if obs['type'] == 'text':
                this_font_scale = (scaled_size / 30.0) * draw_ratio
                this_font_thick = max(1, int(this_font_scale * 1.5))
                
                label = obs['text']
                (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, this_font_scale, this_font_thick)
                
                tx = cx - w // 2
                ty = cy + h // 2
                
                if i == self.selected_index:
                    pad = int(5 * canvas_scale)
                    cv2.rectangle(self.map_image, (tx-pad, ty-h-pad), (tx+w+pad, ty+pad), (0, 255, 255), int(2*canvas_scale))

                cv2.putText(self.map_image, label, (tx, ty), 
                            cv2.FONT_HERSHEY_SIMPLEX, this_font_scale, color, this_font_thick, cv2.LINE_AA)
                continue

            # 일반 도형 라벨
            label = obs.get('label', "")
            if label:
                label_font_scale = 0.4 * draw_ratio * canvas_scale
                label_font_thick = max(1, int(1 * canvas_scale))
                
                if obs['type'] == 'rectangle':
                    label_y = cy - int(obs['h'] * canvas_scale / 2) - int(5 * canvas_scale)
                else:
                    label_y = cy - scaled_size - int(5 * canvas_scale)
                    
                cv2.putText(self.map_image, label, (cx, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0,0,0), label_font_thick, cv2.LINE_AA)

        # Preview Text
        if self.preview_text:
            p = self.preview_text
            font_scale = (p['size'] / 30.0) * draw_ratio * canvas_scale
            font_thick = max(1, int(font_scale * 1.5))
            (w, h), _ = cv2.getTextSize(p['text'], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            cx, cy = int(p['x'] * canvas_scale), int(p['y'] * canvas_scale)
            tx = cx - w // 2
            ty = cy + h // 2
            
            cv2.putText(self.map_image, p['text'], (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, p['color'], font_thick, cv2.LINE_AA)
            
            cross_len = int(10 * canvas_scale)
            cv2.line(self.map_image, (cx-cross_len, cy), (cx+cross_len, cy), (0,0,255), int(1*canvas_scale))
            cv2.line(self.map_image, (cx, cy-cross_len), (cx, cy+cross_len), (0,0,255), int(1*canvas_scale))

        # Free drawing preview
        if len(self.current_free) > 0:
            pts = np.array(self.current_free, dtype=np.float32) * canvas_scale
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.map_image, [pts], False, (0, 255, 0), int(2*canvas_scale))
        
        if self.line_start:
            lx, ly = int(self.line_start[0]*canvas_scale), int(self.line_start[1]*canvas_scale)
            cv2.circle(self.map_image, (lx, ly), int(2*canvas_scale), (0, 255, 0), -1)

        # [NEW] 직사각형 미리보기
        if self.rect_start:
            rx, ry = int(self.rect_start[0]*canvas_scale), int(self.rect_start[1]*canvas_scale)
            cv2.circle(self.map_image, (rx, ry), int(2*canvas_scale), (0, 0, 255), -1)