import cv2
import numpy as np

img_w, img_h = 1200, 600

def resize_no_stretch(img, tw, th, pad=(150,150,150)):
    h, w = img.shape[:2]
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((th, tw, 3) if img.ndim==3 else (th, tw), pad, dtype=img.dtype)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

class MSIS_mapEditor:
    def __init__(self, map_image=None):
        self.base_map = map_image.copy() if map_image is not None else np.full((600, 1200, 3), 200, np.uint8)
        self.map_image = self.base_map.copy()
        self.points = []

    def update_map_image(self, img): self.map_image = img
    def add_point(self, x, y): self.points.append((x, y))
    def undo(self): self.points.pop() if self.points else None
    def clear(self): self.points = []

    def draw_live_preview(self, canvas_scale=1.0):
        if self.map_image is None: return None
        disp = self.map_image.copy()

        # 점과 선의 두께도 스케일에 비례해서 키움
        pt_radius = max(3, int(5 * canvas_scale))
        line_thick = max(1, int(2 * canvas_scale))
        close_thick = max(1, int(1 * canvas_scale))

        # 1. 점 그리기
        for p in self.points:
            # 좌표 스케일링
            cx = int(p[0] * canvas_scale)
            cy = int(p[1] * canvas_scale)
            cv2.circle(disp, (cx, cy), pt_radius, (0, 255, 0), -1)

        # 2. 선 그리기
        if len(self.points) >= 2:
            # 좌표 스케일링
            pts = (np.array(self.points, dtype=np.float32) * canvas_scale).astype(np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (0, 255, 255), line_thick, cv2.LINE_AA)

        # 3. 닫는 선 미리보기
        if len(self.points) >= 3:
            p_start = self.points[0]
            p_end = self.points[-1]
            
            sx, sy = int(p_start[0]*canvas_scale), int(p_start[1]*canvas_scale)
            ex, ey = int(p_end[0]*canvas_scale), int(p_end[1]*canvas_scale)
            
            cv2.line(disp, (ex, ey), (sx, sy), (0, 0, 255), close_thick, cv2.LINE_AA)

        return disp

    def crop_polygon_region(self):
        if self.map_image is None or len(self.points) < 3: return None
        pts = np.array(self.points, dtype=np.int32)
        mask = np.full(self.map_image.shape[:2], 150, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (150, 150, 150))
        
        masked = cv2.bitwise_and(self.map_image, self.map_image, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        if w == 0 or h == 0: return None

        crop_rgb = masked[y:y+h, x:x+w]
        bgra = cv2.merge([*cv2.split(crop_rgb), mask[y:y+h, x:x+w]])
        return resize_no_stretch(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), img_w, img_h)