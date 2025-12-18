import cv2
import numpy as np
import gradio as gr
import json, os, sys, math, threading, time
import traceback
from modules.map import config as cfg
from modules.map import api
from modules.map import map_core as mc
from modules.map import utils

from modules.map.manager import manager # Manager Import

from modules.map import converter_3d

# [NEW] XML 및 인코딩 관련 라이브러리 추가
import xml.etree.ElementTree as ET
import base64
from xml.dom import minidom

RENDER_SCALE = 4.0  # Super-Sampling Scale

# =========================================================
# Map Rendering
# =========================================================

def _apply_map_boundary_cleanup(image, render_scale):
    """
    1. 맵의 흰색 영역(주행가능)을 찾습니다.
    2. 외곽선을 부드럽게 다듬습니다.
    3. 맵 바깥쪽(미탐사 영역 등)을 회색으로 덮어버립니다.
    4. 맵과 회색 배경의 경계에 검은색 선을 그립니다.
    """
    # 1. 그레이스케일 변환 및 이진화 (흰색 영역 검출)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 밝기가 230 이상인 부분을 맵(흰색)으로 간주
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # 2. 모폴로지 연산 (작은 구멍 메우기 및 노이즈 제거)
    # 렌더 스케일에 비례한 커널 크기
    k_size = int(3 * render_scale) | 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    # 닫힘 연산 (흰색 영역 내부의 검은 점 제거)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. 외곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image

    # 4. 새로운 캔버스 생성 (전체 회색 배경)
    # 기존 이미지의 노이즈를 완전히 없애기 위해 회색 캔버스에서 시작
    cleaned_image = np.full_like(image, (150, 150, 150))
    
    # 5. 외곽선 다듬기 및 맵 채우기
    smoothed_contours = []
    for cnt in contours:
        # 너무 작은 영역(노이즈) 무시
        if cv2.contourArea(cnt) < 50 * render_scale:
            continue
            
        # 외곽선 단순화 (매끄럽게)
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        smoothed_contours.append(approx)
        
        # 정리된 맵 영역을 흰색으로 채우기
        cv2.drawContours(cleaned_image, [approx], -1, (255, 255, 255), -1)

    # 6. 검은색 테두리 그리기
    # 흰색과 회색의 경계면에 검은 선을 그립니다.
    border_thick = int(4 * render_scale)
    cv2.drawContours(cleaned_image, smoothed_contours, -1, (0, 0, 0), border_thick, cv2.LINE_AA)
    
    return cleaned_image

# -------------------------------------------------------------------------
# 헬퍼: 맵 외곽선 정리
# -------------------------------------------------------------------------
# def _apply_map_boundary_cleanup(image, render_scale):
#     lower_white = np.array([240, 240, 240])
#     upper_white = np.array([255, 255, 255])
#     white_mask = cv2.inRange(image, lower_white, upper_white)
    
#     k_size = int(3 * render_scale) | 1 
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
#     mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours: return image

#     smoothed_contours = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) < 200 * render_scale: continue
#         epsilon = 0.003 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         smoothed_contours.append(approx)

#     border_thick = int(2 * render_scale)
#     cv2.drawContours(image, smoothed_contours, -1, (0, 0, 0), border_thick, cv2.LINE_AA)
#     return image

def get_robot():
    return manager.get_current_robot()

# =========================================================
# Map Viewer Update (Editor Tab)
# =========================================================
def update_viewer_map(jsons, axis, lidar):
    # [1] 현재 로봇 가져오기
    robot = manager.get_current_robot()
    if not robot: return cfg.DEFAULT_MAP_IMAGE
    
    st = robot.state 

    try:
        if jsons is None: jsons = []

        # [2] 맵 데이터 준비 (데이터가 없으면 가져오기 시도)
        if st.last_rendered_map is None and not st.is_loaded_mode:
            # 1. 저장된 정적 뷰어 맵 로드 시도
            img, info = mc.load_static_viewer_map()
            if img is not None:
                st.last_rendered_map = img
                st.viewer_map_info.update(info)
            else:
                # 2. 정적 맵도 없으면 실시간 탐사 맵 가져오기 (Mapping 탭과 동일)
                grid, info = mc.fetch_explore_map_data(robot)
                if grid is not None:
                    slam = mc.explore_to_slamtec_editor_style(grid)
                    slam = mc.enhance_map(slam)
                    slam = cv2.flip(slam, 1)
                    st.last_rendered_map = slam
                    st.viewer_map_info.update(info)
        
        # 여전히 데이터가 없으면 회색 화면
        if st.last_rendered_map is None:
            return cfg.DEFAULT_MAP_IMAGE

        # -------------------------------------------------------------
        # 슈퍼 샘플링 & 그리기
        # -------------------------------------------------------------
        base = st.last_rendered_map.copy()
        hi_res_base = cv2.resize(base, None, fx=RENDER_SCALE, fy=RENDER_SCALE, interpolation=cv2.INTER_NEAREST)
        
        # 외곽선 정리
        hi_res_base = _apply_map_boundary_cleanup(hi_res_base, RENDER_SCALE)

        # 장애물 그리기 (로봇별 도구 사용)
        st.obstacle_tool.update_map_image(hi_res_base)
        st.obstacle_tool.draw_obstacles(canvas_scale=RENDER_SCALE)

        final_canvas = None 
        
        # [3] 뷰 영역 결정 (Crop vs Full)
        # cfg.VIEW_STATE -> st.view_state 로 변경
        if st.view_state["active_crop"] and st.view_state["crop_poly"] is not None:
            scaled_poly = (st.view_state["crop_poly"] * RENDER_SCALE).astype(np.int32)
            
            mask = np.zeros(hi_res_base.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_poly], 255)
            
            bg = np.full_like(hi_res_base, (150, 150, 150))
            fg = cv2.bitwise_and(hi_res_base, hi_res_base, mask=mask)
            bg_p = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
            processed_map = cv2.add(fg, bg_p)

            # Crop 경계선
            cv2.polylines(processed_map, [scaled_poly], True, (150, 150, 150), int(2 * RENDER_SCALE), cv2.LINE_AA)

            rx, ry, rw, rh = cv2.boundingRect(scaled_poly)
            
            if rw > 0 and rh > 0:
                cropped = processed_map[ry:ry+rh, rx:rx+rw]
                target_w, target_h = cfg.IMG_W, cfg.IMG_H
                scale = min(target_w / rw, target_h / rh)
                new_w, new_h = int(rw * scale), int(rh * scale)
                
                resized_map = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                final_canvas = np.full((target_h, target_w, 3), (150, 150, 150), dtype=np.uint8)
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2
                final_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_map
                
                st.view_state["crop_rect"] = (int(rx/RENDER_SCALE), int(ry/RENDER_SCALE), int(rw/RENDER_SCALE), int(rh/RENDER_SCALE))
                st.view_state["scale"] = scale * RENDER_SCALE
                st.view_state["offset"] = (x_off, y_off)
            else:
                st.view_state["active_crop"] = False

        if not st.view_state["active_crop"]:
            st.view_state["active_crop"] = False
            h, w = hi_res_base.shape[:2]
            target_w, target_h = cfg.IMG_W, cfg.IMG_H
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized_map = cv2.resize(hi_res_base, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            final_canvas = np.full((target_h, target_w, 3), (150, 150, 150), dtype=np.uint8)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            final_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_map

            st.view_state["crop_rect"] = (0, 0, base.shape[1], base.shape[0])
            st.view_state["scale"] = scale * RENDER_SCALE
            st.view_state["offset"] = (x_off, y_off)

        # [4] 오버레이 그리기 (모든 로봇)
        was_active = st.view_state["active_crop"]
        st.view_state["active_crop"] = True
        
        # 다중 로봇 루프
        for r_name, r_obj in manager.robots.items():
            is_me = (r_obj == robot)
            r_st = r_obj.state
            
            final_canvas = mc.draw_overlays(
                final_canvas, 
                r_st.latest_pose, 
                r_st.latest_scan, 
                axis, lidar, 
                st.viewer_map_info, # 맵 정보는 현재 로봇 기준
                r_st.nav_target, 
                jsons if is_me else None, 
                canvas_scale=1.0, 
                virtual_track=r_st.virtual_track,
                is_active=is_me,
                view_state=st.view_state # 뷰 상태 전달
            )
        
        st.view_state["active_crop"] = was_active

        # 5. 에디터 프리뷰 (녹색 점/선)
        # cfg.MSIS_MAPEDITOR -> st.editor_tool 로 변경
        if not st.view_state["active_crop"] and len(st.editor_tool.points) > 0:
            prev_img = final_canvas.copy()
            s = st.view_state["scale"]
            ox, oy = st.view_state["offset"]
            
            screen_pts = []
            for p in st.editor_tool.points:
                px = int(p[0] * s + ox)
                py = int(p[1] * s + oy)
                screen_pts.append((px, py))
                cv2.circle(prev_img, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)
            
            if len(screen_pts) >= 2:
                line_pts = np.array(screen_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(prev_img, [line_pts], False, (0, 255, 255), 2, cv2.LINE_AA)
            
            if len(screen_pts) >= 3:
                 cv2.line(prev_img, screen_pts[-1], screen_pts[0], (0, 0, 255), 1, cv2.LINE_AA)
                 
            final_canvas = prev_img

        try:
            cfg.artifacts["pois"] = robot.get_pois()
            cfg.artifacts["virtual_walls"] = robot.get_lines("virtual_wall")
        except: pass

        st.current_view_image = final_canvas 
        return final_canvas
    
    except Exception as e:
        print(f"Error in update_viewer_map: {e}")
        traceback.print_exc()
        return cfg.DEFAULT_MAP_IMAGE
    
    
# =========================================================
# File IO
# =========================================================
# [수정] 파일 목록 가져오기 (PNG, XML 모두 지원)
def list_map_files():
    """XML 및 PNG 파일 목록 반환"""
    if not os.path.exists(cfg.IMAGE_DIR): return []
    
    # [수정] .xml 확장자도 포함하도록 변경
    # 대소문자 구분 없이 검색 (lower() 사용)
    return [f for f in os.listdir(cfg.IMAGE_DIR) 
            if f.lower().endswith(('.png', '.xml'))]

def save_editor_map_to_disk(filename):
    robot = manager.get_current_robot()
    if not robot or robot.state.last_rendered_map is None:
        return "No map to save", gr.update()
    
    st = robot.state
    base = st.last_rendered_map.copy()
    
    # 1. 고해상도 이미지 생성 및 그리기
    hi_res_base = cv2.resize(base, None, fx=RENDER_SCALE, fy=RENDER_SCALE, interpolation=cv2.INTER_NEAREST)
    hi_res_base = _apply_map_boundary_cleanup(hi_res_base, RENDER_SCALE)
    st.obstacle_tool.update_map_image(hi_res_base)
    st.obstacle_tool.draw_obstacles(canvas_scale=RENDER_SCALE)
    
    target_img = hi_res_base
    
    # 저장할 데이터 준비
    save_map_info = st.viewer_map_info.copy()
    save_view_state = {"active_crop": False, "crop_rect": None, "crop_poly": None, "scale": 1.0, "offset": (0, 0)}
    is_cropped = False

    # Crop 처리 로직 (기존과 동일)
    if st.view_state["active_crop"] and st.view_state["crop_poly"] is not None:
        try:
            scaled_poly = (st.view_state["crop_poly"] * RENDER_SCALE).astype(np.int32)
            mask = np.zeros(hi_res_base.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_poly], 255)
            
            bg = np.full_like(hi_res_base, (150, 150, 150))
            fg = cv2.bitwise_and(hi_res_base, hi_res_base, mask=mask)
            bg_p = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
            processed = cv2.add(fg, bg_p)
            
            # 외곽선 그리기
            cv2.polylines(processed, [scaled_poly], True, (150, 150, 150), int(2 * RENDER_SCALE), cv2.LINE_AA)

            rx, ry, rw, rh = cv2.boundingRect(scaled_poly)
            if rw > 0 and rh > 0:
                cropped = processed[ry:ry+rh, rx:rx+rw]
                target_w, target_h = cfg.IMG_W, cfg.IMG_H
                scale = min(target_w / rw, target_h / rh)
                new_w, new_h = int(rw * scale), int(rh * scale)
                
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                final = np.full((target_h, target_w, 3), (150, 150, 150), dtype=np.uint8)
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2
                final[y_off:y_off+new_h, x_off:x_off+new_w] = resized
                
                target_img = final
                
                # 좌표 정보 업데이트
                total_scale = scale * RENDER_SCALE
                new_res = st.viewer_map_info["res"] / total_scale
                sx, sy = utils.pixel_to_world_with_info(rx/RENDER_SCALE, ry/RENDER_SCALE, st.viewer_map_info)
                new_ox = sx - ((target_w - 1 - x_off) * new_res)
                new_oy = sy - (y_off * new_res)
                
                save_map_info = { "x": new_ox, "y": new_oy, "w": target_w, "h": target_h, "res": new_res }
                is_cropped = True
        except Exception as e:
            print(f"Error processing crop: {e}")
            target_img = base

    # -------------------------------------------------------------
    # [핵심 수정] XML 파일 생성 및 저장
    # -------------------------------------------------------------
    import time
    timestamp = time.strftime("%m%d_%H%M")
    if not filename: filename = f"map_{timestamp}"
    if not filename.endswith(".xml"): filename += ".xml"
    save_path = os.path.join(cfg.IMAGE_DIR, filename)

    try:
        # 1. 이미지 -> PNG 포맷 -> Base64 문자열 변환
        _, buffer = cv2.imencode('.png', target_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # 2. XML 구조 생성
        root = ET.Element("MapData")
        
        # (A) 이미지 데이터 저장
        img_elem = ET.SubElement(root, "Image")
        img_elem.text = img_str
        
        # (B) 맵 정보 저장
        info_elem = ET.SubElement(root, "MapInfo")
        ET.SubElement(info_elem, "Resolution").text = str(save_map_info.get("res", 0.05))
        ET.SubElement(info_elem, "OriginX").text = str(save_map_info.get("x", 0.0))
        ET.SubElement(info_elem, "OriginY").text = str(save_map_info.get("y", 0.0))
        ET.SubElement(info_elem, "Width").text = str(save_map_info.get("w", cfg.IMG_W))
        ET.SubElement(info_elem, "Height").text = str(save_map_info.get("h", cfg.IMG_H))

        # (C) 뷰 상태 저장 (복원용)
        view_elem = ET.SubElement(root, "ViewState")
        # 간단하게 JSON 문자열로 저장
        view_elem.text = json.dumps(save_view_state) 
        
        # (D) 기타 메타데이터
        meta_elem = ET.SubElement(root, "Metadata")
        ET.SubElement(meta_elem, "IsCropped").text = str(is_cropped)
        ET.SubElement(meta_elem, "Created").text = timestamp

        # 3. XML 파일 쓰기 (예쁘게 정렬)
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        return f"Saved XML: {filename}", gr.update(choices=list_map_files())
        
    except Exception as e:
        return f"Error saving XML: {e}", gr.update()
# [수정] 통합 로드 함수 (XML / PNG 분기 처리)
def load_editor_map_from_disk(filename):
    robot = manager.get_current_robot()
    if not robot or not filename: return None, "No file/robot"
    
    file_path = os.path.join(cfg.IMAGE_DIR, filename)
    if not os.path.exists(file_path): return None, "File not found"
    
    st = robot.state
    
    # 확장자 확인
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # ---------------------------------------------------------
    # [CASE A] XML 파일 로드 (새로운 방식)
    # ---------------------------------------------------------
    if ext == '.xml':
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            st.editor_tool.clear() 

            # 1. 맵 정보 파싱
            info_node = root.find("MapInfo")
            if info_node is not None:
                loaded_info = {
                    "res": float(info_node.find("Resolution").text),
                    "x": float(info_node.find("OriginX").text),
                    "y": float(info_node.find("OriginY").text),
                    "w": int(info_node.find("Width").text),
                    "h": int(info_node.find("Height").text)
                }
                st.viewer_map_info.update(loaded_info)

            # 2. 이미지 데이터 디코딩
            img_node = root.find("Image")
            if img_node is not None and img_node.text:
                img_data = base64.b64decode(img_node.text)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                st.last_rendered_map = img
            else:
                return None, "Error: No image data in XML"

            # 3. 뷰 상태 초기화
            st.view_state = { "active_crop": False, "crop_rect": None, "crop_poly": None, "scale": 1.0, "offset": (0, 0) }
            st.is_loaded_mode = True
            
            return img, f"Loaded XML: {filename}"

        except Exception as e:
            print(traceback.format_exc())
            return None, f"Failed to load XML: {e}"

    # ---------------------------------------------------------
    # [CASE B] PNG 파일 로드 (기존 방식)
    # ---------------------------------------------------------
    elif ext == '.png':
        try:
            img = cv2.imread(file_path)
            img_h, img_w = img.shape[:2]
            
            st.view_state = { "active_crop": False, "crop_rect": None, "crop_poly": None, "scale": 1.0, "offset": (0, 0) }
            st.editor_tool.clear()
            
            info_path = file_path.replace(".png", ".json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    data = json.load(f)
                    if "map_info" in data:
                        # 리사이즈 감지 보정 로직
                        json_w = data["map_info"].get("w", img_w)
                        json_res = data["map_info"].get("res", 0.05)
                        if abs(json_w - img_w) > 5 and json_w > 0:
                             scale_factor = json_w / float(img_w)
                             data["map_info"]["res"] = json_res * scale_factor
                             data["map_info"]["w"] = img_w
                             data["map_info"]["h"] = img_h
                        st.viewer_map_info.update(data["map_info"])
                    elif "x" in data:
                        st.viewer_map_info.update(data)

            st.is_loaded_mode = True
            st.last_rendered_map = img
            return img, f"Loaded PNG: {filename}"
        
        except Exception as e:
            return None, f"Failed to load PNG: {e}"

    return None, "Unsupported file format"


def auto_load_latest_map():
    try:
        files = list_map_files()
        if files: 
            # 최신 파일 (XML/PNG 무관)
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(cfg.IMAGE_DIR, f)))
            load_editor_map_from_disk(latest_file)
    except: pass

auto_load_latest_map()

# =========================================================
# Mapping & Actions
# =========================================================
def update_mapping_map(axis, lidar):
    robot = manager.get_current_robot()
    if not robot: return cfg.DEFAULT_MAP_IMAGE
    st = robot.state

    try:
        grid, info = mc.fetch_explore_map_data(robot)
        if grid is None: return cfg.DEFAULT_MAP_IMAGE
        
        st.mapping_map_info.update(info)
        base = mc.explore_to_slamtec_editor_style(grid)
        base = mc.enhance_map(base)
        base = cv2.flip(base, 1)

        # 1. 고해상도 확대 (4배)
        hi_res_base = cv2.resize(base, None, fx=RENDER_SCALE, fy=RENDER_SCALE, interpolation=cv2.INTER_NEAREST)
        # hi_res_base = _apply_map_boundary_cleanup(hi_res_base, RENDER_SCALE)
        
        st.obstacle_tool.update_map_image(hi_res_base)
        st.obstacle_tool.draw_obstacles(canvas_scale=RENDER_SCALE)

        # 2. 화면 맞춤 (Fit-to-Screen) 계산
        h, w = hi_res_base.shape[:2]
        target_w, target_h = cfg.IMG_W, cfg.IMG_H
        
        # 화면 비율에 맞춘 스케일 계산
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 리사이징 및 중앙 정렬
        resized = cv2.resize(hi_res_base, (new_w, new_h), interpolation=cv2.INTER_AREA)
        final_canvas = np.full((target_h, target_w, 3), (150, 150, 150), dtype=np.uint8)
        
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        final_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # [핵심] 클릭 좌표 역변환을 위해 뷰 파라미터를 로봇 상태에 저장
        st.mapping_view_params = {
            "scale": scale,       # 고해상도 -> 화면 비율
            "offset": (x_off, y_off),
            "render_scale": RENDER_SCALE
        }

        # 3. 오버레이 그리기 (좌표 변환을 위해 임시 뷰 상태 생성)
        # map_core에게 "이미지가 scale만큼 줄었고 offset만큼 이동했다"고 알려줌
        temp_view = {
            "active_crop": True, # 좌표 변환 강제 활성화
            "crop_rect": (0, 0, base.shape[1], base.shape[0]), # 원본 맵 전체
            "scale": scale * RENDER_SCALE, # 원본 -> 화면 최종 비율
            "offset": (x_off, y_off)
        }

        # Trajectory Update
        if st.latest_pose is not None:
            rx, ry, _ = st.latest_pose
            px, py = utils.world_to_pixel_with_info(rx, ry, st.mapping_map_info)
            if not st.trajectory or st.trajectory[-1] != (px, py):
                st.trajectory.append((px, py))
        
        scan = robot.get_laserscan()
        
        img = mc.draw_overlays(final_canvas, st.latest_pose, scan, axis, lidar, st.mapping_map_info, st.nav_target, 
                               canvas_scale=1.0, is_active=True, view_state=temp_view)
        
        return img
    
    except Exception as e:
        print(f"Error in update_mapping_map: {e}")
        return cfg.DEFAULT_MAP_IMAGE
    
def reset_mapping_action():
    robot = manager.get_current_robot()
    if robot:
        robot.state.last_rendered_map = None
        robot.state.is_loaded_mode = False
        robot.state.view_state = {"active_crop": False, "scale": 1.0, "offset": (0, 0)}
        return robot.reset_map()
    return "No robot"

def toggle_mapping(enable):
    robot = manager.get_current_robot()
    if robot:
        if enable: robot.state.is_loaded_mode = False
        return robot.enable_mapping(enable)
    return "No robot"

# =========================================================
# Logic: Path & Click
# =========================================================
# [NEW] 고급 직각 이동 경로 계산 (후진 + L자 이동)
# def calculate_orthogonal_path(start_pose, end_point, final_yaw=None):
#     """
#     1. 현재 각도가 X축에 가까우면 X축 이동 우선, Y축에 가까우면 Y축 이동 우선
#     2. 장애물 충돌 시 차선책 선택
#     3. 목표 지점 진입 시 직선 이동 보장
#     """
#     sx, sy, syaw_rad = start_pose
#     ex, ey = end_point
    
#     path = []

#     # 1. 직선 경로 (거리가 매우 가까울 때)
#     if math.hypot(ex - sx, ey - sy) < 0.1:
#         if final_yaw is not None:
#             return [(sx, sy, final_yaw), (ex, ey, final_yaw)]
#         return [(sx, sy, math.degrees(syaw_rad)), (ex, ey, math.degrees(syaw_rad))]

#     # 충돌 체크 헬퍼
#     def check_segment(p1, p2):
#         dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
#         if dist < 0.05: return False
#         steps = int(dist / 0.05) 
#         if steps == 0: steps = 1
#         for i in range(steps + 1):
#             t = i / steps
#             x = p1[0] + (p2[0] - p1[0]) * t
#             y = p1[1] + (p2[1] - p1[1]) * t
            
#             px, py = utils.world_to_pixel_with_info(x, y, cfg.viewer_map_info)
#             if cfg.MSIS_OBSTACLE.is_point_inside(px, py): return True
#             if cfg.last_rendered_map is not None:
#                 h, w = cfg.last_rendered_map.shape[:2]
#                 if 0 <= px < w and 0 <= py < h:
#                     if np.mean(cfg.last_rendered_map[py, px]) < 100: return True
#         return False

#     def is_path_safe(p_list):
#         for i in range(len(p_list)-1):
#             if check_segment(p_list[i], p_list[i+1]): return False
#         return True

#     # 2. 현재 각도 분석 (X축 정렬 vs Y축 정렬)
#     curr_deg = math.degrees(syaw_rad)
#     # -180 ~ 180 정규화
#     curr_deg = (curr_deg + 180) % 360 - 180
    
#     # X축에 가까움: -45~45(0도, 동), 135~180/-180~-135(180도, 서)
#     is_x_aligned = (abs(curr_deg) <= 45) or (abs(curr_deg) >= 135)
    
#     # 3. 경로 후보 생성
#     # Path A (X축 우선): Start -> (ex, sy) -> End
#     # X축으로 먼저 이동한 뒤 Y축으로 진입
#     path_a_pts = [(sx, sy), (ex, sy), (ex, ey)]
    
#     # Path B (Y축 우선): Start -> (sx, ey) -> End
#     # Y축으로 먼저 이동한 뒤 X축으로 진입
#     path_b_pts = [(sx, sy), (sx, ey), (ex, ey)]
    
#     # 4. 경로 선택
#     selected_pts = []
    
#     if is_x_aligned:
#         # 현재 X축을 보고 있다면, X축 이동(Path A)을 먼저 시도
#         if is_path_safe(path_a_pts):
#             selected_pts = path_a_pts
#         elif is_path_safe(path_b_pts):
#             # 막혔다면 Y축 이동(Path B) 시도
#             selected_pts = path_b_pts
#         else:
#             # 둘 다 막히면 직선
#             selected_pts = [(sx, sy), (ex, ey)]
#     else:
#         # 현재 Y축을 보고 있다면, Y축 이동(Path B)을 먼저 시도
#         if is_path_safe(path_b_pts):
#             selected_pts = path_b_pts
#         elif is_path_safe(path_a_pts):
#             # 막혔다면 X축 이동(Path A) 시도
#             selected_pts = path_a_pts
#         else:
#             selected_pts = [(sx, sy), (ex, ey)]

#     # 5. 경로 점에 Yaw 정보 추가 (직선 진입 보장)
#     # 첫 점(Start)은 현재 Yaw 유지
#     path.append((selected_pts[0][0], selected_pts[0][1], curr_deg))
    
#     for i in range(1, len(selected_pts)):
#         p_prev = selected_pts[i-1]
#         p_curr = selected_pts[i]
        
#         # 이동 방향 각도 계산 (이 각도로 진입해야 직선 운동임)
#         move_yaw = math.degrees(math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0]))
        
#         # 만약 마지막 점이고, JSON에서 지정된 final_yaw가 있다면?
#         # -> "마지막에 들어갈 때 해당 각도의 직선운동"을 하려면, 
#         #    마지막 구간의 이동 방향(move_yaw)과 final_yaw가 일치해야 가장 이상적임.
#         #    하지만 일치하지 않을 경우(예: 위쪽에서 진입하는데 목표는 오른쪽을 봐야 함),
#         #    도착 후 제자리 회전을 해야 함.
#         #    요청하신 "직선운동하면서 들어가게 해줘"는 진입 경로의 각도를 의미하는 것으로 해석됨.
        
#         # 코너를 돌기 위해 제자리 회전 추가 (이전 점 위치에서 방향만 바꿈)
#         path.append((p_prev[0], p_prev[1], move_yaw))
        
#         # 이동
#         path.append((p_curr[0], p_curr[1], move_yaw))

#     # 6. 최종 Yaw 정렬 (JSON 지정값 등)
#     if final_yaw is not None:
#         last_pt = path[-1]
#         # 위치는 그대로 두고 각도만 변경 (제자리 회전)
#         path.append((last_pt[0], last_pt[1], final_yaw))

#     return path

# def orthogonal_move_thread(target, final_yaw):
#     cfg.is_tracking = True
#     print(f"[Ortho] Started to {target}")
    
#     try:
#         if not cfg.latest_pose: return
        
#         # [수정] final_yaw까지 넘겨서 경로 계산
#         points = calculate_orthogonal_path(cfg.latest_pose, target, final_yaw)
        
#         # 화면 표시용 업데이트
#         cfg.virtual_track = [(p[0], p[1]) for p in points]
        
#         # 순차 이동
#         for i in range(len(points)):
#             if not cfg.is_tracking: break
            
#             pt = points[i]
#             # pt = (x, y, target_yaw)
            
#             # 현재 위치와 목표가 너무 가까우면 스킵 (중복 점 등)
#             if i > 0:
#                 prev = points[i-1]
#                 if math.hypot(pt[0]-prev[0], pt[1]-prev[1]) < 0.01 and abs(pt[2]-prev[2]) < 1.0:
#                     continue

#             print(f"[Ortho] Step {i}: Go to ({pt[0]:.2f}, {pt[1]:.2f}) Yaw {pt[2]:.1f}")
#             api.send_move_to(pt[0], pt[1], yaw=pt[2])
            
#             # 도착 대기
#             st = time.time()
#             timeout = 30
#             while cfg.is_tracking and time.time() - st < timeout:
#                 if not cfg.latest_pose: 
#                     time.sleep(0.5); continue
                
#                 cx, cy, cyaw_rad = cfg.latest_pose
#                 dist = math.hypot(pt[0] - cx, pt[1] - cy)
                
#                 # 각도 차이
#                 cyaw_deg = math.degrees(cyaw_rad)
#                 yaw_diff = abs(pt[2] - cyaw_deg)
#                 yaw_diff = (yaw_diff + 180) % 360 - 180
                
#                 # 거리 10cm, 각도 5도 이내 도착 확인
#                 if dist < 0.10 and abs(yaw_diff) < 5.0: 
#                     break
#                 time.sleep(0.2)
                
#     except Exception as e:
#         print(f"[Ortho] Error: {e}")
#         traceback.print_exc()
#     finally:
#         cfg.is_tracking = False
#         print("[Ortho] Finished")


# [수정] 직각 이동 경로 계산 (꼭짓점만 추출하여 단순화)
# [수정] Orthogonal Path Logic (cfg 대신 state 사용)
def calculate_orthogonal_path(start_pose, end_point, final_yaw=None, map_info=None):
    sx, sy, syaw_rad = start_pose
    ex, ey = end_point
    
    # 1. 직선
    if math.hypot(ex - sx, ey - sy) < 0.1:
        if final_yaw is not None: return [(sx, sy, final_yaw), (ex, ey, final_yaw)]
        return [(sx, sy, math.degrees(syaw_rad)), (ex, ey, math.degrees(syaw_rad))]

    robot = manager.get_current_robot()
    
    def check_segment(p1, p2):
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 0.05: return False
        steps = int(dist / 0.05) or 1
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            
            px, py = utils.world_to_pixel_with_info(x, y, map_info or robot.state.viewer_map_info)
            if robot and robot.state.obstacle_tool.is_point_inside(px, py): return True
            # 벽 체크 등 추가 가능
        return False

    def is_path_safe(pts):
        for i in range(len(pts)-1):
            if check_segment(pts[i], pts[i+1]): return False
        return True

    # (이하 알고리즘 로직 동일)
    curr_deg = (math.degrees(syaw_rad) + 180) % 360 - 180
    start_is_x = (abs(curr_deg) <= 45) or (abs(curr_deg) >= 135)
    
    if final_yaw is not None:
        end_is_x = (abs((final_yaw + 180)%360-180) <= 45) or (abs((final_yaw + 180)%360-180) >= 135)
    else: end_is_x = not start_is_x

    mid_x, mid_y = (sx + ex)/2, (sy + ey)/2
    
    path_z_x = [(sx, sy), (mid_x, sy), (mid_x, ey), (ex, ey)]
    path_z_y = [(sx, sy), (sx, mid_y), (ex, mid_y), (ex, ey)]
    path_l_xy = [(sx, sy), (ex, sy), (ex, ey)]
    path_l_yx = [(sx, sy), (sx, ey), (ex, ey)]

    pq = []
    if start_is_x and end_is_x: pq = [path_z_x, path_l_xy, path_l_yx, path_z_y]
    elif not start_is_x and not end_is_x: pq = [path_z_y, path_l_yx, path_l_xy, path_z_x]
    elif start_is_x: pq = [path_l_xy, path_z_x, path_z_y, path_l_yx]
    else: pq = [path_l_yx, path_z_y, path_z_x, path_l_xy]

    sel = [(sx, sy), (ex, ey)]
    for cand in pq:
        if is_path_safe(cand): sel = cand; break
            
    final = []
    final.append((sel[0][0], sel[0][1], curr_deg))
    for i in range(1, len(sel)):
        p_prev, p_curr = sel[i-1], sel[i]
        if math.hypot(p_curr[0]-p_prev[0], p_curr[1]-p_prev[1]) < 0.01: continue
        yaw = math.degrees(math.atan2(p_curr[1]-p_prev[1], p_curr[0]-p_prev[0]))
        final.append((p_prev[0], p_prev[1], yaw))
        final.append((p_curr[0], p_curr[1], yaw))
        
    if final_yaw is not None:
        final.append((final[-1][0], final[-1][1], final_yaw))

    return final

# [수정] 스레드 로직 (한 번에 전송하여 고속 이동)
def orthogonal_move_thread(target, final_yaw):
    robot = manager.get_current_robot()
    if not robot: return
    st = robot.state
    st.is_tracking = True
    
    try:
        if not st.latest_pose: return
        pts = calculate_orthogonal_path(st.latest_pose, target, final_yaw, st.viewer_map_info)
        st.virtual_track = [(p[0], p[1]) for p in pts]
        
        for i, pt in enumerate(pts):
            if not st.is_tracking: break
            target_yaw = pt[2]
            
            robot.send_move_to(pt[0], pt[1], yaw=target_yaw)
            
            s = time.time()
            while st.is_tracking and time.time()-s < 30:
                if not st.latest_pose: time.sleep(0.5); continue
                cx, cy, cyaw = st.latest_pose
                d = math.hypot(pt[0]-cx, pt[1]-cy)
                yd = abs(target_yaw - math.degrees(cyaw))
                yd = (yd + 180) % 360 - 180
                if d < 0.10 and abs(yd) < 5.0: break
                time.sleep(0.2)
    except: traceback.print_exc()
    finally: st.is_tracking = False

def execute_move_action(nav_type):
    robot = manager.get_current_robot()
    if not robot: return "⚠️ No robot selected."
    st = robot.state
    
    if not st.nav_target: return "⚠️ Set target first."
    
    tx, ty = st.nav_target
    tyaw = st.nav_target_yaw

    # [수정] UI 라디오 버튼 이름("Linear")에 맞춤
    if nav_type == "Linear" or nav_type == "Target Point":
        st.virtual_track = []
        if st.target_name and "charge" in str(st.target_name).lower():
            api.go_charge() # robot.go_charge가 아니라 api 모듈 사용 시 주의. robot.go_charge() 권장
            # 여기서는 robot 객체에 go_charge 메서드가 있다고 가정 (api.py 리팩토링 시 추가했다면)
            robot.go_charge() 
            return f"⚡ Charging..."
        
        # 일반 이동
        robot.send_move_to(tx, ty, yaw=tyaw)
        yaw_msg = f", Yaw {tyaw:.1f}" if tyaw else ""
        return f"🚀 Linear Move to ({tx:.2f}, {ty:.2f}){yaw_msg}..."
    
    # [수정] UI 라디오 버튼 이름("Orthogonal")에 맞춤 (Orthogonal Move 포함)
    elif "Orthogonal" in nav_type:
        if st.is_tracking: return "⚠️ Already tracking."
        
        # 스레드 시작
        import threading
        threading.Thread(target=orthogonal_move_thread, args=(st.nav_target, st.nav_target_yaw), daemon=True).start()
        return "📐 Orthogonal Move Started"

    return f"Unknown Type: {nav_type}"

def stop_tracking_action():
    robot = manager.get_current_robot()
    if robot: 
        robot.state.is_tracking = False
        return f"{robot.stop_now()} (Cleared)"
    return "No robot"


def execute_track_action():
    """ 'Follow Points' 버튼 클릭 시 호출 """
    robot = manager.get_current_robot()
    if not robot: return "⚠️ No robot selected."
    st = robot.state

    if not st.virtual_track:
        return "⚠️ No points. Click map to draw points."
    
    # 1. 로봇 위치 확인
    if st.latest_pose is None:
        # 위치를 모르면 그냥 전체 경로 전송
        success, msg = robot.follow_path(st.virtual_track)
        return f"▶️ {msg}"

    rx, ry, _ = st.latest_pose
    
    # 2. 가장 가까운 포인트 찾기
    closest_index = 0
    min_dist = float('inf')

    for i, point in enumerate(st.virtual_track):
        px, py = point
        dist = math.hypot(px - rx, py - ry)
        if dist < min_dist:
            min_dist = dist
            closest_index = i
            
    # 3. 경로 슬라이싱 (가까운 점부터 시작)
    resume_path = st.virtual_track[closest_index:]
    
    # [안전장치] 경로 점이 1개뿐이면 로봇이 동작 안 할 수 있음 -> 현재 위치를 시작점으로 추가
    if len(resume_path) == 1:
        # 현재 위치에서 마지막 점으로 이동
        resume_path = [(rx, ry)] + resume_path
    elif len(resume_path) == 0:
        # 이미 다 지나왔으면 다시 처음부터? 아니면 종료
        # 여기서는 마지막 점으로 이동 유도
        if st.virtual_track:
            resume_path = [(rx, ry), st.virtual_track[-1]]
        else:
            return "⚠️ End of track reached."

    print(f"[Track] Sending {len(resume_path)} points: {resume_path}")
    
    success, msg = robot.follow_path(resume_path)
    
    if success:
        return f"▶️ Moving ({closest_index + 1}/{len(st.virtual_track)} pts)"
    else:
        return f"❌ Failed: {msg}"

# [수정] Clear 기능 통합 (Target + Track)
def clear_track_action():
    robot = manager.get_current_robot()
    if robot:
        robot.state.virtual_track = []
        return "🗑️ Track Cleared"
    return "No robot"

def clear_target_action():
    robot = manager.get_current_robot()
    if robot:
        robot.state.nav_target = None
        robot.state.target_name = None
        robot.state.nav_target_yaw = None
        robot.state.virtual_track = [] # 선도 같이 지움
        return "Target & Track Cleared"
    return "No robot"

# =========================================================
# Obstacle / Tool Helpers
# =========================================================
def update_obstacle_props(size, angle, label):
    robot = manager.get_current_robot()
    if robot:
        robot.state.obstacle_tool.update_selected_property(size=size, angle=angle, label=label)
        return "Updated Properties"
    return "No robot"

def get_selected_label():
    robot = manager.get_current_robot()
    if robot:
        _, _, label = robot.state.obstacle_tool.get_selected_info()
        return label
    return ""
def set_pending_shape(shape_type):
    if shape_type == "free": 
        return "free", "✏️ Click map to add points"
    if "brush" in shape_type:
        color = shape_type.replace("brush_", "").capitalize()
        return shape_type, f"🖌️ Painting {color}. Click to stamp."
    return shape_type, f"📍 Click map to place {shape_type}"

def create_object(shape_type):
    # 이 함수는 예전 방식의 잔재일 수 있으나, 호환성을 위해 유지
    robot = manager.get_current_robot()
    if not robot: return shape_type
    
    cx, cy = 200, 200
    if robot.state.last_rendered_map is not None:
        h, w = robot.state.last_rendered_map.shape[:2]
        cx, cy = w // 2, h // 2
        
    if shape_type == "free": return shape_type
    robot.state.obstacle_tool.add_shape(shape_type, cx, cy, size=20)
    return shape_type

def obstacle_undo():
    robot = manager.get_current_robot()
    if robot:
        return robot.state.obstacle_tool.undo()
    return "No robot"

def obstacle_delete():
    robot = manager.get_current_robot()
    if robot:
        return robot.state.obstacle_tool.delete_selected()
    return "No robot"

def obstacle_clear():
    robot = manager.get_current_robot()
    if robot:
        robot.state.obstacle_tool.clear()
        return "All Cleared"
    return "No robot"

# =========================================================
# Sync Map to Robot
def save_map_to_robot_action():
    robot = manager.get_current_robot()
    if robot:
        success, msg = robot.sync_map_to_robot()
        return f"🤖 {msg}"
    return "No robot"

# =========================================================
# JSON Marker / Crop Tools
# =========================================================
def save_json_marker(label, color, current_selection):
    robot = manager.get_current_robot()
    if not robot or not robot.state.latest_pose: 
        return "No Pose Data", gr.update()
    if not label or not label.strip(): 
        return "Empty Label", gr.update()

    if current_selection is None: current_selection = []
        
    x, y, yaw_rad = robot.state.latest_pose
    yaw_deg = math.degrees(yaw_rad)
    
    # RGBA -> HEX 변환
    if color.startswith("rgba") or color.startswith("rgb"):
        try:
            content = color.split("(")[1].split(")")[0]
            parts = content.split(",")
            r, g, b = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
            color = f"#{r:02X}{g:02X}{b:02X}"
        except: pass

    path = os.path.join(cfg.DATA_DIR, f"{label}.json")
    data = {
        "map_name": label, 
        "positions": [{
            "name": label, "x": x, "y": y, "yaw": yaw_deg, "color": color
        }]
    }
    
    try:
        with open(path, "w") as f: json.dump(data, f, indent=2)
        new_file = f"{label}.json"
        all_files = [f for f in os.listdir(cfg.DATA_DIR) if f.endswith(".json")]
        if new_file not in current_selection: current_selection.append(new_file)
        return f"Saved: {label}.json", gr.update(choices=all_files, value=current_selection)
    except Exception as e:
        return f"Save Error: {e}", gr.update()

def delete_json(files):
    if files:
        for f in files:
            p = os.path.join(cfg.DATA_DIR, f)
            if os.path.exists(p): os.remove(p)
    new_choices = [f for f in os.listdir(cfg.DATA_DIR) if f.endswith(".json")]
    return gr.update(choices=new_choices, value=[])

def crop_actions(action):
    robot = manager.get_current_robot()
    if not robot: return
    st = robot.state
    
    if action == "clear":
        st.editor_tool.clear()
        st.view_state["active_crop"] = False
        st.view_state["crop_poly"] = None
    elif action == "undo":
        st.editor_tool.undo()
    elif action == "apply":
        if len(st.editor_tool.points) >= 3:
            st.view_state["active_crop"] = True
            st.view_state["crop_poly"] = np.array(st.editor_tool.points, dtype=np.int32)
            st.view_state["crop_rect"] = cv2.boundingRect(st.view_state["crop_poly"])

# =========================================================
# Click Handler
# =========================================================      
# [수정] handle_click 함수
def handle_click(mode, shape_pending, size, angle, active_jsons, nav_type, 
                 text_content, text_color, evt: gr.SelectData, is_svg=False):
    robot = manager.get_current_robot()
    if not robot: return "No robot", shape_pending, size, angle
    
    st = robot.state
    px, py = evt.index
    if active_jsons is None: active_jsons = []

    # 좌표 역변환 (현재 View State 기준)
    scale = st.view_state.get("scale", 1.0)
    offset = st.view_state.get("offset", (0, 0))
    crop_rect = st.view_state.get("crop_rect", (0, 0, 0, 0))
    start_x, start_y = crop_rect[0], crop_rect[1]
    
    if scale > 0:
        px = int((px - offset[0]) / scale + start_x)
        py = int((py - offset[1]) / scale + start_y)
    
    if st.last_rendered_map is not None:
        h, w = st.last_rendered_map.shape[:2]
        px = max(0, min(px, w-1))
        py = max(0, min(py, h-1))

    # Obstacle Check
    if mode in ["move", "track"]:
        if st.obstacle_tool.is_point_inside(px, py): return "⛔ Blocked!", shape_pending, size, angle
        if st.last_rendered_map is not None and np.mean(st.last_rendered_map[py, px]) < 100:
            return "⛔ Wall!", shape_pending, size, angle

    # Move Mode
    if mode == "move":
        wx, wy = utils.pixel_to_world_with_info(px, py, st.viewer_map_info)
        clicked_pos = None
        clicked_name = ""
        clicked_yaw = None
        min_dist = float('inf')
        thresh = 15.0 / (scale if st.view_state["active_crop"] else 1.0)
        if thresh < 10: thresh = 10

        if os.path.exists(cfg.DATA_DIR):
            for f_name in os.listdir(cfg.DATA_DIR):
                if f_name.endswith(".json") and f_name in active_jsons:
                    try:
                        with open(os.path.join(cfg.DATA_DIR, f_name), "r") as f:
                            data = json.load(f)
                            for pos in data.get("positions", []):
                                jx, jy = utils.world_to_pixel_with_info(pos["x"], pos["y"], st.viewer_map_info)
                                d = math.hypot(px-jx, py-jy)
                                if d < thresh and d < min_dist:
                                    min_dist = d
                                    clicked_pos = (pos["x"], pos["y"])
                                    clicked_name = pos.get("name", "Unknown")
                                    clicked_yaw = pos.get("yaw", None)
                    except: pass
        
        if clicked_pos:
            st.selected_marker = clicked_pos
            st.nav_target = clicked_pos
            st.nav_target_yaw = clicked_yaw
            st.target_name = clicked_name
            msg = f"📍 Selected: {clicked_name}"
            
            if nav_type == "Orthogonal Move" and st.latest_pose:
                path = calculate_orthogonal_path(st.latest_pose, clicked_pos, clicked_yaw, st.viewer_map_info)
                st.virtual_track = [(p[0], p[1]) for p in path]
                msg += " (Preview)"
            else: st.virtual_track = []
            return msg, shape_pending, size, angle
        else:
            st.selected_marker = None
            st.nav_target = (wx, wy)
            st.nav_target_yaw = None
            st.target_name = None
            msg = f"Target: {wx:.2f}, {wy:.2f}"
            
            if nav_type == "Orthogonal Move" and st.latest_pose:
                path = calculate_orthogonal_path(st.latest_pose, st.nav_target, None, st.viewer_map_info)
                st.virtual_track = [(p[0], p[1]) for p in path]
                msg += " (Preview)"
            else: st.virtual_track = []
            return msg, shape_pending, size, angle

    # Track Mode
    elif mode == "track":
        wx, wy = utils.pixel_to_world_with_info(px, py, st.viewer_map_info)
        st.virtual_track.append((wx, wy))
        return "Track Pt Added", shape_pending, size, angle

    # Obstacle Mode
    elif mode == "obstacle":
        if shape_pending == "text_tool":
            return st.obstacle_tool.set_preview_text(px, py, text_content, size, text_color), shape_pending, size, angle
        
        if shape_pending and shape_pending != "none":
            if "brush" in shape_pending:
                return st.obstacle_tool.add_brush(px, py, shape_pending.replace("brush_", ""), size, angle), shape_pending, size, angle
            if shape_pending == "rectangle":
                return st.obstacle_tool.add_rect_point(px, py), shape_pending, size, angle
            if shape_pending == "free": 
                return st.obstacle_tool.add_free_point(px, py), shape_pending, size, angle
            if shape_pending == "line": 
                return st.obstacle_tool.add_line_point(px, py), shape_pending, size, angle
            
            st.obstacle_tool.add_shape(shape_pending, px, py, size)
            return "Shape Added", None, size, 0
        
        if st.obstacle_tool.select_object(px, py):
             s, a, _ = st.obstacle_tool.get_selected_info()
             return "Selected", None, s, a
        return "None Selected", None, size, angle

    # Crop Mode
    elif mode == "crop":
        st.editor_tool.add_point(px, py)
        return "Crop Pt Added", shape_pending, size, angle

    return "Clicked", shape_pending, size, angle

def handle_mapping_click(evt: gr.SelectData):
    robot = manager.get_current_robot()
    if not robot: return "No robot"
    
    st = robot.state
    # 저장된 뷰 파라미터가 없으면(아직 렌더링 전이면) 기본값 사용
    view_params = getattr(st, "mapping_view_params", {"scale": 1.0, "offset": (0,0), "render_scale": 1.0})
    
    # 1. 화면 클릭 좌표 (픽셀)
    screen_x, screen_y = evt.index
    
    # 2. 역변환: (화면 - 오프셋) / 화면스케일 = 고해상도 이미지 좌표
    scale = view_params["scale"]
    off_x, off_y = view_params["offset"]
    
    if scale <= 0: scale = 1.0 # 0 나누기 방지
    
    hi_res_x = (screen_x - off_x) / scale
    hi_res_y = (screen_y - off_y) / scale
    
    # 3. 원본 해상도로 변환: 고해상도 / 렌더스케일
    render_scale = view_params["render_scale"]
    orig_px = int(hi_res_x / render_scale)
    orig_py = int(hi_res_y / render_scale)
    
    # 4. 맵 정보 유효성 확인
    if not st.mapping_map_info or st.mapping_map_info.get("res", 0) == 0:
        return "Map info not ready"

    # 5. 픽셀 -> 월드 좌표 변환
    wx, wy = utils.pixel_to_world_with_info(orig_px, orig_py, st.mapping_map_info)
    
    # 이동 명령 전송
    robot.send_move_to(wx, wy)
    return f"MoveTo {wx:.2f}, {wy:.2f}"

# [수정] Mapping 탭에서 PC로 맵 저장 (XML 포맷)
def save_current_map():
    """Mapping 탭의 실시간 맵을 XML로 저장"""
    robot = manager.get_current_robot()
    if not robot: return "No robot"
    
    # 1. 맵 데이터 가져오기
    grid, info = mc.fetch_explore_map_data(robot)
    if grid is None: return "❌ No Map Data"

    # 2. 이미지 가공
    slam = mc.explore_to_slamtec_editor_style(grid)
    slam = mc.enhance_map(slam)
    slam = cv2.flip(slam, 1) 
    
    target_img = slam

    # 3. XML 변환 및 저장
    import time
    timestamp = time.strftime("%m%d_%H%M")
    filename = f"mapping_{timestamp}.xml" # 확장자 xml
    save_path = os.path.join(cfg.IMAGE_DIR, filename)

    try:
        # (A) 이미지를 Base64 문자열로 인코딩
        _, buffer = cv2.imencode('.png', target_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # (B) XML 구조 생성
        root = ET.Element("MapData")
        
        # 이미지 태그
        img_elem = ET.SubElement(root, "Image")
        img_elem.text = img_str
        
        # 맵 정보 태그
        info_elem = ET.SubElement(root, "MapInfo")
        ET.SubElement(info_elem, "Resolution").text = str(info.get("res", 0.05))
        ET.SubElement(info_elem, "OriginX").text = str(info.get("x", 0.0))
        ET.SubElement(info_elem, "OriginY").text = str(info.get("y", 0.0))
        ET.SubElement(info_elem, "Width").text = str(info.get("w", cfg.IMG_W))
        ET.SubElement(info_elem, "Height").text = str(info.get("h", cfg.IMG_H))

        # 뷰 상태 (매핑 탭은 기본값)
        view_elem = ET.SubElement(root, "ViewState")
        default_view = {"active_crop": False, "crop_rect": None, "crop_poly": None, "scale": 1.0, "offset": (0, 0)}
        view_elem.text = json.dumps(default_view)
        
        # 메타데이터
        meta_elem = ET.SubElement(root, "Metadata")
        ET.SubElement(meta_elem, "Source").text = "Real-time Mapping"
        ET.SubElement(meta_elem, "Created").text = timestamp

        # (C) 파일 쓰기
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
            
        return f"💾 Saved XML: {filename}"

    except Exception as e:
        print(f"Save Error: {e}")
        return f"❌ Error: {e}"

# =========================================================
# Text Tool Actions (로봇별 상태 사용)
# =========================================================
def apply_text_action():
    robot = manager.get_current_robot()
    if robot:
        return robot.state.obstacle_tool.apply_text()
    return "No robot selected"

def update_text_preview(c, s, cl):
    robot = manager.get_current_robot()
    if robot:
        robot.state.obstacle_tool.update_preview_props(c, s, cl)
        return "Updated"
    return "No robot selected"
# =========================================================
# Robot Connection Management (Manager 위임)
# =========================================================

def add_new_robot(name, ip, port):
    if not name or not ip:
        return "⚠️ Name and IP are required.", gr.update(), manager.get_connection_status_string()
    
    success, msg = manager.add_robot(name, ip, port)
    
    if success:
        manager.select_robot(name)
        # [수정] outputs 개수에 맞춰 상태 문자열 추가 (만약 app.py에서 outputs를 늘린다면)
        # 현재 app.py는 outputs=[txt_conn_log, dd_robots] 이므로 2개만 리턴
        return f"✅ {msg}", gr.update(choices=list(cfg.ROBOT_LIST.keys()), value=name)
    else:
        return f"⚠️ {msg}", gr.update()

def delete_robot(name):
    manager.delete_robot(name)
    # [수정] 삭제 후 갱신된 상태 반환을 위해 app.py 연결 필요할 수 있음
    # 일단 기존 호환성 유지
    return f"🗑️ Deleted {name}", gr.update(choices=list(cfg.ROBOT_LIST.keys()), value=None)

def connect_robot(name):
    """저장된 목록에서 로봇을 선택하여 연결/전환"""
    if not name:
        return "⚠️ Select a robot first", manager.get_connection_status_string()
    
    if name not in cfg.ROBOT_LIST:
        return "⚠️ Robot info not found", manager.get_connection_status_string()
        
    info = cfg.ROBOT_LIST[name]
    
    # 연결 시도
    success, msg = manager.add_robot(name, info['ip'], info['port'])
    
    if success:
        manager.select_robot(name)
        # [핵심 수정] 전체 로봇 상태 문자열 반환
        status_str = manager.get_connection_status_string()
        return f"Switching to {name}...", status_str
    else:
        return f"❌ Connection Failed: {msg}", manager.get_connection_status_string()

def disconnect_robot():
    """현재 선택된 로봇의 연결을 해제"""
    robot = manager.get_current_robot()
    if robot:
        name = robot.name
        manager.delete_robot(name) # 리스트에서 제거 = 연결 끊기
        
        # [핵심 수정] 갱신된 상태 문자열 반환
        status_str = manager.get_connection_status_string()
        return f"⚪ Disconnected {name}", status_str
        
    return "No active robot", manager.get_connection_status_string()


# [NEW] 3D 맵 변환 및 로드 (수정됨)
def convert_to_3d_action():
    # 1. 현재 로봇 가져오기
    robot = manager.get_current_robot()
    if not robot: return None
    
    # 2. 최신 맵 데이터 가져오기 (탐사 데이터)
    # [수정] robot 인자 전달 필수
    grid, info = mc.fetch_explore_map_data(robot)
    
    if grid is None:
        return None
    
    # 3. 3D 변환 (PLY 파일 생성)
    ply_path = converter_3d.generate_3d_ply(grid, info)
    
    # 파일 경로 확인 후 반환
    if ply_path and os.path.exists(ply_path):
        return os.path.abspath(ply_path)
    
    return None

# [NEW] SVG 클릭 처리를 위한 래퍼 함수
def handle_svg_click_wrapper(coords_json, mode, shape_pending, size, angle, active_jsons, nav_type, txt_content, txt_color):
    """
    JS에서 보낸 좌표({"x": 123, "y": 456})를 받아서 처리하고,
    변경된 맵(SVG)을 반환합니다.
    """
    if not coords_json:
        return "No coords", shape_pending, size, angle, gr.update()

    try:
        # 1. JSON 파싱
        coords = json.loads(coords_json)
        px = coords['x']
        py = coords['y']
        
        # 2. SelectData 형식으로 가짜 이벤트 객체 생성 (기존 함수 재사용을 위해)
        # 또는 handle_click 함수를 수정해서 x, y를 직접 받게 해도 됨
        class MockSelectData:
            def __init__(self, index):
                self.index = index
        
        evt = MockSelectData([px, py])

        # 3. 기존 로직 실행
        # 주의: handle_click 내부에서 좌표 변환(Crop 역변환) 로직이 있다면, 
        # SVG 방식에서는 JS가 이미 월드 좌표를 보냈는지, 화면 좌표를 보냈는지에 따라 
        # handle_click 내부의 좌표 변환 로직을 건너뛰어야 할 수도 있음.
        
        # SVG 방식은 보통 '맵 좌표' 자체를 유지하므로 추가 역변환이 필요 없을 수 있음.
        # 따라서 여기서는 handle_click을 직접 부르지 않고 로직을 분리하거나,
        # handle_click 내부의 scale/offset 로직을 SVG 모드일 땐 타지 않게 해야 함.
        
        # (간단하게 handle_click을 호출하되, SVG 모드임을 알리는 플래그가 필요할 수 있음)
        msg, next_shape, next_sz, next_ang = handle_click(
            mode, shape_pending, size, angle, active_jsons, nav_type, txt_content, txt_color, evt
        )
        
        # 4. 맵 갱신 (SVG 다시 그리기)
        # 맵 데이터(XML/Obstacles)가 변경되었을 수 있으므로 화면 갱신 필요
        # 현재 화면 상태를 반영한 새로운 SVG 문자열을 반환해야 함
        new_svg_content = update_svg_map_display() # 이 함수는 아래에 정의 필요
        
        return msg, next_shape, next_sz, next_ang, new_svg_content

    except Exception as e:
        print(f"SVG Click Error: {e}")
        return f"Error: {e}", shape_pending, size, angle, gr.update()

# [NEW] SVG 화면 갱신 함수 (xml_data.py와 연동 가정)
def update_svg_map_display():
    robot = manager.get_current_robot()
    if not robot: return "<div>No Robot</div>"
    
    # 로봇 위치 등 최신 정보 가져오기
    pose = robot.state.latest_pose
    
    # XML/SVG 매니저를 통해 최신 SVG 문자열 생성
    # (xml_data.py의 XMLMapData 클래스 사용 가정)
    from modules.map.xml_data import xml_manager # 상단 import로 이동 권장
    return xml_manager.get_svg_content(robot_pose=pose)