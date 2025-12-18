import os
import json
import numpy as np 

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MAP_DIR = os.path.join(BASE_DIR, "map")
DATA_DIR = os.path.join(BASE_MAP_DIR, "map_data")
IMAGE_DIR = os.path.join(BASE_MAP_DIR, "map_image")
ROBOT_LIST_PATH = os.path.join(BASE_DIR, "robot_list.json")


## 뷰어 경로 정의
VIEWER_IMG_PATH = os.path.join(IMAGE_DIR, "map_image.png")
VIEWER_INFO_PATH = os.path.join(IMAGE_DIR, "map_image_info.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Constants ---
IMG_W, IMG_H = 1600, 900
RENDER_SCALE = 4.0

# [NEW] 초기화용 기본 맵 이미지 (회색 바탕)
DEFAULT_MAP_IMAGE = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)

# --- Robot Connection List (Persistent) ---
def load_robot_list():
    if os.path.exists(ROBOT_LIST_PATH):
        try:
            with open(ROBOT_LIST_PATH, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_robot_list(data):
    with open(ROBOT_LIST_PATH, "w") as f: json.dump(data, f, indent=4)

ROBOT_LIST = load_robot_list() # {"Name": {"ip": "...", "port": "..."}}