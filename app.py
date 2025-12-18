import os
import gradio as gr

from modules.map import config as cfg
from modules.map import assets
from modules.map import handler as h



with gr.Blocks(fill_height=True) as demo:

    # ============================================================
    # Header & AI Voice
    # ============================================================
    with gr.Row(elem_id="msis-header"):
        with gr.Column(scale=6):
            gr.Markdown("## 🤖 MSIS AMR Control System")
        with gr.Column(scale=2):
            # 현재 연결된 로봇 상태 표시
            lbl_conn_status = gr.Label(value="Disconnected", label="Current Robot", show_label=True)

    # [NEW] 로봇 관리 매니저 (아코디언)
    with gr.Accordion("🔌 Robot Connection Manager", open=False):
        with gr.Row():
            # 왼쪽: 로봇 추가
            with gr.Column(scale=1):
                gr.Markdown("### Add New Robot")
                with gr.Row():
                    in_name = gr.Textbox(label="Name", placeholder="AMR_01", scale=1)
                    in_ip = gr.Textbox(label="IP", placeholder="192.168.0.100", scale=2)
                    in_port = gr.Textbox(label="Port", value="1448", scale=1)
                btn_add_robot = gr.Button("➕ Add to List")
            
            # 오른쪽: 로봇 선택 및 연결
            with gr.Column(scale=1):
                gr.Markdown("### Select & Connect")
                # 저장된 로봇 목록 불러오기
                saved_robots = list(cfg.ROBOT_LIST.keys())
                dd_robots = gr.Dropdown(choices=saved_robots, label="Saved Robots", interactive=True)
                
                with gr.Row():
                    btn_connect = gr.Button("🔗 Connect / Switch", variant="primary")
                    btn_disconnect = gr.Button("⚪ Disconnect")
                    btn_delete_robot = gr.Button("🗑️ Delete", variant="stop")
                
                txt_conn_log = gr.Markdown("")

    gr.Markdown("# MSIS AMR")
    with gr.Tabs():
        # --- TAB 1: Map Editor ---
        with gr.TabItem("🗺️ Map Editor"):
            with gr.Row():
                # 1. 왼쪽: 맵 뷰어
                with gr.Column(scale=9):
                    map_image = gr.Image(
                        value=cfg.DEFAULT_MAP_IMAGE,
                        elem_id="map_viewer",
                        show_label=False,
                        interactive=True,
                        sources=[],
                        container=False,
                        format="jpeg"
                    )
                    
                # 2. 오른쪽: 컨트롤 패널
                with gr.Column(scale=3, min_width=350):
                    
                    # [상태 변수]
                    ui_tab_state = gr.State(value="move_panel")
                    interact_mode = gr.State(value="move")
                    pending_shape_state = gr.State(value=None)

                    # ---------------------------------------------------------
                    # [A] Top Level Mode Buttons
                    # ---------------------------------------------------------
                    gr.Markdown("### 🔧 Mode Select")
                    with gr.Row():
                        mode_btn_move = gr.Button("Move", variant="primary", scale=1, min_width=80)
                        mode_btn_obs = gr.Button("Obstacle", variant="secondary", scale=1, min_width=80)
                        mode_btn_Edit = gr.Button("Edit", variant="secondary", scale=1, min_width=80)
                        
                    # ---------------------------------------------------------
                    # [B] Control Panels
                    # ---------------------------------------------------------

                    # 1. Move & Track Integrated Panel
                    with gr.Group(visible=True, elem_classes="control-panel-card") as grp_move:
                        
                        gr.Markdown("### Move Controls")
                        move_type_radio = gr.Radio(
                            choices=["Linear", "Orthogonal", "Multiple Points"],
                            value="Linear",
                            label="Navigation Type",
                            interactive=True,
                            elem_classes="nav-radio-group"
                        )
                        
                        status_info = gr.Markdown("Ready.")

                        # 1-A. Target Point UI
                        with gr.Column(visible=True) as col_target:
                            gr.Markdown("Click map to set **Target** (Purple Dot)")
                            with gr.Row(equal_height=True):
                                btn_go = gr.Button("🚀 Go / Start", variant="primary", scale=2)
                                btn_stop_move = gr.Button("■ STOP", elem_classes="stop-btn", scale=1)
                            btn_clear_target = gr.Button("❌ Clear Target")

                        # 1-B. Orthogonal Move UI
                        with gr.Column(visible=False) as col_ortho:
                            gr.Markdown("Click map to set **Target** (Auto Path)")
                            with gr.Row(equal_height=True):
                                btn_go_ortho = gr.Button("📐 Go Orthogonal", variant="primary", scale=2)
                                btn_stop_ortho = gr.Button("■ STOP", elem_classes="stop-btn", scale=1)
                            btn_clear_ortho = gr.Button("❌ Clear Path")

                        # 1-C. Virtual Track UI
                        with gr.Column(visible=False) as col_track:
                            gr.Markdown("Click map to draw **Points** (Yellow Line)")
                            with gr.Row(equal_height=True):
                                btn_follow = gr.Button("▶️ Follow Points", variant="primary", scale=2)
                                btn_stop_track = gr.Button("■ STOP", elem_classes="stop-btn", scale=1)
                            btn_clear_track = gr.Button("🗑️ Clear Points")

                        with gr.Accordion("📁 Point Tools", open=False):
                            with gr.Group(elem_classes="json-compact-card"):
                                with gr.Row(equal_height=True):
                                    j_color = gr.ColorPicker(label="Color", value="#10D63B", scale=1, min_width=80, container=True)
                                with gr.Row(equal_height=True):
                                    j_label = gr.Textbox(label="Label", placeholder="Point A", value="Point A", scale=3, min_width=100)
                                    j_save = gr.Button("💾 Save", scale=1, min_width=60, variant="primary")
                                with gr.Row(elem_classes="compact-input", equal_height=True):
                                    with gr.Column(scale=2): gr.Markdown("**Saved List**")
                                    j_refresh = gr.Button("🔄", scale=0, min_width=40, size="sm")
                                    j_del = gr.Button("🗑️", scale=0, min_width=40, size="sm", variant="stop")
                                with gr.Group(elem_classes="json-scroll-list"):
                                    j_list = gr.CheckboxGroup(
                                        label="", 
                                        choices=[f for f in os.listdir(cfg.DATA_DIR) if f.endswith(".json")], 
                                        value=[], 
                                        container=False
                                    )
                                j_status = gr.Markdown("", elem_id="json_status")


                    # 2. Obstacle Panel
                    with gr.Group(visible=False, elem_classes="control-panel-card") as grp_obstacle:
                        gr.Markdown("### 🧊 Object Tools")

                        # [NEW] Text Tool
                        with gr.Accordion("📝 Text Tool", open=False):
                            with gr.Row():
                                btn_text_mode = gr.Button("T Input Mode", variant="secondary", size="sm")
                                btn_text_apply = gr.Button("✔ Apply", variant="primary", size="sm")
                            obs_text_input = gr.Textbox(show_label=False, placeholder="Type text content...", interactive=True)
                            obs_text_color = gr.ColorPicker(label="Text Color", value="#000000")
                                
                        # [2] 도형 도구 (아이콘 위주 배치)
                        gr.Markdown("**Shapes**")
                        with gr.Row(elem_classes="tool-row"):
                            btn_squ = gr.Button("■", size="sm", min_width=40)
                            btn_rect = gr.Button("▅", size="sm", min_width=40)
                            btn_circle = gr.Button("●", size="sm", min_width=40)
                            btn_tri = gr.Button("▲", size="sm", min_width=40)
                            btn_line = gr.Button("📏", size="sm", min_width=40)
                        
                        with gr.Row(elem_classes="tool-row"):
                            btn_pent = gr.Button("⬠", size="sm", min_width=40)
                            btn_semi = gr.Button("◐", size="sm", min_width=40)
                            btn_free = gr.Button("✏️ Free", size="sm", min_width=60)

                        # [3] 브러시 도구 (색상 선택 추가)
                        gr.Markdown("**Paint (Brush)**")
                        with gr.Row(elem_classes="tool-row"):
                            btn_brush_black = gr.Button("⬛ Wall", size="sm")   # 검은색 (장애물)
                            btn_brush_gray = gr.Button("⬜ Unkown", size="sm")  # 회색 (미탐사)
                            btn_brush_white = gr.Button("⚪ Clear", size="sm")  # 흰색 (지우개)

                        gr.Markdown("---")
                        # [4] 속성 수정 (라벨, 크기, 회전)
                        gr.Markdown("**Modify Selected**")
                        obs_label_input = gr.Textbox(label="Label", placeholder="Enter name...", interactive=True)

                        with gr.Row():
                            obs_size_slider = gr.Slider(1, 100, value=5, step=1, label="Size")
                        with gr.Row():
                            obs_angle_slider = gr.Slider(0, 360, value=0, step=10, label="Rotate")
                        
                        # [5] 편집 버튼
                        with gr.Row():
                            btn_obs_undo = gr.Button("↩ Undo", size="sm")
                            btn_del_obj = gr.Button("🗑️ Del", variant="stop", size="sm")
                            btn_clear_all = gr.Button("💥 Reset", variant="secondary", size="sm")


                    # 3. Crop Panel
                    with gr.Group(visible=False, elem_classes="control-panel-card") as grp_crop:
                        gr.Markdown("### 🧩 Crop Tools")
                        with gr.Row():
                            undo_crop = gr.Button("Undo")
                            clear_crop = gr.Button("Clear")
                            apply_crop = gr.Button("Apply", variant="primary")

                    # Settings
                    with gr.Group(elem_classes="control-panel-card"):
                        with gr.Accordion("🔧 Settings", open=False):
                            with gr.Row():
                                show_axis = gr.Checkbox(label="Axis", value=False, min_width=60)
                                show_lidar = gr.Checkbox(label="Lidar", value=True, min_width=60)
                    
                    # Map File Ops
                    with gr.Accordion("💾 Map File Ops", open=False):
                        with gr.Group(elem_classes="control-panel-card"):
                            gr.Markdown("#### Save/Load Map")
                            with gr.Row(equal_height=True):
                                save_name = gr.Textbox(show_label=False, placeholder="File name...", scale=2, container=False)
                                btn_save_disk = gr.Button("💾 Save", size="sm", scale=1)
                            with gr.Row(equal_height=True):
                                map_file_dropdown = gr.Dropdown(
                                    choices=[f for f in os.listdir(cfg.IMAGE_DIR) if f.endswith(".png")],
                                    show_label=False, container=False, scale=2, interactive=True
                                )
                                btn_refresh_files = gr.Button("🔄", size="sm", scale=0, min_width=40)
                                btn_load_disk = gr.Button("📂 Load", size="sm", scale=1)
                            file_status = gr.Markdown("", elem_classes="status-text")

        # --- TAB 2: Mapping ---
        with gr.TabItem("📡 Real-time Mapping"):
            with gr.Row():
                with gr.Column(scale=9):
                    mapping_image = gr.Image(value=cfg.DEFAULT_MAP_IMAGE, elem_id="mapping_viewer", interactive=True, sources=[], show_label=False, container=False, format="jpeg")
                with gr.Column(scale=3):
                    map_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Group(elem_classes="control-panel-card"):
                        with gr.Accordion("🔧 Settings", open=True):
                            with gr.Row():
                                map_show_axis = gr.Checkbox(label="Axis", value=False , min_width=60)
                                map_show_lidar = gr.Checkbox(label="Lidar", value=True, min_width=60)
                    with gr.Accordion("📡 Mapping Controls", open=True):
                        with gr.Row():
                            map_enable = gr.Button("▶ Start Mapping", variant="primary")
                            map_disable = gr.Button("⏹ Stop Mapping")
                        with gr.Row():
                            btn_save_pc = gr.Button("💾 Save to PC", size="sm")
                            btn_save_robot = gr.Button("🤖 Save to Robot", size="sm", variant="secondary")
                            map_reset = gr.Button("💥 Reset Map", variant="stop")

           
            # ============================================================
            # Logic & Events
            # ============================================================
            
            timer1 = gr.Timer(0.05)
            timer1.tick(h.update_viewer_map, inputs=[j_list, show_axis, show_lidar], outputs=[map_image]).then(None, None, None, js="setup_pan_zoom")
            timer2 = gr.Timer(0.2)
            timer2.tick(h.update_mapping_map, inputs=[map_show_axis, map_show_lidar], outputs=[mapping_image]).then(None, None, None, js="setup_pan_zoom")

            # Mode Switching
            def set_ui_panel(btn_name, current_radio_val):
                show_move = (btn_name == "move")
                show_obs = (btn_name == "obstacle")
                show_crop = (btn_name == "crop")
                
                if btn_name == "move":
                    if current_radio_val == "Virtual Track": next_mode = "track"
                    elif current_radio_val == "Orthogonal Move": next_mode = "orthogonal"
                    else: next_mode = "move"
                elif btn_name == "obstacle": next_mode = "obstacle"
                else: next_mode = "crop"

                return (
                    gr.update(visible=show_move), 
                    gr.update(visible=show_obs), 
                    gr.update(visible=show_crop),
                    gr.update(variant="primary" if show_move else "secondary"),
                    gr.update(variant="primary" if show_obs else "secondary"),
                    gr.update(variant="primary" if show_crop else "secondary"),
                    next_mode 
                )

            def on_radio_change(radio_val):
                show_target = (radio_val == "Linear")
                show_ortho = (radio_val == "Orthogonal")
                show_track = (radio_val == "Multiple Points")
                next_mode = "track" if show_track else "move"
                return (gr.update(visible=show_target), gr.update(visible=show_ortho), gr.update(visible=show_track), next_mode)

            top_btn_outputs = [grp_move, grp_obstacle, grp_crop, mode_btn_move, mode_btn_obs, mode_btn_Edit, interact_mode]
            mode_btn_move.click(lambda r: set_ui_panel("move", r), inputs=[move_type_radio], outputs=top_btn_outputs)
            mode_btn_obs.click(lambda r: set_ui_panel("obstacle", r), inputs=[move_type_radio], outputs=top_btn_outputs)
            mode_btn_Edit.click(lambda r: set_ui_panel("crop", r), inputs=[move_type_radio], outputs=top_btn_outputs)
            move_type_radio.change(on_radio_change, inputs=[move_type_radio], outputs=[col_target, col_ortho, col_track, interact_mode])

            # -----------------------------------------------------------
            # [Map Interaction] - 중요 수정!
            # -----------------------------------------------------------
            # [수정] 텍스트 도구 관련 입력(obs_text_input, obs_text_color) 추가
            def on_map_click(mode, shape_p, sz, ang, jsons, nav_type, txt_content, txt_color, evt: gr.SelectData):
                msg, next_shape, next_sz, next_ang = h.handle_click(
                    mode, shape_p, sz, ang, jsons, nav_type, 
                    txt_content, txt_color, # 추가됨
                    evt
                )
                return msg, next_shape, next_sz, next_ang

            map_image.select(
                h.handle_click, 
                inputs=[
                    interact_mode, pending_shape_state, 
                    obs_size_slider, obs_angle_slider, 
                    j_list, move_type_radio,
                    obs_text_input, obs_text_color # [NEW]
                ], 
                outputs=[status_info, pending_shape_state, obs_size_slider, obs_angle_slider]
            ).then(
                h.get_selected_label, 
                inputs=None,
                outputs=[obs_label_input]
            )

            # -----------------------------------------------------------
            # [Action Buttons]
            # -----------------------------------------------------------
            btn_go.click(fn=h.execute_move_action, inputs=[move_type_radio], outputs=[status_info])
            btn_stop_move.click(fn=h.stop_tracking_action, outputs=[status_info])
            btn_clear_target.click(fn=h.clear_target_action, outputs=[status_info])

            btn_follow.click(fn=h.execute_track_action, outputs=[status_info])
            btn_stop_track.click(fn=h.stop_tracking_action, outputs=[status_info])
            btn_clear_track.click(fn=h.clear_track_action, outputs=[status_info])

            btn_go_ortho.click(fn=h.execute_move_action, inputs=[move_type_radio], outputs=[status_info])
            btn_stop_ortho.click(fn=h.stop_tracking_action, outputs=[status_info])
            btn_clear_ortho.click(fn=h.clear_target_action, outputs=[status_info])

            # [NEW] Text Tool Events
            btn_text_mode.click(lambda: "text_tool", outputs=[pending_shape_state]) \
                        .then(lambda: "Click map to place text", outputs=[status_info])

            # 텍스트 속성 변경 시 미리보기 업데이트
            text_props = [obs_text_input, obs_size_slider, obs_text_color]
            obs_text_input.change(h.update_text_preview, inputs=text_props, outputs=[status_info])
            obs_text_color.change(h.update_text_preview, inputs=text_props, outputs=[status_info])
            obs_size_slider.change(h.update_text_preview, inputs=text_props, outputs=[status_info])

            btn_text_apply.click(h.apply_text_action, outputs=[status_info])

            # Shapes
            btn_squ.click(lambda: h.set_pending_shape("square"), outputs=[pending_shape_state, status_info])
            btn_rect.click(lambda: h.set_pending_shape("rectangle"), outputs=[pending_shape_state, status_info])
            btn_circle.click(lambda: h.set_pending_shape("circle"), outputs=[pending_shape_state, status_info])
            btn_tri.click(lambda: h.set_pending_shape("triangle"), outputs=[pending_shape_state, status_info])
            btn_line.click(lambda: h.set_pending_shape("line"), outputs=[pending_shape_state, status_info])
            btn_pent.click(lambda: h.set_pending_shape("pentagon"), outputs=[pending_shape_state, status_info])
            btn_semi.click(lambda: h.set_pending_shape("semicircle"), outputs=[pending_shape_state, status_info])
            btn_free.click(lambda: h.set_pending_shape("free"), outputs=[pending_shape_state, status_info])

            # [NEW] Brushes (색상별 연결)
            # handler.py에서 'brush_black', 'brush_white' 등을 처리하도록 되어 있어야 합니다 (이미 되어 있음)
            btn_brush_black.click(lambda: h.set_pending_shape("brush_black"), outputs=[pending_shape_state, status_info])
            btn_brush_gray.click(lambda: h.set_pending_shape("brush_gray"), outputs=[pending_shape_state, status_info])
            btn_brush_white.click(lambda: h.set_pending_shape("brush_white"), outputs=[pending_shape_state, status_info])
            
            # Properties
            prop_inputs = [obs_size_slider, obs_angle_slider, obs_label_input]
            obs_size_slider.change(h.update_obstacle_props, inputs=prop_inputs, outputs=[status_info])
            obs_angle_slider.change(h.update_obstacle_props, inputs=prop_inputs, outputs=[status_info])
            obs_label_input.change(h.update_obstacle_props, inputs=prop_inputs, outputs=[status_info])

            # Actions
            btn_obs_undo.click(lambda: cfg.MSIS_OBSTACLE.undo(), outputs=[status_info])
            btn_del_obj.click(lambda: cfg.MSIS_OBSTACLE.delete_selected(), outputs=[status_info])
            btn_clear_all.click(lambda: cfg.MSIS_OBSTACLE.clear(), outputs=[status_info])

            # Text Tool
            btn_text_mode.click(lambda: "text_tool", outputs=[pending_shape_state]) \
                        .then(lambda: "Click map to place text", outputs=[status_info])
            
            text_props = [obs_text_input, obs_size_slider, obs_text_color]
            obs_text_input.change(h.update_text_preview, inputs=text_props, outputs=[status_info])
            obs_text_color.change(h.update_text_preview, inputs=text_props, outputs=[status_info])
            btn_text_apply.click(h.apply_text_action, outputs=[status_info])

            # 4. Crop & File Tools
            clear_crop.click(lambda: h.crop_actions("clear")).then(h.update_viewer_map, inputs=[j_list, show_axis, show_lidar], outputs=[map_image])
            apply_crop.click(lambda: h.crop_actions("apply")).then(h.update_viewer_map, inputs=[j_list, show_axis, show_lidar], outputs=[map_image])
            undo_crop.click(lambda: h.crop_actions("undo")).then(h.update_viewer_map, inputs=[j_list, show_axis, show_lidar], outputs=[map_image])

            j_save.click(h.save_json_marker, inputs=[j_label, j_color, j_list], outputs=[j_status, j_list])
            j_refresh.click(lambda: gr.update(choices=[f for f in os.listdir(cfg.DATA_DIR) if f.endswith(".json")]), outputs=[j_list])
            j_del.click(h.delete_json, inputs=[j_list], outputs=[j_list])

            btn_save_disk.click(h.save_editor_map_to_disk, inputs=[save_name], outputs=[file_status, map_file_dropdown])
            btn_refresh_files.click(fn=lambda: gr.update(choices=h.list_map_files()), inputs=None, outputs=[map_file_dropdown])
            btn_load_disk.click(h.load_editor_map_from_disk, inputs=[map_file_dropdown], outputs=[map_image, file_status])

            # 5. Mapping Controls
            map_enable.click(lambda: h.toggle_mapping(True), outputs=[map_status])
            map_disable.click(lambda: h.toggle_mapping(False), outputs=[map_status])
            btn_save_pc.click(h.save_current_map, outputs=[map_status])
            btn_save_robot.click(h.save_map_to_robot_action, outputs=[map_status])
            map_reset.click(h.reset_mapping_action, outputs=[map_status])
            mapping_image.select(h.handle_mapping_click, outputs=[map_status])


            # ============================================================
            # Events for Connection
            # ============================================================
            
            # 1. 로봇 추가
            btn_add_robot.click(
                fn=h.add_new_robot,
                inputs=[in_name, in_ip, in_port],
                outputs=[txt_conn_log, dd_robots]
            )
            
            # 2. 로봇 삭제
            btn_delete_robot.click(
                fn=h.delete_robot,
                inputs=[dd_robots],
                outputs=[txt_conn_log, dd_robots]
            )
            
            # 3. 연결
            btn_connect.click(
                fn=h.connect_robot,
                inputs=[dd_robots],
                outputs=[txt_conn_log, lbl_conn_status] # 상태 메시지와 상단 라벨 업데이트
            )
            
            # 4. 연결 해제
            btn_disconnect.click(
                fn=h.disconnect_robot,
                inputs=None,
                outputs=[txt_conn_log, lbl_conn_status]
            )

    # demo.load(None, None, None, js=assets.COMBINED_JS)

demo.launch(share=True, css=assets.CUSTOM_CSS)
