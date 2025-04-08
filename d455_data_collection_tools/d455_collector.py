"""
Author: Shang Xu
Feel free to use this code for your own purposes, but please give credit to the original author.
"""

import pyrealsense2 as rs
from queue import Queue
from threading import Thread
import time
import os
import shutil
import cv2
import numpy as np
import numba
import glob
from tqdm import tqdm


############################################################
# 全局设置
############################################################
FPS_DEPTH       = 90                    # 深度相机帧率
FPS_RGB         = 30                    # 彩色相机帧率
WIDTH           = 848                   # 图像宽度
HEIGHT          = 480                   # 图像高度
LASER_POWER     = 100                   # 激光功率 0% - 100%
POST_DEPTH      = True                  # 是否做深度后处理
ALIGN_DEPTH     = True                  # 是否对齐深度到彩色
MAX_TIME_DIFF   = 0.03                  # 容忍的最大时间差（秒），可自行调整
OUTPUT_DIR      = "rs_d455_data"        # 输出目录
CLEAN_LAST      = True                  # 是否清空上次的输出目录
RECORD          = True                  # 是否保存数据
VISUALIZE       = True                  # 是否可视化
PRESET          = 3                     # 高精度模式下的预设

depth_frames_queue  = Queue()
depth_post_queue    = Queue()
color_frames_queue  = Queue()
ir1_frames_queue    = Queue()
ir2_frames_queue    = Queue()
ready_queue         = Queue()
output_rgbd_queue   = Queue()
output_stereo_queue = Queue()
visual_queue        = Queue()
R, t, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c = None, None, None, None, None, None, None, None, None, None
color_intr, depth_intr, d2c_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr = None, None, None, None, None, None, None


############################################################
# 辅助函数
############################################################
def ensure_dir(dirpath):
    """确保目录存在。"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def remove_dir(dirpath):
    """删除整个目录。"""
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
        
def get_camera_params_from_device():
    """从连接的RealSense设备获取相机参数"""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("没有找到RealSense设备")
    
    device = devices[0]
    print(f"使用设备: {device.get_info(rs.camera_info.name)}")
    
    # 创建临时pipeline获取相机参数
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS_DEPTH)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS_RGB)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS_DEPTH)
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS_DEPTH)
    
    # 启动pipeline
    profile = pipeline.start(config)
    
    # 获取深度传感器和彩色传感器的内参
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ir1_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    
    depth_intr = depth_stream.get_intrinsics()
    color_intr = color_stream.get_intrinsics()
    ir1_intr = ir1_stream.get_intrinsics()
    ir2_intr = ir2_stream.get_intrinsics()
    
    # 获取深度到彩色的外参
    depth_to_color = depth_stream.get_extrinsics_to(color_stream)
    ir2_to_ir1 = ir2_stream.get_extrinsics_to(ir1_stream)
    ir1_to_ir2 = ir1_stream.get_extrinsics_to(ir2_stream)
    
    # 停止pipeline
    pipeline.stop()
    
    # 提取旋转矩阵和平移向量
    R = np.array(depth_to_color.rotation).reshape(3, 3)
    t = np.array(depth_to_color.translation)
    
    return color_intr, depth_intr, depth_to_color, ir1_intr, ir2_intr, ir2_to_ir1, ir1_to_ir2

@numba.njit(parallel=True, cache=True)
def align_depth(
    depth_image,
    R, t,
    fx_d, fy_d, cx_d, cy_d,
    fx_c, fy_c, cx_c, cy_c,
    out_height, out_width,
    depth_scale = 0.001
    ):
    """
    使用逐像素投影+Numba加速，将深度图对齐到彩色坐标系。
    返回对齐后的深度图(单位:与原depth_image相同)。
    """
    # 输出图像
    aligned = np.zeros((out_height, out_width), dtype=np.uint16)

    h_d, w_d = depth_image.shape
    scale_x = w_d / out_width
    scale_y = h_d / out_height
    fx_d = fx_d * scale_x
    fy_d = fy_d * scale_y
    cx_d = cx_d * scale_x
    cy_d = cy_d * scale_y
    

    for vd in numba.prange(h_d):
        for ud in range(w_d):
            z_raw = depth_image[vd, ud]
            if z_raw == 0:
                continue
            Z_d = z_raw * depth_scale
            if Z_d <= 0:
                continue

            # 1) 反投影到 depth 坐标系
            X_d = (ud - cx_d) * Z_d / fx_d
            Y_d = (vd - cy_d) * Z_d / fy_d

            # 2) 转到 color 坐标
            X_c = R[0,0]*X_d + R[0,1]*Y_d + R[0,2]*Z_d + t[0]
            Y_c = R[1,0]*X_d + R[1,1]*Y_d + R[1,2]*Z_d + t[1]
            Z_c = R[2,0]*X_d + R[2,1]*Y_d + R[2,2]*Z_d + t[2]

            if Z_c <= 0:
                continue

            # 3) 投影到 color 像素
            u_c = fx_c * (X_c / Z_c) + cx_c
            v_c = fy_c * (Y_c / Z_c) + cy_c

            uc_round = int(round(u_c))
            vc_round = int(round(v_c))

            if 0 <= uc_round < out_width and 0 <= vc_round < out_height:
                old_val = aligned[vc_round, uc_round]
                # 保留更近Z
                if old_val == 0 or old_val > z_raw:
                    aligned[vc_round, uc_round] = z_raw

    return aligned

def process_depth_directory():
    """
    只处理深度图像，不需要RGB图像
    """
    
    # 创建输出目录
    aligned_output_dir = os.path.join(OUTPUT_DIR, "depth_aligned")
    os.makedirs(aligned_output_dir, exist_ok=True)
    
    # 获取所有深度图像
    depth_files = glob.glob(os.path.join(OUTPUT_DIR, 'depth', '*.png'))
    
    print(f"找到 {len(depth_files)} 个深度图像")
    
    # 处理每个深度图像
    for depth_file in tqdm(depth_files):
        # 读取深度图像
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)  # 16位无符号整数
        
        if depth_img is None:
            print(f"Warning: 无法读取图像 - {depth_file}")
            continue
        
        # 对齐深度图
        aligned_depth = align_depth(
            depth_img,
            R, t,
            fx_d, fy_d, cx_d, cy_d,
            fx_c, fy_c, cx_c, cy_c,
            HEIGHT, WIDTH
        )
        
        # 保存对齐后的深度图
        output_file = os.path.join(aligned_output_dir, os.path.basename(depth_file))
        cv2.imwrite(output_file, aligned_depth)
    
    print(f"处理完成! {len(depth_files)} 个深度图像已对齐并保存到 {aligned_output_dir}")

def write_camera_profile():
    k1_c, k2_c, p1_c, p2_c, k3_c = color_intr.coeffs[:5]
    camera_params = {
        "Camera.fx":  fx_c,
        "Camera.fy":  fy_c,
        "Camera.cx":  cx_c,
        "Camera.cy":  cy_c,
        "Camera.k1":  k1_c,
        "Camera.k2":  k2_c,
        "Camera.p1":  p1_c,
        "Camera.p2":  p2_c,
        "Camera.k3":  k3_c,
        "Camera.fps": FPS_RGB,
        "Camera.RGB": 0,  # 如果你保存的是RGB格式，填 1；BGR 就填 0
        "ORBextractor.nFeatures": 1000,
        "ORBextractor.scaleFactor": 1.2,
        "ORBextractor.nLevels": 8,
        "ORBextractor.iniThFAST": 20,
        "ORBextractor.minThFAST": 7,
        "Viewer.KeyFrameSize": 0.05,
        "Viewer.KeyFrameLineWidth": 1,
        "Viewer.GraphLineWidth": 0.9,
        "Viewer.PointSize": 2,
        "Viewer.CameraSize": 0.08,
        "Viewer.CameraLineWidth": 3,
        "Viewer.ViewpointX": 0,
        "Viewer.ViewpointY": -0.7,
        "Viewer.ViewpointZ": -1.8,
        "Viewer.ViewpointF": 500,
    }
            
    with open(os.path.join(OUTPUT_DIR, "RS_D455.yaml"), 'w') as f:
        f.write("%YAML:1.0\n\n")
        for key, val in camera_params.items():
            f.write(f"{key}: {val}\n")
            
    # 获取IR1到IR2的变换（左IR到右IR）
    R_ir = np.array(l2r_extr.rotation).reshape(3, 3)
    t_ir = np.array(l2r_extr.translation)

    # 获取深度到彩色的变换（注意外参方向）
    R_dc = np.array(d2c_extr.rotation).reshape(3, 3)
    t_dc = np.array(d2c_extr.translation)

    # 构建投影矩阵（KITTI格式要求3x4矩阵，世界坐标系=左IR相机）
    # ---------------------------------------------------------------------
    # P0: 左IR相机（IR1）的投影矩阵（参考坐标系）
    K_ir1 = np.array([
        [ir1_intr.fx, 0, ir1_intr.ppx],
        [0, ir1_intr.fy, ir1_intr.ppy],
        [0, 0, 1]
    ])
    P0 = np.hstack((K_ir1, np.zeros((3, 1))))  # 无外参变换

    # P1: 右IR相机（IR2）的投影矩阵（左IR坐标系到右IR坐标系）
    K_ir2 = np.array([
        [ir2_intr.fx, 0, ir2_intr.ppx],
        [0, ir2_intr.fy, ir2_intr.ppy],
        [0, 0, 1]
    ])
    Rt_ir = np.hstack((R_ir, t_ir.reshape(3, 1)))  # [R|t]左IR到右IR
    P1 = K_ir2 @ Rt_ir  # 右IR内参 × 左IR到右IR变换

    # P2: 彩色相机的投影矩阵（左IR坐标系到彩色坐标系）
    K_color = np.array([
        [color_intr.fx, 0, color_intr.ppx],
        [0, color_intr.fy, color_intr.ppy],
        [0, 0, 1]
    ])
    # 假设color相机与左IR的外参为 R_ic, t_ic（需根据实际数据获取）
    # 若color相机参考系不同，需补充对应外参计算
    P2 = np.hstack((K_color, np.zeros((3, 1))))  # 假设与左IR坐标系对齐

    # P3: 深度相机的投影矩阵（左IR坐标系到深度坐标系）
    K_depth = np.array([
        [depth_intr.fx, 0, depth_intr.ppx],
        [0, depth_intr.fy, depth_intr.ppy],
        [0, 0, 1]
    ])
    # 深度到彩色的外参 => 左IR到深度的变换需根据实际标定数据推导
    # 假设存在从左IR到深度的外参 R_id, t_id（需补充实际数据）
    # 此处示例使用深度到彩色的逆变换（若深度坐标系独立）
    R_color_to_depth = R_dc.T
    t_color_to_depth = -R_dc.T @ t_dc
    Rt_dc = np.hstack((R_color_to_depth, t_color_to_depth.reshape(3, 1)))
    P3 = K_depth @ Rt_dc  # 深度内参 × 彩色到深度变换

    # 写入calib.txt
    with open(os.path.join(OUTPUT_DIR, "calib.txt"), 'w') as f:
        f.write(f"P0: {' '.join([f'{x:.12e}' for x in P0.flatten()])}\n")
        f.write(f"P1: {' '.join([f'{x:.12e}' for x in P1.flatten()])}\n")
        f.write(f"P2: {' '.join([f'{x:.12e}' for x in P2.flatten()])}\n")
        f.write(f"P3: {' '.join([f'{x:.12e}' for x in P3.flatten()])}\n")


############################################################
# 回调函数
############################################################
def depth_callback(frame):
    """
    当 Depth 或 IR 帧到来时会调用这个函数
    """
    # 判断帧类型
    st = frame.get_profile().stream_type()
    timestamp = frame.get_timestamp() / 1000.0
    if st == rs.stream.depth:
        depth_frames_queue.put((timestamp, frame))
        # print("Got depth frame, ts=", timestamp)
    elif st == rs.stream.infrared:
        idx = frame.get_profile().stream_index()
        if idx == 1:
            ir1_frames_queue.put((timestamp, frame))
            # print("Got IR1 frame, ts=", timestamp)
        elif idx == 2:
            ir2_frames_queue.put((timestamp, frame))
            # print("Got IR2 frame, ts=", timestamp)

def color_callback(frame):
    """
    当 Color 帧到来时会调用这个函数
    """
    timestamp = frame.get_timestamp() / 1000.0
    color_frames_queue.put((timestamp, frame))
    # print("Got color frame, ts=", timestamp)
    

############################################################
# 深度后处理线程
############################################################
def depth_post_thread():
    """
    不断从 depth_frames_queue 中取帧，做后处理。
    """
    ############################################################
    # 初始化滤波器
    ############################################################
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    depth2disparity = rs.disparity_transform(True)
    disparity2depth = rs.disparity_transform(False)
    threshold = rs.threshold_filter()


    ############################################################
    # 设置滤波器参数
    ############################################################
    decimation.set_option(rs.option.filter_magnitude, 2)
    threshold.set_option(rs.option.min_distance, 0.4)
    threshold.set_option(rs.option.max_distance, 6.0)
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 0)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.holes_fill, 3)
    
    
    while True:
        ts, frame = depth_frames_queue.get()
        if frame is None:
            # 收到停止信号
            depth_frames_queue.task_done()
            break
        filtered_depth = frame
        if POST_DEPTH:
            filtered_depth = decimation.process(filtered_depth)
            filtered_depth = threshold.process(filtered_depth)
            filtered_depth = depth2disparity.process(filtered_depth)
            filtered_depth = spatial.process(filtered_depth)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = disparity2depth.process(filtered_depth)
        
        depth_post_queue.put((ts, filtered_depth))
        depth_frames_queue.task_done()
        

############################################################
# 图像转换线程
############################################################
def img_convert_thread():
    """
    不断从 ready_queue 中取帧，转换成图片。
    """
    while True:
        # print("Output queues:", output_rgbd_queue.qsize(), output_stereo_queue.qsize())
        # print("Visual queues:", visual_queue.qsize())
        color_ts, color_frame, depth_ts, depth_frame, ir1_ts, ir1_frame, ir2_ts, ir2_frame = ready_queue.get()
        if color_frame is None:
            # 收到停止信号
            ready_queue.task_done()
            break
        
        # 转换成图片
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        # depth_img = align_depth(depth_img, R, t, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, HEIGHT // 2, WIDTH // 2)  # Align depth to color and resize
        # depth_img = align_depth_vectorized(depth_img, R, t, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, HEIGHT, WIDTH)  # Align depth to color and resize
        depth_img = cv2.resize(depth_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        ir1_img = np.asanyarray(ir1_frame.get_data())
        ir2_img = np.asanyarray(ir2_frame.get_data())
        
        if RECORD:
            output_rgbd_queue.put((color_ts, color_img, depth_ts, depth_img))
            output_stereo_queue.put((ir1_ts, ir1_img, ir2_ts, ir2_img))
        
        if VISUALIZE:
            visual_queue.put((color_ts, color_img, depth_ts, depth_img, ir1_ts, ir1_img, ir2_ts, ir2_img))
        
        ready_queue.task_done()

############################################################
# 文件写入线程
############################################################
def rgbd_write_thread():
    """
    不断从 output_rgbd_queue 中取帧，保存成图片。记录rgb.txt, depth.txt, associations.txt
    """
    while True:
        color_ts, color_img, depth_ts, depth_img = output_rgbd_queue.get()
        if color_img is None:
            # 收到停止信号
            output_rgbd_queue.task_done()
            break

        # 保存图片
        color_fname = f"{color_ts:.6f}.png"
        depth_fname = f"{depth_ts:.6f}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "rgb", color_fname), color_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "depth", depth_fname), depth_img)

        # 记录到文件
        with open(os.path.join(OUTPUT_DIR, "rgb.txt"), 'a') as f:
            f.write(f"{color_ts:.6f} rgb/{color_fname}\n")
        with open(os.path.join(OUTPUT_DIR, "depth.txt"), 'a') as f:
            f.write(f"{depth_ts:.6f} depth/{depth_fname}\n")
        with open(os.path.join(OUTPUT_DIR, "associations.txt"), 'a') as f:
            f.write(f"{color_ts:.6f} rgb/{color_fname} {depth_ts:.6f} depth/{depth_fname}\n")
        
        output_rgbd_queue.task_done()
        
def stereo_write_thread():
    """
    不断从 output_stereo_queue 中取帧，保存成图片。记录times.txt
    """
    frame_count = 0
    while True:
        ir1_ts, ir1_img, ir2_ts, ir2_img = output_stereo_queue.get()
        if ir1_img is None:
            # 收到停止信号
            output_stereo_queue.task_done()
            break

        # 保存图片, 6位数序号编码
        fname = f"{frame_count:06d}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "left", fname), ir1_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "right", fname), ir2_img)

        # 记录到文件
        with open(os.path.join(OUTPUT_DIR, "times.txt"), 'a') as f:
            f.write(f"{ir1_ts:.6f}\n")
        
        frame_count += 1
        
        output_stereo_queue.task_done()
        
        
############################################################
# 图像更新函数
############################################################
def visualize_step():
    """
    用于在主线程中调用，更新图像显示。
    """
    # 检查窗口是否已被关闭
    if cv2.getWindowProperty('RealSense', cv2.WND_PROP_VISIBLE) < 1:
        return False
    
    color_ts, color_img, depth_ts, depth_img, ir1_ts, ir1_img, ir2_ts, ir2_img = visual_queue.get()
    if color_img is None:
        # 收到停止信号
        visual_queue.task_done()
        return False

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
    images_U = np.hstack((color_img, depth_colormap))
    images_B = np.hstack((ir1_img, ir2_img))
    images_B = cv2.cvtColor(images_B, cv2.COLOR_GRAY2BGR)
    images = np.vstack((images_U, images_B))
    images = cv2.resize(images, (1280, 720))
    cv2.imshow('RealSense', images)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        return False

    visual_queue.task_done()
    return True


############################################################
# 匹配辅助函数
############################################################
def find_best_match_and_discard(queue: Queue, target_ts: float):
    """
    从 `queue` 中取出所有可用帧，找出与 `target_ts` 最接近的那一帧。
    丢弃所有 timestamp < target_ts 的帧。
    返回 (best_ts, best_frame) 或 (None, None) 如果队列里完全没帧可用。
    
    注意：本函数会把没使用到的帧再放回队列（只放回那些 timestamp >= target_ts 的）。
    """
    best_ts = None
    best_frame = None
    best_diff = float('inf')

    leftover = []
    # 先把队列里的所有帧全部取出
    while not queue.empty():
        ts, frame = queue.get()
        # 对每个帧进行判断
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
            best_frame = frame
        leftover.append((ts, frame))

    # 丢弃所有 timestamp < target_ts 的帧
    # 只把 timestamp >= target_ts 的帧放回 queue
    for (ts, frame) in leftover:
        if ts >= best_ts - MAX_TIME_DIFF:
            queue.put((ts, frame))

    # print("best_diff=", best_diff, "best_ts=", best_ts)
    # 如果最优帧与 target_ts 的差值太大，认为没匹配到
    if best_ts is not None and abs(best_ts - target_ts) > MAX_TIME_DIFF:
        return None, None
    return best_ts, best_frame


############################################################
# 匹配线程：以 color 为主，找 depth + IR1 + IR2
############################################################
def match_thread():
    """
    不断从 color_frames_queue 中取帧，
    与 depth_queue / ir1_queue / ir2_queue 做时间戳匹配。
    匹配成功后，可做后续操作（保存/缓存/可视化）。
    """
    fail_tolerance = 5
    fail_count = 0
    while True:
        # print(color_frames_queue.qsize(), depth_frames_queue.qsize(), ir1_frames_queue.qsize(), ir2_frames_queue.qsize(), depth_post_queue.qsize(), ready_queue.qsize())
        # 取一帧 color
        color_ts, color_frame = color_frames_queue.get()
        if color_frame is None:
            # 收到停止信号
            color_frames_queue.task_done()
            break

        # 与 depth 匹配
        depth_ts, depth_frame = find_best_match_and_discard(depth_post_queue, color_ts)

        # 与 IR1 匹配
        ir1_ts, ir1_frame = find_best_match_and_discard(ir1_frames_queue, color_ts)

        # 与 IR2 匹配
        ir2_ts, ir2_frame = find_best_match_and_discard(ir2_frames_queue, color_ts)
  
        if depth_frame is not None and ir1_frame is not None and ir2_frame is not None:
            # depth_aligned = align_depth_to_color(depth_frame, color_frame)
            # print(f"Matched color_ts={color_ts:.3f}, depth_ts={depth_ts:.3f}, ir1_ts={ir1_ts:.3f}, ir2_ts={ir2_ts:.3f}")
            ready_queue.put((color_ts, color_frame, depth_ts, depth_frame, ir1_ts, ir1_frame, ir2_ts, ir2_frame))
        else:
            fail_count += 1
            # print(f"Failed to match, fail_count={fail_count}")
            
            if fail_count > fail_tolerance:
                print("Too many failures, stopping...")
                break

        # 标记 color_queue 取走完毕
        color_frames_queue.task_done()

        

def main():
    ############################################################
    # 获取相机参数
    ############################################################
    global R, t, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, color_intr, depth_intr, d2c_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr
    color_intr, depth_intr, d2c_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr = get_camera_params_from_device()
    print("相机参数已获取")
    
    fx_d, fy_d = depth_intr.fx, depth_intr.fy
    cx_d, cy_d = depth_intr.ppx, depth_intr.ppy
    fx_c, fy_c = color_intr.fx, color_intr.fy
    cx_c, cy_c = color_intr.ppx, color_intr.ppy
    
    # 提取旋转矩阵和平移向量
    R = np.array(d2c_extr.rotation).reshape(3, 3)
    t = np.array(d2c_extr.translation)
    
    
    ############################################################
    # 检查输出目录
    ############################################################
    if RECORD:
        if CLEAN_LAST:
            remove_dir(OUTPUT_DIR)
        rgb_dir = os.path.join(OUTPUT_DIR, "rgb")
        depth_dir = os.path.join(OUTPUT_DIR, "depth")
        left_dir = os.path.join(OUTPUT_DIR, "left")
        right_dir = os.path.join(OUTPUT_DIR, "right")
        ensure_dir(rgb_dir)
        ensure_dir(depth_dir)
        ensure_dir(left_dir)
        ensure_dir(right_dir)
            
        for fname in ["rgb.txt", "depth.txt", "associations.txt", "times.txt"]:
            open(os.path.join(OUTPUT_DIR, fname), 'w').close()
            
        write_camera_profile()
    
    
    # 找设备并获取传感器
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No device found")
        exit()

    device = devices[0]

    # 传感器
    # depth_sensor 通常负责 depth、infrared
    # color_sensor 负责 rgb
    # imu_sensor   负责加速度计、陀螺仪
    depth_sensor = None
    color_sensor = None
    imu_sensor = None
    for s in device.sensors:
        name = s.get_info(rs.camera_info.name)
        if "RGB Camera" in name:
            color_sensor = s
        elif "Stereo Module" in name:
            depth_sensor = s
        elif "Motion Module" in name:
            imu_sensor = s

    if depth_sensor is None or color_sensor is None:
        print("Sensor not found!")
        exit()

    # 获取可用的 stream_profile
    depth_profiles = []
    color_profiles = []
    for p in depth_sensor.get_stream_profiles():
        sp      = p.as_video_stream_profile()
        stype   = p.stream_type()
        sformat = p.format()
        w, h    = sp.width(), sp.height()
        fps     = sp.fps()
        idx     = sp.stream_index()  # IR1 是 index=1，IR2 是 index=2
        
        # Depth: 848x480, z16, 90fps
        if stype == rs.stream.depth and sformat == rs.format.z16 and w == WIDTH and h == HEIGHT and fps == FPS_DEPTH:
            depth_profiles.append(p)
        
        # IR1+IR2: 848x480, y8, 90fps
        if stype == rs.stream.infrared and sformat == rs.format.y8 and w == WIDTH and h == HEIGHT and fps == FPS_DEPTH:
            if idx in [1, 2]:
                depth_profiles.append(p)

    for p in color_sensor.get_stream_profiles():
        sp      = p.as_video_stream_profile()
        stype   = p.stream_type()
        sformat = p.format()
        w, h    = sp.width(), sp.height()
        fps     = sp.fps()
        idx     = sp.stream_index()  # IR1 是 index=1，IR2 是 index=2
                
        # Color: 848x480, bgr8, 30fps
        if stype == rs.stream.color and sformat == rs.format.bgr8 and w == WIDTH and h == HEIGHT and fps == FPS_RGB:
            color_profiles.append(p)

    # 打开Sensor
    depth_sensor.open(depth_profiles)
    color_sensor.open(color_profiles)
    

    # Use preset
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    print('Visual Preset Options:')
    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print('%02d: %s'%(i,visulpreset))
        
    depth_sensor.set_option(rs.option.visual_preset, PRESET)
    print(f"深度预设已设置为: {depth_sensor.get_option_value_description(rs.option.visual_preset, int(depth_sensor.get_option(rs.option.visual_preset)))}")
    
    # 设置激光功率
    laser_power_range = depth_sensor.get_option_range(rs.option.laser_power)
    min_power = laser_power_range.min
    max_power = laser_power_range.max
    print(f"Laser power range: {min_power} - {max_power}")
    current_power = depth_sensor.get_option(rs.option.laser_power)
    print(f"Current laser power: {current_power}/{max_power} ({(current_power/max_power)*100:.1f}%)")
    target_power = max_power * LASER_POWER / 100
    depth_sensor.set_option(rs.option.laser_power, target_power)
    actual_power = depth_sensor.get_option(rs.option.laser_power)
    print(f"Laser power set to: {actual_power}/{max_power} ({(actual_power/max_power)*100:.1f}%)")
    
    # 启动Sensor
    depth_sensor.start(depth_callback)
    color_sensor.start(color_callback)
    
    print("Sensors started. Press Ctrl+C to stop...")
    
    ready = False
    
    while True:
        st = time.time()
        if color_frames_queue.qsize() > 0 and depth_frames_queue.qsize() > 0 and ir1_frames_queue.qsize() > 0 and ir2_frames_queue.qsize() > 0:
            ready = True
            print("All queues are ready.")
            break
        time_spent = time.time() - st
        if time_spent > 5:
            print("Failed to start sensors.")
            break

    t_post = Thread(target=depth_post_thread, daemon=True)
    t_match = Thread(target=match_thread, daemon=True)
    t_convert = Thread(target=img_convert_thread, daemon=True)
    if RECORD:
        t_write_rgbd = Thread(target=rgbd_write_thread, daemon=True)
        t_write_stereo = Thread(target=stereo_write_thread, daemon=True)
    
    t_post.start()      # 深度后处理
    t_match.start()     # 匹配线程
    t_convert.start()   # 图像转换线程
    if RECORD:          # 文件写入线程
        print("Recording data...")
        t_write_rgbd.start()
        t_write_stereo.start()
    if VISUALIZE:       # 图像可视化
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    
    try:
        while ready:
            if VISUALIZE:
                if not visualize_step():
                    print("Window closed.")
                    break
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping sensors...")
        
    # 停止可视化线程
    if VISUALIZE:
        # 先清空队列
        while not visual_queue.empty():
            try:
                visual_queue.get_nowait()
                visual_queue.task_done()
            except:
                pass
        # 发送停止信号并处理
        visual_queue.put((0.0, None, 0.0, None, 0.0, None, 0.0, None))
        # 手动调用一次visualize_step来处理停止信号
        visualize_step()
        cv2.destroyAllWindows()
        
    # 停止转换线程
    ready_queue.put((0.0, None, 0.0, None, 0.0, None, 0.0, None))
    ready_queue.join()
    # 停止匹配线程
    color_frames_queue.put((0.0, None))
    color_frames_queue.join()
    # 停止后处理线程
    depth_frames_queue.put((0.0, None))
    depth_frames_queue.join()
    
    # 停止储存线程
    if RECORD:
        output_rgbd_queue.put((0.0, None, 0.0, None))
        output_rgbd_queue.join()
        output_stereo_queue.put((0.0, None, 0.0, None))
        output_stereo_queue.join()
        print("Data saved to", OUTPUT_DIR)

    # 停止并关闭Sensor
    color_sensor.stop()
    depth_sensor.stop()
    color_sensor.close()
    depth_sensor.close()
    
    if ALIGN_DEPTH and RECORD:
        print("Aligning depth images...")
        process_depth_directory()
    
if __name__ == "__main__":
    main()
