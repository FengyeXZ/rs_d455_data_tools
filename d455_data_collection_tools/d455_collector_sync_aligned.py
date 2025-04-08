"""
Author: Shang Xu
Feel free to use this code for your own purposes, but please give credit to the original author.
"""

import pyrealsense2 as rs
from queue import Queue
from threading import Thread, Event
import time
import os
import shutil
import cv2
import numpy as np

############################################################
# 全局设置
############################################################
FPS             = 60                    # 相机帧率
WIDTH           = 848                   # 图像宽度
HEIGHT          = 480                   # 图像高度
LASER_POWER     = 100                   # 激光功率 0% - 100%
POST_DEPTH      = True                  # 是否做深度后处理
ALIGN_DEPTH     = True                  # 是否对齐深度到彩色
OUTPUT_DIR      = "rs_d455_sync"        # 输出目录
CLEAN_LAST      = True                  # 是否清空上次的输出目录
RECORD          = True                  # 是否保存数据
VISUALIZE       = True                  # 是否可视化
PRESET          = 1                     # 默认模式下的预设
MIN_DISTANCE    = 0.4                   # 最小深度距离(米)
MAX_DISTANCE    = 6.0                   # 最大深度距离(米)

# 创建管道
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS)

frames_queue        = Queue()
aligns_queue        = Queue()
ready_queue         = Queue()
output_rgbd_queue   = Queue()
output_stereo_queue = Queue()
visual_queue        = Queue()
R, R_inv, t, t_inv, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c = None, None, None, None, None, None, None, None, None, None, None, None
color_intr, depth_intr, d2c_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr = None, None, None, None, None, None, None
terminate_event     = Event()
STOP_SIGNAL         = object()


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
    color_to_depth = color_stream.get_extrinsics_to(depth_stream)
    ir2_to_ir1 = ir2_stream.get_extrinsics_to(ir1_stream)
    ir1_to_ir2 = ir1_stream.get_extrinsics_to(ir2_stream)
    
    # 停止pipeline
    pipeline.stop()
    
    return color_intr, depth_intr, depth_to_color, color_to_depth, ir1_intr, ir2_intr, ir2_to_ir1, ir1_to_ir2

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
        "Camera.fps": FPS,
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
# 数据获取线程
############################################################
def frameset_thread():
    """
    不断从传感器获取帧，放入队列。
    """
    
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]

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
    
    frame_count = 0
    st = time.time()
    while True:
        frame_count += 1
        if frame_count > 300:
            et = time.time()
            print(f"帧率: {frame_count / (et - st):.2f}")
            frame_count = 0
            st = et
        frames = pipeline.wait_for_frames()
        frames_queue.put(frames)
        if terminate_event.is_set():
            break
    

############################################################
# 深度后处理线程
############################################################
def depth_post_thread():
    """
    不断从 frames_queue 中取帧，做后处理。
    """
    align = rs.align(rs.stream.color)
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
    threshold.set_option(rs.option.min_distance, MIN_DISTANCE)
    threshold.set_option(rs.option.max_distance, MAX_DISTANCE)
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 0)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.holes_fill, 3)
    
    while True:
        frames = frames_queue.get()
        if frames is STOP_SIGNAL:
            frames_queue.task_done()
            break
        
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)
        
        if not depth_frame or not color_frame:
            continue
        
        color_timestamp = color_frame.get_timestamp()
        depth_timestamp = depth_frame.get_timestamp()
        ir1_timestamp = ir1_frame.get_timestamp()
        ir2_timestamp = ir2_frame.get_timestamp()
        
        filtered_depth = depth_frame
   
        if POST_DEPTH:
            filtered_depth = decimation.process(filtered_depth)
            filtered_depth = threshold.process(filtered_depth)
            filtered_depth = depth2disparity.process(filtered_depth)
            filtered_depth = spatial.process(filtered_depth)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = disparity2depth.process(filtered_depth)
        
        ready_queue.put((color_timestamp, color_frame, depth_timestamp, filtered_depth, ir1_timestamp, ir1_frame, ir2_timestamp, ir2_frame))
        frames_queue.task_done()
        

############################################################
# 图像转换线程
############################################################
def img_convert_thread():
    """
    不断从 ready_queue 中取帧，转换成图片。
    """
    while True:
        data = ready_queue.get()
        if data is STOP_SIGNAL:
            # 收到停止信号
            ready_queue.task_done()
            break
        
        # print("Output queues:", output_rgbd_queue.qsize(), output_stereo_queue.qsize())
        # print("Visual queues:", visual_queue.qsize())
        color_ts, color_frame, depth_ts, depth_frame, ir1_ts, ir1_frame, ir2_ts, ir2_frame = data
        
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
        data = output_rgbd_queue.get()
        if data is STOP_SIGNAL:
            # 收到停止信号
            output_rgbd_queue.task_done()
            break
        
        color_ts, color_img, depth_ts, depth_img = data

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
        data = output_stereo_queue.get()
        if data is STOP_SIGNAL:
            # 收到停止信号
            output_stereo_queue.task_done()
            break

        ir1_ts, ir1_img, ir2_ts, ir2_img = data
        
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
    
    data = visual_queue.get()
    if data is STOP_SIGNAL:
        # 收到停止信号
        visual_queue.task_done()
        return False
    
    color_ts, color_img, depth_ts, depth_img, ir1_ts, ir1_img, ir2_ts, ir2_img = data

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


def main():
    ############################################################
    # 获取相机参数
    ############################################################
    global R, R_inv, t, t_inv, fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, color_intr, depth_intr, d2c_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr
    color_intr, depth_intr, d2c_extr, c2d_extr, ir1_intr, ir2_intr, r2l_extr, l2r_extr = get_camera_params_from_device()
    print("相机参数已获取")
    
    fx_d, fy_d = depth_intr.fx, depth_intr.fy
    cx_d, cy_d = depth_intr.ppx, depth_intr.ppy
    fx_c, fy_c = color_intr.fx, color_intr.fy
    cx_c, cy_c = color_intr.ppx, color_intr.ppy
    
    # 提取旋转矩阵和平移向量
    R = np.array(d2c_extr.rotation).reshape(3, 3)
    t = np.array(d2c_extr.translation)
    
    R_inv = np.array(c2d_extr.rotation).reshape(3, 3)
    t_inv = np.array(c2d_extr.translation)
    
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
    
    profile = pipeline.start(config)

    t_frameset = Thread(target=frameset_thread, daemon=True)
    t_post = Thread(target=depth_post_thread, daemon=True)
    t_convert = Thread(target=img_convert_thread, daemon=True)
    if RECORD:
        t_write_rgbd = Thread(target=rgbd_write_thread, daemon=True)
        t_write_stereo = Thread(target=stereo_write_thread, daemon=True)
    
    t_frameset.start()  # 数据获取线程
    t_post.start()      # 深度后处理
    t_convert.start()   # 图像转换线程
    if RECORD:          # 文件写入线程
        print("Recording data...")
        t_write_rgbd.start()
        t_write_stereo.start()
    if VISUALIZE:       # 图像可视化
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
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
        visual_queue.put(STOP_SIGNAL)
        # 手动调用一次visualize_step来处理停止信号
        visualize_step()
        cv2.destroyAllWindows()
        
    # 停止获取线程
    terminate_event.set()
    # 停止后处理线程
    frames_queue.put(STOP_SIGNAL)
    frames_queue.join()
    # 停止转换线程
    ready_queue.put(STOP_SIGNAL)
    ready_queue.join()
    
    # 停止储存线程
    if RECORD:
        output_rgbd_queue.put(STOP_SIGNAL)
        output_rgbd_queue.join()
        output_stereo_queue.put(STOP_SIGNAL)
        output_stereo_queue.join()
        print("Data saved to", OUTPUT_DIR)

    # 停止并关闭Sensor
    pipeline.stop()
    
if __name__ == "__main__":
    main()
