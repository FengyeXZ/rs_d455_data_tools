import os
import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
import json
import glob
from tqdm import tqdm
import numba


def align_depth_vectorized(
    depth_image, 
    R, t, 
    fx_d, fy_d, cx_d, cy_d, 
    fx_c, fy_c, cx_c, cy_c, 
    depth_scale, 
    out_height, out_width
    ):
    """
    使用NumPy矢量化操作加速深度图对齐
    """
    h_d, w_d = depth_image.shape
    
    # 创建结果图像
    aligned = np.zeros((out_height, out_width), dtype=np.uint16)
    
    # 创建网格坐标
    u_coords, v_coords = np.meshgrid(np.arange(w_d), np.arange(h_d))
    
    # 筛选有效深度点
    mask = depth_image > 0
    
    # 提取有效坐标和深度值
    v_valid = v_coords[mask]
    u_valid = u_coords[mask]
    z_valid = depth_image[mask] * depth_scale
    
    # 计算3D坐标（反投影）
    X_d = (u_valid - cx_d) * z_valid / fx_d
    Y_d = (v_valid - cy_d) * z_valid / fy_d
    Z_d = z_valid
    
    # 旋转和平移到彩色相机坐标系
    X_c = R[0,0]*X_d + R[0,1]*Y_d + R[0,2]*Z_d + t[0]
    Y_c = R[1,0]*X_d + R[1,1]*Y_d + R[1,2]*Z_d + t[1]
    Z_c = R[2,0]*X_d + R[2,1]*Y_d + R[2,2]*Z_d + t[2]
    
    # 筛选有效Z值
    valid_z = Z_c > 0
    X_c, Y_c, Z_c = X_c[valid_z], Y_c[valid_z], Z_c[valid_z]
    depth_values = depth_image[mask][valid_z]  # 原始深度值（不是缩放后的）
    
    # 投影到彩色相机像素坐标
    u_c = np.round(fx_c * (X_c / Z_c) + cx_c).astype(np.int32)
    v_c = np.round(fy_c * (Y_c / Z_c) + cy_c).astype(np.int32)
    
    # 筛选有效范围内的像素
    valid_pixels = (u_c >= 0) & (u_c < out_width) & (v_c >= 0) & (v_c < out_height)
    u_c, v_c = u_c[valid_pixels], v_c[valid_pixels]
    depth_values = depth_values[valid_pixels]
    
    # 使用像素坐标和深度值构建索引数组
    idx = v_c * out_width + u_c
    
    # 创建扁平化的结果数组
    aligned_flat = aligned.ravel()
    
    # 按深度值排序，确保较近的点覆盖较远的点
    sort_idx = np.argsort(depth_values)[::-1]  # 降序排列
    idx = idx[sort_idx]
    depth_values = depth_values[sort_idx]
    
    # 填充结果数组
    aligned_flat[idx] = depth_values
    
    return aligned.reshape((out_height, out_width))

@numba.njit(parallel=True)
def align_depth_numba(
    depth_image,
    R, t,
    fx_d, fy_d, cx_d, cy_d,
    fx_c, fy_c, cx_c, cy_c,
    depth_scale,
    out_height, out_width
):
    """
    使用逐像素投影+Numba加速，将深度图对齐到彩色坐标系。
    返回对齐后的深度图(单位:与原depth_image相同)。
    """
    # 输出图像
    aligned = np.zeros((out_height, out_width), dtype=np.uint16)

    h_d, w_d = depth_image.shape

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

def save_camera_params(file_path, width, height, 
                      fx, fy, cx, cy, 
                      k1=0, k2=0, p1=0, p2=0, k3=0,
                      rotation=None, translation=None):
    """保存相机参数到JSON文件"""
    params = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_coeffs": [k1, k2, p1, p2, k3]
    }
    
    if rotation is not None and translation is not None:
        params["rotation"] = rotation.tolist()
        params["translation"] = translation.tolist()
    
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)

def load_camera_params(file_path):
    """从JSON文件加载相机参数"""
    with open(file_path, 'r') as f:
        params = json.load(f)
    
    rotation = None
    translation = None
    if 'rotation' in params and 'translation' in params:
        rotation = np.array(params['rotation'])
        translation = np.array(params['translation'])
    
    return (params['width'], params['height'], 
            params['fx'], params['fy'], 
            params['cx'], params['cy'], 
            params['distortion_coeffs'],
            rotation, translation)

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
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    # 启动pipeline
    profile = pipeline.start(config)
    
    # 获取深度传感器和彩色传感器的内参
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    
    depth_intr = depth_stream.get_intrinsics()
    color_intr = color_stream.get_intrinsics()
    
    # 获取深度到彩色的外参
    depth_to_color = depth_stream.get_extrinsics_to(color_stream)
    
    # 停止pipeline
    pipeline.stop()
    
    # 提取旋转矩阵和平移向量
    R = np.array(depth_to_color.rotation).reshape(3, 3)
    t = np.array(depth_to_color.translation)
    
    # 保存相机参数
    save_camera_params('depth_camera.json', 
                      depth_intr.width, depth_intr.height,
                      depth_intr.fx, depth_intr.fy,
                      depth_intr.ppx, depth_intr.ppy,
                      *depth_intr.coeffs)
    
    save_camera_params('color_camera.json',
                      color_intr.width, color_intr.height,
                      color_intr.fx, color_intr.fy,
                      color_intr.ppx, color_intr.ppy,
                      *color_intr.coeffs)
    
    save_camera_params('extrinsics.json',
                      color_intr.width, color_intr.height,
                      color_intr.fx, color_intr.fy,
                      color_intr.ppx, color_intr.ppy,
                      *color_intr.coeffs,
                      R, t)
    
    print("相机参数已保存到 depth_camera.json, color_camera.json 和 extrinsics.json")
    
    return (depth_intr, color_intr, R, t)

def process_depth_directory(depth_dir, output_dir, camera_params, depth_scale=0.001):
    """
    只处理深度图像，不需要RGB图像
    
    参数:
    - depth_dir: 深度图像目录
    - output_dir: 输出目录
    - camera_params: 相机参数元组 (fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, R, t, width_c, height_c)
    - depth_scale: 深度比例因子
    """
    fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, R, t, width_c, height_c = camera_params
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有深度图像
    depth_files = glob.glob(os.path.join(depth_dir, '*.png'))
    
    print(f"找到 {len(depth_files)} 个深度图像")
    
    # 处理每个深度图像
    for depth_file in tqdm(depth_files):
        # 读取深度图像
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)  # 16位无符号整数
        
        if depth_img is None:
            print(f"Warning: 无法读取图像 - {depth_file}")
            continue
        
        # 对齐深度图
        aligned_depth = align_depth_numba(
            depth_img,
            R, t,
            fx_d, fy_d, cx_d, cy_d,
            fx_c, fy_c, cx_c, cy_c,
            depth_scale,
            height_c, width_c
        )
        
        # 保存对齐后的深度图
        output_file = os.path.join(output_dir, os.path.basename(depth_file))
        cv2.imwrite(output_file, aligned_depth)
    
    print(f"处理完成! {len(depth_files)} 个深度图像已对齐并保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='高效离线深度图像对齐工具')
    parser.add_argument('--depth-dir', type=str, required=True, 
                        help='输入深度图像目录')
    parser.add_argument('--output-dir', type=str, required=True, 
                        help='输出对齐深度图像目录')
    parser.add_argument('--depth-scale', type=float, default=0.001, 
                        help='深度比例因子，通常为0.001（毫米转米）')
    parser.add_argument('--use-device', action='store_true', 
                        help='使用连接的RealSense设备获取相机参数')
    parser.add_argument('--rgb-width', type=int, default=848, 
                        help='RGB图像宽度')
    parser.add_argument('--rgb-height', type=int, default=480, 
                        help='RGB图像高度')
    
    args = parser.parse_args()
    
    # 获取相机参数
    if args.use_device:
        try:
            depth_intr, color_intr, R, t = get_camera_params_from_device()
            # 从内参和外参中提取参数
            fx_d, fy_d = depth_intr.fx, depth_intr.fy
            cx_d, cy_d = depth_intr.ppx, depth_intr.ppy
            fx_c, fy_c = color_intr.fx, color_intr.fy
            cx_c, cy_c = color_intr.ppx, color_intr.ppy
            width_c, height_c = color_intr.width, color_intr.height
        except Exception as e:
            print(f"无法从设备获取相机参数: {e}")
            print("尝试从JSON文件加载相机参数...")
            args.use_device = False
    
    if not args.use_device:
        # 尝试从JSON文件加载相机参数
        try:
            _, _, fx_d, fy_d, cx_d, cy_d, dist_d, _, _ = load_camera_params('depth_camera.json')
            width_c, height_c, fx_c, fy_c, cx_c, cy_c, _, R, t = load_camera_params('extrinsics.json')
            # 如果指定了RGB尺寸，使用指定的尺寸
            if args.rgb_width != 848 or args.rgb_height != 480:
                width_c, height_c = args.rgb_width, args.rgb_height
                print(f"使用指定的RGB尺寸: {width_c}x{height_c}")
        except Exception as e:
            raise RuntimeError(f"无法加载相机参数: {e}\n请先连接设备并使用 --use-device 选项")
    
    # 封装相机参数
    camera_params = (fx_d, fy_d, cx_d, cy_d, fx_c, fy_c, cx_c, cy_c, R, t, width_c, height_c)
    
    # 处理深度图像
    process_depth_directory(
        args.depth_dir, 
        args.output_dir, 
        camera_params, 
        args.depth_scale
    )

if __name__ == "__main__":
    main()