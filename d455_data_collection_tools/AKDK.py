"""
Author: Shang Xu
Feel free to use this code for your own purposes, but please give credit to the original author.
"""

import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, ImageFormat, FPS

ESC_KEY = 27

def main():
    # 设置相机配置（包括同步 RGB 和 Depth）
    k4a = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_1536P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            color_format=ImageFormat.COLOR_BGRA32,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    print("Camera started. Press 'q' or ESC to quit.")
    try:
        while True:
            capture = k4a.get_capture()
            color_image = capture.color
            depth_image = capture.transformed_depth  # 自动对齐到 RGB

            if color_image is None or depth_image is None:
                continue

            # 缩放深度图用于可视化
            depth_scaled = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_scaled = depth_scaled.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

            # 确保大小一致（理论上transformed_depth与color分辨率一致）
            if depth_colored.shape[:2] != color_image.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))

            # 混合显示 RGB + 深度图
            blended = cv2.addWeighted(color_image[..., :3], 0.5, depth_colored, 0.5, 0)
            blended = cv2.resize(blended, (800, 600))

            cv2.imshow("Kinect RGB + Depth Viewer", blended)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
