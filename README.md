📘 This README is also available in [English](README_EN.md).

# RS D455 数据采集工具

这个仓库包含了用于英特尔RealSense D455深度相机的数据采集和处理工具。这些工具可以帮助您同步采集RGB图像、深度图像和双红外立体图像，并提供深度图像的后处理和对齐功能。
程序使用完全异步方法处理数据流以最大化获取数据规格，因此该程序对于上位机实时处理性能和相机链接带宽有一定要求。
请注意，该程序暂时为实验性阶段，无法确保采集过程的稳定性，如果遇到崩溃可能是由底层c++接口触发，可以多次尝试运行。

## 主要功能

- **多模式数据采集**：同步采集RGB、深度和双红外图像
- **深度图像后处理**：包括深度滤波和降噪
- **深度图像对齐**：将深度图像对齐到RGB图像坐标系
- **实时可视化**：数据流的实时可视化显示
- **数据存储**：将采集的数据保存为格式化的数据集
- **相机参数管理**：保存相机内参和外参信息

## 安装

### 依赖项

```
numba
numpy
opencv_python
pyrealsense2
tqdm
```

### 安装步骤

1. 克隆此仓库：
   ```
   git clone https://github.com/yourusername/rs_d455_data_tools.git
   cd rs_d455_data_tools
   ```

2. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 数据采集工具

```
python d455_data_collection_tools/d455_collector.py
```

采集工具会创建一个名为`rs_d455_data`的目录（可通过修改脚本中的`OUTPUT_DIR`变量更改），其中包含以下子目录：
- `rgb/`：存储RGB图像
- `depth/`：存储深度图像
- `left/`：存储左红外图像
- `right/`：存储右红外图像
- `depth_aligned/`：存储对齐到RGB的深度图像（如果启用`ALIGN_DEPTH`）

此外，还会生成以下文件：
- `rgb.txt`：RGB图像文件列表和时间戳
- `depth.txt`：深度图像文件列表和时间戳
- `associations.txt`：RGB和深度图像的关联信息
- `times.txt`：立体图像的时间戳
- `RS_D455.yaml`：相机参数配置文件
- `calib.txt`：相机标定信息（KITTI格式）

### 深度图像对齐工具

```
python d455_data_collection_tools/depth_align.py --depth-dir INPUT_DEPTH_DIR --output-dir OUTPUT_ALIGNED_DIR [--use-device]
```

参数说明：
- `--depth-dir`：输入深度图像目录
- `--output-dir`：输出对齐深度图像目录
- `--depth-scale`：深度比例因子，默认为0.001（毫米转米）
- `--use-device`：使用连接的RealSense设备获取相机参数
- `--rgb-width`：RGB图像宽度，默认为848
- `--rgb-height`：RGB图像高度，默认为480

## 配置选项

在`d455_collector.py`中，您可以调整以下参数：

```python
FPS_DEPTH       = 90                # 深度相机帧率
FPS_RGB         = 30                # 彩色相机帧率
WIDTH           = 848               # 图像宽度
HEIGHT          = 480               # 图像高度
LASER_POWER     = 100               # 激光功率 0% - 100%
POST_DEPTH      = True              # 是否做深度后处理
ALIGN_DEPTH     = True              # 是否对齐深度到彩色
OUTPUT_DIR      = "rs_d455_data"    # 输出目录
CLEAN_LAST      = True              # 是否清空上次的输出目录
RECORD          = True              # 是否保存数据
VISUALIZE       = True              # 是否可视化
PRESET          = 3                 # 高精度模式下的预设
```

## 注意事项

1. 确保RealSense D455相机已正确连接到您的计算机
2. 相机需要足够的USB带宽，建议使用USB 3.0或更高版本的接口
3. 采集过程中避免相机剧烈移动，以获得更好的数据质量
4. 深度后处理可能会增加CPU负载，如果您的计算机性能有限，可以考虑禁用此功能

## 作者

**Shang Xu**

- 项目创建者与主要维护者
- 如果您在项目中使用了此代码，请务必引用原作者
- 欢迎通过Issue提交问题反馈或功能建议
- 如需进一步的技术支持或合作，请通过GitHub联系作者

## 许可证

本项目采用GNU通用公共许可证v3.0 - 详见LICENSE文件。
