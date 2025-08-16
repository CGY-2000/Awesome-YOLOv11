# Gradio/Streamlit Web App
import gradio as gr
from PIL import Image
import torch
# 导入 ultralytics YOLO
from ultralytics import YOLO

# --- 1. 加载模型 (在全局作用域加载，避免每次调用都重新加载) ---
# 使用 ultralytics 加载模型非常简单
# 它会自动检测可用的设备 (CUDA 或 CPU)
MODEL_PATH = '/home/ubuntu/code/CGY/ultralytics-yolo11/weights/yolo11n.pt' # 你的模型路径
# 尽管你提到了YOLOv11，但ultralytics官方库主要是YOLOv5/v8/v9。
# 只要你的模型是用ultralytics框架训练的，加载方式就一样。
# 我们这里用官方的 'yolov8n.pt' 作为示例，请替换成你自己的 'weights/yolov11n.pt'
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    # 如果用户提供的路径不存在，给出一个更友好的提示
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please make sure the model path is correct.")
    # 你可以在这里下载一个默认模型作为备用
    print("Using 'yolov8n.pt' as a fallback.")
    model = YOLO('yolov8n.pt')


# 获取类别名称 (可选，因为 .plot() 会自动处理)
# 这一行仍然有效，如果需要可以保留
CLASS_NAMES = model.names

# --- 2. 定义核心推理函数 ---
def predict_image(image):
    """
    输入一张PIL Image，返回一张画好框的PIL Image
    ultralytics将所有复杂的步骤都封装好了。
    """
    # 直接使用模型进行预测，输入可以是PIL图像
    # 可以直接在 predict 函数中设置 conf 和 iou 阈值
    results = model.predict(source=image, conf=0.4, iou=0.5)

    # results 是一个列表，每个元素对应一个输入图像的 Results 对象
    # 我们只有一个输入，所以取第一个结果
    result = results[0]

    # ultralytics 的 Results 对象有一个内置的 plot 方法
    # 它可以直接返回画好边界框的图像 (NumPy array 格式)
    annotated_image_np = result.plot()

    # .plot() 返回的图像是 BGR 格式的 NumPy 数组，
    # 我们需要将它转换为 RGB 格式以在 Gradio 中正确显示。
    # 然后再从 NumPy 数组转换为 PIL Image。
    annotated_image_pil = Image.fromarray(annotated_image_np[:, :, ::-1])

    return annotated_image_pil

# --- 3. 创建 Gradio Interface ---
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="Ultralytics YOLO Detector",
    description="Upload an image and see the YOLO model detect objects. This interface uses the 'ultralytics' library for streamlined inference and plotting.",
    examples=[
        # 请确保这些示例图片路径是正确的
        # 你可能需要从 ultralytics 仓库或网上找一些示例图片
        ["assets/bus.jpg"],
        ["assets/zidane.jpg"],
    ]
)

# --- 4. 启动App ---
if __name__ == "__main__":
    iface.launch()
