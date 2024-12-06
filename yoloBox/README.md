# YOLO Box

使用了Ultralytics YOLO API （YOLOv8）

## 环境配置

1. 创建 Python 环境：

   ```bash
   python3 -m venv env
   source env/bin/activate  # Mac/Linux
   
   ```

2. 在`yoloBox/datasets/data`下载数据集

   ```
   mkdir -p datasets/data
   curl -L "https://universe.roboflow.com/ds/xaw2JvX2ej?key=s9onrnHcEf" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
   ```

3. 跑下面的代码之前，terminal要到当前folder

   `yoloBox`

4. 安装依赖
   `pip install -r requirements.txt`

5. 训练模型
   `python scripts/train.py`

6. 跑 test（我已经训练了一个 model 并且 push 了上来，所以也可以跳过 3 跑 4 这一步）

   `python scripts/inference.py`



notes:

1. 如果你要自己再训一下的话，你要修改train.py 文件。如果你改动了dataset位置，要去data.yaml里面更新path（感觉精确度中等，可以修改参数再train一下model）
2. 我把test dataset里面的图片中，水平横着的sbb门的图片都删掉了
