from ultralytics import YOLO

# 加载预训练模型
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # 使用 YOLOv8 Nano 模型
print("Model loaded successfully.")

# 开始训练
print("Starting training...")
model.train(data='./data.yaml', epochs=10, imgsz=640)

print("Training finished.")