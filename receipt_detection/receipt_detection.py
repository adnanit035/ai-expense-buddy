from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8m.pt')
 
# Training.
results = model.train(
   data='recep.yaml',
   imgsz=1080,
   epochs=50,
   batch=8,
   name='yolov8m_v8_50e'
)


	

