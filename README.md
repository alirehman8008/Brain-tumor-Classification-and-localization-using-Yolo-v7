# Project Title
Brain Tumor Classification and localization Using Yolo V7
A brief description of what this project does and who it's for

## Steps For Implementation:
Open google colab.
       
       # Mounting the drive to gdrive 
       from google.colab import drive 
       drive.mount("/content/gdrive")

       # changing to data set directory      
       %cd /content/gdrive/MyDrive/yolo-training-dataset

       # Colning yolo v7 repository 
       !git clone https://github.com/WongKinYiu/yolov7.git
       
       # Changing to yolo v7 drive
       cd yolov7
       
       # Installing requirements file for yolov7
       !pip install -r requirements.txt

        # Gettinng yolov7 training file.
        !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

        # Traning the modle and tune the parameter epoch and bacthsize.
        !python train.py --workers 8 --device 0  --batch-size 8 --data custom.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.py' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --epoch 60


        # Model Prediction
        !python detect.py --weights yolov7_custom.pt --conf 0.2 --img-size 640 --source /content/gdrive/MyDrive/fyp/yolo-training-dataset/yolov7/Testing/pituitary_tumor
