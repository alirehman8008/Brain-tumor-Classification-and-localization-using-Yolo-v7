
# Project Title
🧠 Brain Tumor Detection Using YOLO v7
Implemented a deep learning model using YOLO v7 to detect three types of brain tumors: meningioma, glioma, and pituitary. Achieved an impressive 96.7% accuracy!

Tools: Python, TensorFlow, OpenCV
Techniques: YOLO v7

🖼️ Image Annotation for Brain Tumor Dataset

Annotated 3,000 brain tumor images using LabelImg and Roboflow for training the detection models.

Types of Tumors: Meningioma, Glioma, Pituitary
Tools: LabelImg, Roboflow

🔄 Data Preprocessing & Augmentation

Processed and augmented the annotated dataset to enhance model performance by increasing data variability. Techniques included resizing, normalization, and random transformations to improve model generalization.

Tools: Python, OpenCV, TensorFlow
Techniques: Resizing, Normalization, Data Augmentation (rotation, flipping, zoom)

🤖 Model Training & Optimization

Trained a deep learning model using YOLO v7 to detect brain tumors. Focused on optimizing the model with hyperparameter tuning and reducing overfitting.

Tools: TensorFlow, PyTorch, YOLO v7
Techniques: Transfer learning, Hyperparameter tuning, Early stopping

📊 Model Evaluation & Validation

Evaluated the trained model on a test dataset, achieving 96.7% accuracy. Used performance metrics such as precision, recall, F1-score, and confusion matrix to validate the model's effectiveness in detecting brain tumors.

Tools: Scikit-learn, TensorFlow
Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

💾 Mount Google Drive in Colab

            from google.colab import drive
            drive.mount("/content/gdrive")

This code mounts your Google Drive into Google Colab, allowing access to files stored in your Drive for use in notebooks.

📂 Change Directory to YOLO Training Dataset

            %cd /content/gdrive/MyDrive/yolo-training-dataset

This command changes the current working directory in Colab to the folder where your YOLO training dataset is stored, making it easier to access the dataset files for training.

🛠️ Clone YOLOv7 Repository from GitHub

            !git clone https://github.com/WongKinYiu/yolov7.git

This command clones the official YOLOv7 repository from GitHub into your current working directory, downloading all the necessary files and scripts for model training and evaluation.

📦 Install Required Dependencies

            !pip install -r requirements.txt

This command installs all the necessary libraries and dependencies listed in the requirements.txt file, ensuring the environment is properly set up for YOLOv7 or any other project.


📥 Download Pre-trained YOLOv7 Model Weights

            !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

This command downloads the pre-trained YOLOv7 model weights (yolov7.pt) from the official repository, allowing you to either fine-tune the model or use it directly for inference.

🚀 Train YOLOv7 Model on Custom Dataset

            !python train.py --workers 8 --device 0 --batch-size 8 --data custom.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml --epochs 60

🔍 Run Inference on Images

            !python detect.py --weights yolov7_custom.pt --conf 0.2 --img-size 640 --source /content/gdrive/MyDrive/fyp/yolo-training-dataset/yolov7/Testing/pituitary_tumor

This command executes the detection script using the trained YOLOv7 model (yolov7_custom.pt), specifying the confidence threshold, image size, and source directory containing images for inference. The output will include detected objects with bounding boxes and confidence scores.
