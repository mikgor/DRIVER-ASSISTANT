# DRIVER ASSISTANT
DRIVER ASSISTANT (DA) is an application that detects and classifies Polish road signs.
## Development

### Virtual environment setup
1. Change working directory to `DRIVER-ASSISTANT`.
2. Create and activate a virtual environment to isolate our package dependencies locally using:
   ```
   python -m venv venv
   source env/bin/activate
   ```
### Installing requirements.txt
1. Change working directory to `DRIVER-ASSISTANT`.
2. Run following command:
   ```
   pip3 install -r requirements.txt
   ```

### Installing Mask_RCNN
1. Download mrcnn directory from https://github.com/akTwelve/Mask_RCNN and paste to `DRIVER-ASSISTANT`
2. Download pre-trained weights from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 and paste to `DRIVER-ASSISTANT`
   
### Data sources
* Detection
    * Train (data/detection/train) with annotations (data/detection/train.csv) [12 630 images] and Test data (data/detection/test) with annotations (data/detection/test.csv) [39 270 images]- https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
    * Test data (data/detection/images) - Google Street View
* Classification: 
    * Train (data/classification/train) [16 702 images] and test data (data/classification/test) [4 298 images]- https://www.kaggle.com/kasia12345/polish-traffic-signs-dataset