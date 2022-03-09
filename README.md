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
### Configuration
1. Change working directory to `DRIVER-ASSISTANT/configurations`.
2. Create a copy of config.yaml named _config.yaml and adjust parameters if needed.

## Data sources
* Test data (data/test) - Google Street View
    * Test images (data/test/images) - Google Street View & Google Graphics
    * Test signs (data/test/signs) - Selected from data/classification/train
    * Test videos (data/test/videos) - YouTube
* Detection
    * Train (data/detection/Train) with annotations (data/detection/train.csv) [39 270 images] and Test data (data/detection/Test) with annotations (data/detection/test.csv) [12 630 images] - https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
    * Train_frames (data/detection/Train_frames) [350 images] - YouTube & Google Graphics
* Classification
    * Train (data/classification/train) [16 702 images] and test data (data/classification/test) [4 298 images]- https://www.kaggle.com/kasia12345/polish-traffic-signs-dataset
* Semantic segmentation
    * Pre-trained model (segmentation/model.net) - Cityscapes Dataset