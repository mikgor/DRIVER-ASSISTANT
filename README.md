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
   
### Data sources
* Detection
    * Test data (data/detection/images) - Google Street View
* Classification: 
    * Train (data/classification/train) and test data (data/classification/test) - https://www.kaggle.com/kasia12345/polish-traffic-signs-dataset