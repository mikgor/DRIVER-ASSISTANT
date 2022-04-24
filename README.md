# DRIVER ASSISTANT
[![Detection and classification example](https://s439393.students.wmi.amu.edu.pl/da/detection_and_classification_and_segmentation_example.png)](https://s439393.students.wmi.amu.edu.pl/da/detection_and_classification_and_segmentation_example.png)

DRIVER ASSISTANT (DA) is an application that has implemented following functions:

### Detection
Finds bounding boxes of road signs, including sign object/ classified sign id 
depending on single/ multiclass configuration and confidence score. 
Also allows to train own model to detect objects.

### Classification
Returns classified sign id and name. Also allows to train own model to detect objects.
Detectable Polish road signs:
* Niebezpieczny zakręt w prawo
* Nierówna droga
* Próg zwalniający
* Zwężenie jezdni — dwustronne
* Roboty na drodze
* Śliska jezdnia
* Przejście dla pieszych
* Dzieci
* Zwierzęta dzikie
* Niebezpieczny zakręt w lewo
* Odcinek jezdni o ruchu dwukierunkowym
* Tramwaj
* Rowerzyści
* Sygnały świetlne
* Niebezpieczne zakręty — pierwszy w prawo
* Inne niebezpieczeństwo
* Oszronienie jezdni
* Niebezpieczne zakręty — pierwszy w lewo
* Skrzyżowanie z drogą podporządkowaną występującą po obu stronach
* Skrzyżowanie z drogą podporządkowaną występującą po prawej stronie
* Skrzyżowanie z drogą podporządkowaną występującą po lewej stronie
* Wlot drogi jednokierunkowej z prawej strony
* Wlot drogi jednokierunkowej z lewej strony
* Ustąp pierwszeństwa
* Skrzyżowanie o ruchu okrężnym
* Zakaz ruchu w obu kierunkach
* Zakaz wjazdu pojazdów o rzeczywistej masie całkowitej ponad …t
* Zakaz wjazdu
* Stop
* Zakaz skręcania w lewo
* Zakaz skręcania w prawo
* Zakaz wyprzedzania
* Zakaz wyprzedzania przez samochody ciężarowe
* Koniec zakazu wyprzedzania
* Ograniczenie prędkości
* Koniec ograniczenia prędkości
* Zakaz zatrzymywania się
* Zakaz ruchu pieszych
* Koniec zakazów
* Strefa ograniczonej prędkości
* Koniec strefy ograniczonej prędkości
* Zakaz wjazdu samochodów ciężarowych
* Zakaz wjazdu pojazdów innych niż samochodowe
* Zakaz wjazdu pojazdów zaprzęgowych
* Zakaz wjazdu rowerów
* Nakaz jazdy z lewej strony znaku
* Ruch okrężny
* Droga dla rowerów
* Droga dla pieszych i rowerów
* Koniec drogi dla rowerów
* Koniec drogi dla pieszych i rowerów
* Droga dla pieszych
* Nakaz jazdy w prawo za znakiem
* Nakaz jazdy w lewo za znakiem
* Nakaz jazdy prosto
* Nakaz jazdy prosto lub w prawo
* Nakaz jazdy prosto lub w lewo
* Nakaz jazdy z prawej strony znaku
* Droga z pierwszeństwem
* Koniec pasa ruchu
* Przystanek autobusowy
* Parking
* Parking zadaszony
* Koniec drogi z pierwszeństwem
* Szpital
* Stacja paliwowa
* Stacja paliwowa tylko z gazem do napędu pojazdów
* Telefon
* Stacja obsługi technicznej
* Myjnia
* Toaleta publiczna
* Bufet lub kawiarnia
* Restauracja
* Hotel (motel)
* Droga jednokierunkowa
* Strefa zamieszkania
* Koniec strefy zamieszkania
* Obszar zabudowany
* Koniec obszaru zabudowanego
* Droga bez przejazdu
* Wjazd na drogę bez przejazdu
* Automatyczna kontrola prędkości
* Strefa ruchu
* Koniec strefy ruchu
* Przejście dla pieszych
* Przejście dla pieszych i przejazd dla rowerzystów
* Droga ekspresowa
* Koniec drogi ekspresowej
* Autostrada
* Zbiorcza tablica informacyjna
* Słupek wskaźnikowy z trzema kreskami umieszczany po prawej stronie jezdni
* Krzyż św. andrzeja przed przejazdem kolejowo-drogowym jednotorowym

### Semantic segmentation
Classifies each pixel in the image. Returns masked image and bounding boxes for
labels selected in segmentation config `labels_to_detect`.
Detectable object names:
* Unlabeled
* Road
* Sidewalk
* Building
* Wall
* Fence
* Pole
* Traffic light
* Traffic sign
* Vegetation
* Terrain
* Sky
* Person
* Rider
* Car
* Truck
* Bus
* Train
* Motorcycle
* Bicycle

## Modes
After running main.py, mode specified in config executes.
Specify mode using number in startup config `mode_selected_option`. 

Specify mode function using number(s) in startup config: `function_selected_options`.

If mode not provided in config, console menu appears and allows to choose so:

[![Menu](https://s439393.students.wmi.amu.edu.pl/da/menu.png)](https://s439393.students.wmi.amu.edu.pl/da/menu.png)

### Inference [1-3]
Allows to detect and classify road signs and detect (using semantic segmentation) some of 
road objects.

Inference modes: [`mode_selected_option`]
* Inference (images) [1] 

  Loads images from inference config `images_path`, applies
chosen functions to each of images and displays them.
All detected bounding boxes (from detection and segmentation) are saved to separate 
.csv files.

* Inference (videos) [2] 

  Loads videos from inference videos config `path`, applies
chosen functions to each of frames and displays them (if `show_labeled_frame`).
All detected bounding boxes (from detection and segmentation) are saved to separate 
.csv files. Video with labels is saved (if `save_labeled`) to `output_path`.

* Inference (live) [3]

  Applies chosen functions to frames from detected camera and displays 
them (if `show_labeled_frame`). All detected bounding boxes (from detection 
and segmentation) are saved to separate .csv files. Video with labels is 
saved (if `save_labeled`) to `output_path`.


Inference functions: [`function_selected_options`]. Can select multiple.
* Detection [1]
* Classification [2]
* Semantic segmentation [3]

### Train [4]
Allows to train detection and classification models on custom datasets.

Train functions: [`function_selected_options`]
* Detection [1] 

  Performs detection model training with given config. Trained model and charts
are saved.

* Classification [2]

  Performs classification model training with given config. Trained model and charts
are saved.

### Apply labels to videos and play [5]
Loads videos from inference videos config `path`, adds bounding boxes from saved separate 
.csv files on frames and play video frame by frame.

### Label images [6]
Uses detection and classification to semi-automated labeling (bounding box and label id). 
Loads images from image labeling config `images_folder_path`, and allows actions:
* Left click on box-free area to select starting x and starting y for bounding box. 
Do it again to choose ending x and ending y. Once done `default_label_id` is used as
box id and console input waits for updated label id (dismiss by pressing Enter).
* Left click inside box area to updated label id for bounding box. Console input waits 
for updated label id (dismiss by pressing `Enter`).
* Right click inside box area to delete bounding box.
* Press `z` key to delete most recently added bounding box.
* Press any key (except `z`) to apply changes to displayed view.
* Press `Space` to save boxes to .csv and display next image.

Image labeling functions: [`function_selected_options`]. Both are required.
* Detection [1]
* Classification [2]

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

## Trained & ready-to-use models
### Detection
Path: `detection/models/frames_single/model50.pth`

Configuration file: `configurations/azure_frames_detection_single_config.yaml`

Training data length: 3286

Validating data length: 372

### Classification
Path: `classification/models/augmented_model`

Configuration file: `configurations/azure_frames_classification_config.yaml`

Training data length: 27059 

Validating data length: 9019

### Semantic segmentation
Pre-trained model - Cityscapes Dataset

Path: `segmentation/model.net`

## Data organisation & sources
* Test data (data/test)
    * Test images (data/test/images) - Google Street View & Google Graphics
    * Test signs (data/test/signs) - Selected from data/classification/train
    * Test videos (data/test/videos) - YouTube - [1][2][3]
* Detection
    * Train (data/detection/Train) with annotations (data/detection/train.csv) [39 270 images] and Test data (data/detection/Test) with annotations (data/detection/test.csv) [12 630 images] - [4]
    * Test_frames (data/detection/Test_frames) [69 images] - Google Street View
    * train_subset.csv (data/detection/train_subset.csv) [350 images] & test_subset.csv 
    (data/detection/test_subset.csv) [70 images] - random signs from GTSRB looking similarly to Polish 
    (B-1 [id: 15 -> 26], B-2 [id: 17 -> 28], B-20 [id: 14 -> 29], B-25 [id: 9 -> 32], 
    B-29 [id: 29 -> 45], B-33 [id: 0-5, 7-8 -> 35], D-1 [id: 12 -> 59]) using 
    `create_random_subset_from_gtsrb_df('data/detection/X.csv', 'data/detection/X_subset.csv', 'ClassId',
    [15, 17, 14, 9, 29, 0, 1, 2, 3, 4, 5, 7, 8, 12], [26, 28, 29, 32, 45, 35, 35, 35, 35, 35, 35, 35, 35, 59], Y)`
* Classification
    * Train (data/classification/train) [16 702 images] and test data (data/classification/test) [4 298 images]- [5]
    * Train_frames (data/classification/Test_frames) [1 293 images] - random signs from detection/Train_frames created using
    `create_sign_classification_dataset_from_gtsrb_df('data/detection/train_frames.csv', 'data/classification/Train_frames', config['detection'],
    config['classification'])`
    * Test_frames (data/classification/Test_frames) [116 images] - random signs from detection/Test_frames created using
    `create_sign_classification_dataset_from_gtsrb_df('data/detection/test_frames.csv', 'data/classification/Test_frames', config['detection'],
    config['classification'])`

### Sources:
[1] Mateusz Baryła (https://www.youtube.com/channel/UCKDhmtsuwlP2Hs0KUrBkK5Q) - Cracow 4K night driving (https://youtu.be/qEd4CB7OEkM)

[2] GoodPlace33 travel [Paweł Koronacki] (https://www.youtube.com/channel/UColdbTFeTctW0FLHsuh3VkQ) - Driving around Poznań - Part 1, Wilda to Winogrady - 30th March 2021 (https://youtu.be/QRIUaYbZyXg), Driving around Poznań - Part 3 - 31th March 2021 (https://youtu.be/GfsviXXLQ6o)

[3] przychodzkipl - aviation & travel [Michał Przychodzki] (https://www.youtube.com/channel/UCQJzMRYjkjJCu0FnAknwjXA), “Warsaw -
driving through the center of the Polish capital | Warszawa przejazd przez
centrum stolicy“ (https://youtu.be/K8eSOt8u25w)

[4] Mykola, “GTSRB - German Traffic Sign Recognition Benchmark“ (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-germantraffic-sign)

[5] KASIA12345, “polish traffic signs dataset“ (https://www.kaggle.com/datasets/kasia12345/polish-traffic-signs-dataset)
