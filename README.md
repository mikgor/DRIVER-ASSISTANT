# DRIVER ASSISTANT
DRIVER ASSISTANT (DA) is an application that has implemented following modes:

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
Classifies each pixel in the image. Returns masked image and bounding boxes.
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
After running main.py, mode specified in config executes. If mode not provided in config, 
console menu appears and allows to choose so. 

Specify mode using number in startup config `mode_selected_option`. 

Specify mode function using number(s) in startup config: `function_selected_options`.

### Inference [1-3]
Allows to detect and classify road signs and detect (using semantic segmentation) some of 
road objects.

Inference modes: [`mode_selected_option`]
* Inference (images) [1] 

  Loads images from inference config `images_path`, applies
chosen functions to each of images and displays them.

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
for updated label id (dismiss by pressing Enter).
* Right click inside box area to delete bounding box.
* Press `z` key to delete most recently added bounding box.
* Press any key (except `z`) to apply changes to displayed view.

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

## Data sources
* Test data (data/test) - Google Street View
    * Test images (data/test/images) - Google Street View & Google Graphics
    * Test signs (data/test/signs) - Selected from data/classification/train
    * Test videos (data/test/videos) - YouTube
* Detection
    * Train (data/detection/Train) with annotations (data/detection/train.csv) [39 270 images] and Test data (data/detection/Test) with annotations (data/detection/test.csv) [12 630 images] - https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
    * Train_frames (data/detection/Train_frames) [405 images] - YouTube & Google Graphics
* Classification
    * Train (data/classification/train) [16 702 images] and test data (data/classification/test) [4 298 images]- https://www.kaggle.com/kasia12345/polish-traffic-signs-dataset
* Semantic segmentation
    * Pre-trained model (segmentation/model.net) - Cityscapes Dataset