# Artificial-Intelligence-final-project
## Main approach
### Overview
我們的目的是辨識影像中的人有沒以戴口罩。
我們使用Transfer Learning的方法，部分採用MobilenetV2這個模型，採用的部分是對影像進行特徵擷取，載入這個模型之後我們把最後一層改成要辨識的兩種類別，就是有無戴口罩，然後針對這些進行訓練。
### 架構
- application
   - mask_detector
      - main_approach_train.py
      - main_approach_test.py
      - main_approach_detect.py
- dataset
   - train
      - with_mask
      - without_mask
   - test
      - with_mask
      - without_mask
   - real
      - real_image
### python環境需求
- tensorflow == 2.0.4
- imutils
- matplotlib
### 訓練模型
開啟terminal，切換到mian_approach_train.py檔案的目錄下，執行python main_approach_train.py -d {你的train資料集目錄}，即可開始訓練模型。
### 測試
main_approach_test.py
執行方法跟上面一樣，其中的輸出是有沒有戴口罩的照片數，以及兩者判斷對的。
### 實際應用
如果要應用在實際影像上（有可能有許多張人臉或是沒有人臉），需要用main_approach_detect.py的程式碼，執行方法也跟上面一樣，會輸出匡出人臉並判斷有沒有戴口罩的影像。
