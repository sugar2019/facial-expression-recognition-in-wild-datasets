# facial-expression-recognition-in-wild-datasets
use softmax, weighted softmax, weighted cluster,weighted Island loss etal on RAF-DB and FERPLUS2013,based on Pytorch

run train_RAF_sm_pre.py and train_RAF_Wsm_pre.py will save trained models, you can preserve the last file in every model_file, which has the best test accuracy.
Then you can run test_cm.py to produce confusion matrix on RAF-DB. Don't forget modify the path of model and save img.
If you want to see the confusion matrix on FERPLUS,then you run test_generallize_cm.py

The other train_XXXX.py all use "centers"，so you run them to produce correlate model.
Then you can run test_cm_center.py to produce confusion matrix on RAF-DB. Don't forget modify the path of model and save img.
If you want to see the confusion matrix on FERPLUS,then you run test_generallize_cm_center.py
