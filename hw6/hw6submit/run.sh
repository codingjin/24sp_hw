#!/bin/bash

#1 SVM
python3 svm_cv.py
printf "\n"

python3 svm_report.py
printf "\n"


#2 Logistic Regression
python3 lr_cv.py
printf "\n"

python3 lr_report.py
printf "\n"

#3 SVM over trees
# long time to train depth-5 trees and generate transformed datasets
python3 tree5.py
printf "\n"

# long time to train depth-10 trees and generate transformed datasets
python3 tree10.py
printf "\n"

cd extra/

python3 svmd5_cv.py
printf "\n"

python3 svmd10_cv.py
printf "\n"

python3 svmd10_report.py
printf "\n"

