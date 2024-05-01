#!/bin/bash

# label misc dataset
python3 label_misc.py

# do one_hot_encode on misc dataset
python3 one_hot_encode_misc.py

# cv split misc dataset
python3 misc_cv_split.py

# concat the tfidf and (one_hot_encoded)misc dataset
python3 concat_tfidf_misc.py

# cv split tfidf_misc dataset
python3 cv_split_tfidf_misc.py

