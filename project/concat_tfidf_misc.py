import pandas as pd
import os

TFIDF_DIR, MISC_ENCODED_DIR, NEW_TFIDF_MISC_DIR = "data/tfidf/", "data/misc/encoded/", "data/tfidf_misc/"
tfidf_train_csv, misc_encoded_train_csv = TFIDF_DIR + "tfidf.train.csv", MISC_ENCODED_DIR + "misc.encoded.train.csv"
tfidf_test_csv, misc_encoded_test_csv = TFIDF_DIR + "tfidf.test.csv", MISC_ENCODED_DIR + "misc.encoded.test.csv"
tfidf_eval_csv, misc_encoded_eval_csv = TFIDF_DIR + "tfidf.eval.anon.csv", MISC_ENCODED_DIR + "misc.encoded.eval.csv"

os.mkdir(NEW_TFIDF_MISC_DIR)
tfidf_misc_train_csv = NEW_TFIDF_MISC_DIR + "tfidf.misc.train.csv"
tfidf_misc_test_csv = NEW_TFIDF_MISC_DIR + "tfidf.misc.test.csv"
tfidf_misc_eval_csv = NEW_TFIDF_MISC_DIR + "tfidf.misc.eval.csv"

# generate new file: NEW_TFIDF_MISC_DIR + "tfidf.misc.train.csv"
df_tfidf, df_misc = pd.read_csv(tfidf_train_csv), pd.read_csv(misc_encoded_train_csv).drop("label", axis='columns')
pd.concat([df_tfidf, df_misc], axis=1).to_csv(tfidf_misc_train_csv, index=False)

# generate new file: NEW_TFIDF_MISC_DIR + "tfidf.misc.test.csv"
df_tfidf, df_misc = pd.read_csv(tfidf_test_csv), pd.read_csv(misc_encoded_test_csv).drop("label", axis='columns')
pd.concat([df_tfidf, df_misc], axis=1).to_csv(tfidf_misc_test_csv, index=False)

# generate new file: NEW_TFIDF_MISC_DIR + "tfidf.misc.eval.csv"
df_tfidf, df_misc = pd.read_csv(tfidf_eval_csv), pd.read_csv(misc_encoded_eval_csv).drop("label", axis='columns')
pd.concat([df_tfidf, df_misc], axis=1).to_csv(tfidf_misc_eval_csv, index=False)

