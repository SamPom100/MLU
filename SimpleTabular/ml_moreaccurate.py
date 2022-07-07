import re, os, pandas as pd, autogluon.core as ag
from autogluon.tabular import TabularPredictor, TabularDataset

def first_num(in_val):
    num_string = in_val.split(" ")[0]
    digits = re.sub(r"[^0-9\.]", "", num_string)
    return float(digits)
def year_get(in_val):
    m = re.compile(r"\d{4}").findall(in_val)
    # print(in_val, m)
    if len(m) > 0:
        return int(m[0])
    else:
        return None
def month_get(in_val):
    m = re.compile(r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec").findall(in_val)
    # print(in_val, m)
    if len(m) > 0:
        return m[0]
    else:
        return "None"
def drop_features(in_feat):
    train_data_feateng.drop(in_feat, axis=1, inplace=True)
    val_data_feateng.drop(in_feat, axis=1, inplace=True)
    return
def filterGenre(in_str):
    if "Books" in in_str:
        in_str = in_str.replace("Books","")
        if "()" in in_str:
            in_str = in_str.replace("()","")
    return in_str

df_train = TabularDataset(data="./data/training.csv")
df_test = TabularDataset(data="./data/mlu-leaderboard-test.csv")

#CLEAN TRAIN DATA
full_feateng = df_train.copy()
full_feateng['Reviews-n'] = full_feateng['Reviews'].apply(first_num)
full_feateng['Ratings-n'] = full_feateng['Ratings'].apply(first_num)
full_feateng['hard-paper'] = full_feateng['Edition'].apply(lambda x : x.split(",")[0])
full_feateng['year'] = full_feateng['Edition'].apply(year_get)
full_feateng['month'] = full_feateng['Edition'].apply(month_get)

#Split BookCategory into two
full_feateng['Genre'] = full_feateng['Genre'].apply(filterGenre)
full_feateng['BookCategory'] = full_feateng['BookCategory'].apply(filterGenre)

full_feateng[['Genre-1', 'Genre-2']] = full_feateng['Genre'].str.split('&', 1, expand=True)
full_feateng[['Cata-1', 'Cata-2']] = full_feateng['BookCategory'].str.split('&', 1, expand=True)
full_feateng.drop(['Edition', 'Ratings', 'Reviews','BookCategory','Genre'], axis=1, inplace=True)



#BUILD MODEL
predictor = TabularPredictor(label="Price",).fit(
    full_feateng,
    hyperparameters='multimodal',
    presets='best_quality',
)


#CLEAN TEST DATA
test_data_feateng = df_test.copy()
test_data_feateng['Reviews-n'] = test_data_feateng['Reviews'].apply(first_num)
test_data_feateng['Ratings-n'] = test_data_feateng['Ratings'].apply(first_num)
test_data_feateng['hard-paper'] = test_data_feateng['Edition'].apply(lambda x : x.split(",")[0])
test_data_feateng['year'] = test_data_feateng['Edition'].apply(year_get)
test_data_feateng['month'] = test_data_feateng['Edition'].apply(month_get)

#Split BookCategory into two
test_data_feateng['Genre'] = test_data_feateng['Genre'].apply(filterGenre)
test_data_feateng['BookCategory'] = test_data_feateng['BookCategory'].apply(filterGenre)

test_data_feateng[['Genre-1', 'Genre-2']] = test_data_feateng['Genre'].str.split('&', 1, expand=True)
test_data_feateng[['Cata-1', 'Cata-2']] = test_data_feateng['BookCategory'].str.split('&', 1, expand=True)
test_data_feateng.drop(['Edition', 'Ratings', 'Reviews','BookCategory','Genre'], axis=1, inplace=True)

#BUILD PREDICTIONS
prediction = predictor.predict(test_data_feateng)

submission = df_test[["ID"]].copy(deep=True)
submission["Price"] = prediction
submission.to_csv(
    "./data/predictions/multiModalComplex.csv",
    index=False,
)