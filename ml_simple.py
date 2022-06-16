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






df_train = TabularDataset(data="../data/training.csv")
df_test = TabularDataset(data="../data/mlu-leaderboard-test.csv")

#CLEAN TRAIN DATA
full_feateng = df_train.copy()
full_feateng['Reviews-n'] = full_feateng['Reviews'].apply(first_num)
full_feateng['Ratings-n'] = full_feateng['Ratings'].apply(first_num)
full_feateng['hard-paper'] = full_feateng['Edition'].apply(lambda x : x.split(",")[0])
full_feateng['year'] = full_feateng['Edition'].apply(year_get)
full_feateng['month'] = full_feateng['Edition'].apply(month_get)
full_feateng.drop(['Edition', 'Ratings', 'Reviews'], axis=1, inplace=True)






# Set Neural Net options
# Specifies non-default hyperparameter values for neural network models
nn_options = {
    # number of training epochs (controls training time of NN models)
    "num_epochs": 50,
    # learning rate used in training (real-valued hyperparameter searched on log-scale)
    "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
    # activation function used in NN (categorical hyperparameter, default = first entry)
    "activation": ag.space.Categorical("relu", "softrelu", "tanh"),
    # each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
    "layers": ag.space.Categorical([100], [1000], [200, 100], [300, 200, 100]),
    # dropout probability (real-valued hyperparameter)
    "dropout_prob": ag.space.Real(0.0, 0.5, default=0.1),
}

# Set GBM options
# Specifies non-default hyperparameter values for lightGBM gradient boosted trees
gbm_options = {
    # number of boosting rounds (controls training time of GBM models)
    "num_boost_round": 500,
    # number of leaves in trees (integer hyperparameter)
    "num_leaves": ag.space.Int(lower=26, upper=66, default=36),
}

# Add both NN and GBM options into a hyperparameter dictionary
# hyperparameters of each model type
# When these keys are missing from the hyperparameters dict, no models of that type are trained
hyperparameters = {
    "GBM": gbm_options,
    "NN": nn_options,
}

# To tune hyperparameters using Bayesian optimization to find best combination of params
search_strategy = "auto"

# Number of trials for hyperparameters
num_trials = 5

# HPO is not performed unless hyperparameter_tune_kwargs is specified
hyperparameter_tune_kwargs = {
    "num_trials": num_trials,
    "scheduler": "local",
    "searcher": search_strategy,
}


#BUILD MODEL
predictor = TabularPredictor(label="Price").fit(
    train_data=full_feateng
)



predictor.feature_importance(full_feateng)



#CLEAN TEST DATA
test_data_feateng = df_test.copy()
test_data_feateng['Reviews-n'] = test_data_feateng['Reviews'].apply(first_num)
test_data_feateng['Ratings-n'] = test_data_feateng['Ratings'].apply(first_num)
test_data_feateng['hard-paper'] = test_data_feateng['Edition'].apply(lambda x : x.split(",")[0])
test_data_feateng['year'] = test_data_feateng['Edition'].apply(year_get)
test_data_feateng['month'] = test_data_feateng['Edition'].apply(month_get)
test_data_feateng.drop(['Edition', 'Ratings', 'Reviews'], axis=1, inplace=True)

#BUILD PREDICTIONS
prediction = predictor.predict(test_data_feateng)




submission = df_test[["ID"]].copy(deep=True)
submission["Price"] = prediction
submission.to_csv(
    "../data/predictions/autoStack1.csv",
    index=False,
)

