import autogluon.text, autogluon.tabular, pandas as pd, numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.text import TextPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config



'''
python3 run_competition.py --train_file price_of_books/Participants_Data/Data_Train.xlsx \
                           --test_file price_of_books/Participants_Data/Data_Test.xlsx \
                           --sample_submission price_of_books/Participants_Data/Sample_Submission.xlsx \
                           --task price_of_books \
                           --eval_metric r2 \
                           --exp_dir ag_price_of_books \
                           --mode stacking 2>&1  | tee -a ag_price_of_books/log.txt
'''

def load_price_of_books():
    train_df = TabularDataset(data="./data/training.csv")
    test_df = TabularDataset(data="./data/mlu-leaderboard-test.csv")
    # Convert Reviews
    train_df.loc[:, 'Reviews'] = pd.to_numeric(train_df['Reviews'].apply(
        lambda ele: ele[:-len(' out of 5 stars')]))
    test_df.loc[:, 'Reviews'] = pd.to_numeric(
        test_df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    # Convert Ratings
    train_df.loc[:, 'Ratings'] = pd.to_numeric(train_df['Ratings'].apply(
        lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    test_df.loc[:, 'Ratings'] = pd.to_numeric(
        test_df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    # Convert Price to log scale
    train_df.loc[:, 'Price'] = np.log10(train_df['Price'] + 1)
    return train_df, test_df, 'Price'



train_df, test_df, label_column = load_price_of_books()

hyperparameters = get_hyperparameter_config('multimodal')
hyperparameters['AG_TEXT_NN']['presets'] = 'best_quality'
#predictor = TabularPredictor(label=label_column,eval_metric='r2',path='ag_price_of_book')
predictor = TabularPredictor.load("ag_price_of_book/")
#predictor.fit(train_data=train_df,hyperparameters=hyperparameters,num_bag_folds=5,num_stack_levels=1)


prediction = predictor.predict(test_df)
submission = test_df[["ID"]].copy(deep=True)
submission["Price"] = prediction
submission.to_csv(
    "./data/predictions/pureGluon.csv",
    index=False,
)