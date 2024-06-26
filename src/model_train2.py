import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss
import time
import math

# Debugging.
from src.details_error import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

def evaluate_models_cv_regression(models, X_train, y_train, n_folds=5):

    try:
        # Dictionaries with validation and training scores of each model for plotting further.
        models_sq_mean_scores = dict()
        models_mean_scores = dict()
        models_r2_scores = dict()
        models_time = dict()

        for model in models:
            # Getting the model object from the key with his name.
            model_instance = models[model]

            start_time = time.time()

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            sq_mean_scores = cross_val_score(model_instance, X_train, y_train, scoring='neg_mean_squared_error', cv=n_folds)
            mean_scores = cross_val_score(model_instance, X_train, y_train, scoring='neg_mean_absolute_error', cv=n_folds)
            r2_scores = cross_val_score(model_instance, X_train, y_train, scoring='r2', cv=n_folds)

            sq_mean_score = np.sqrt(-1 * sq_mean_scores.mean())
            mean_score = (-1 * mean_scores.mean())
            r2_score = (r2_scores.mean())
        
            end_time = time.time()
            training_time = round(end_time - start_time, 5)

            # Adding the model scores to the validation and training scores dictionaries.
            models_sq_mean_scores[model] = sq_mean_score
            models_mean_scores[model] = mean_score
            models_r2_scores[model] = r2_score
            models_time[model] = training_time

            # Printing the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'RMSE score: {sq_mean_score}')
            print(f'MAE score: {mean_score}')
            print(f'R² score: {r2_score}')
            print(f'Training time: {training_time} seconds')
            print()


        # Plotting the results.
        print('Plotting the results: ')

        # Converting scores to a dataframe
        RMSE_df = pd.DataFrame(list(models_sq_mean_scores.items()), columns=['Model', 'RMSE'])
        MAE_df = pd.DataFrame(list(models_mean_scores.items()), columns=['Model', 'MAE'])
        R2_df = pd.DataFrame(list(models_r2_scores.items()), columns=['Model', 'R2'])
        TIME_df = pd.DataFrame(list(models_time.items()), columns=["Model", "Time"])
        eval_df = MAE_df.merge(RMSE_df, on='Model')
        eval_df = eval_df.merge(R2_df, on='Model')
        eval_df = eval_df.merge(TIME_df, on="Model")

        # Sorting the dataframe by the best RMSE.
        eval_df  = eval_df.sort_values(['MAE'], ascending=True).reset_index(drop=True)

    
        # Plotting each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=(16, 6))
        width = 0.35

        x = np.arange(len(eval_df['Model']))
        y = np.arange(len(eval_df['MAE']))

        val_bars = ax.bar(x - width/2, eval_df['RMSE'], width, label='Root Mean Squared Error', color="blue")
        train_bars = ax.bar(x + width/2, eval_df['MAE'], width, label='Mean Absolute Error', color="lightgreen")

        ax.set_xlabel('Model', labelpad=20, fontsize=10.8)
        ax.set_ylabel('Errors', labelpad=20, fontsize=10.8)
        ax.set_title("Performances dos Modelos", fontweight='bold', fontsize=12, pad=30)
        ax.set_xticks(x, eval_df['Model'], rotation=0, fontsize=10.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

        ax.legend(loc='upper left')

        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))