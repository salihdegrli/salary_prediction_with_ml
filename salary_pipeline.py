

import joblib
from lightgbm import LGBMRegressor
from pandas import CategoricalDtype
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from xgboost import XGBRegressor
from Helper.eda import *
from Helper.data_prep import *
import warnings

warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', None)

# Data Preprocessing & Feature Engineering
def hitters_data_prep(df):
    rename_col_up_low(df, method="upper")
    # feature engineering
    df['EXPERIENCE_LEVEL'] = pd.cut(x=df['YEARS'], bins=[0, 2, 5, 10, max(df["YEARS"])],
                                    labels=["JUNIOR", "MID", "SENIOR", "EXPERT"]).astype("O")
    # to_ordinal
    levels = ["JUNIOR", "MID", "SENIOR", "EXPERT"]
    df['EXPERIENCE_LEVEL'] = df['EXPERIENCE_LEVEL'].astype(CategoricalDtype(categories=levels, ordered=True))
    df["NEW_BATTINGAVERAGE"] = df["CHITS"] / df["CATBAT"]
    df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
    df["NEW_TOTALBASES"] = ((df["CHITS"] * 2) + (4 * df["CHMRUN"]))
    df["NEW_SLUGGINGPERCENTAGE"] = df["NEW_TOTALBASES"] / df["CATBAT"]
    df["NEW_ISOLATEDPOWER"] = df["NEW_SLUGGINGPERCENTAGE"] - df["NEW_BATTINGAVERAGE"]
    df["NEW_TRIPLECROWN"] = (df["CHMRUN"] * 0.4) + (df["CRBI"] * 0.25) + (df["NEW_BATTINGAVERAGE"] * 0.35)
    df["NEW_BATTINGAVERAGEONBALLS"] = (df["CHITS"] - df["CHMRUN"]) / (df["CATBAT"] - df["CHMRUN"])
    df['NEW_PUTOUTSYEARS'] = df['PUTOUTS'] * df['YEARS']
    df["NEW_RBIWALKSRATIO"] = df["RBI"] / df["WALKS"]
    df["NEW_CHMRUBCATBATRATIO"] = df["CHMRUN"] / df["CATBAT"]

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df.dropna(inplace=True)
    remove_outlier_with_lof(df, num_cols, threshold=5, plot=False, inplace=True)

    df = one_hot_encoder(df, cat_cols)

    df.dropna(inplace=True)
    y = df["SALARY"]
    X = df.drop(["SALARY"], axis=1)
    df.head()
    return X, y


# Base Models
def base_models(X, y, scoring='neg_root_mean_squared_error'):
    print("Base Models....")
    regressors = [('LR', LinearRegression()),
                  ("RF", RandomForestRegressor()),
                  ('Adaboost', AdaBoostRegressor()),
                  ('GBM', GradientBoostingRegressor()),
                  ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric="rmse")),
                  ('LightGBM', LGBMRegressor())
                  ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y,
                                    cv=5,
                                    scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# Hyperparameter Optimization

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

regresors = [("GBM", GradientBoostingRegressor(), gbm_params),
             ("RF", RandomForestRegressor(), rf_params),
             ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='rmse'),
              xgboost_params),
             ('LightGBM', LGBMRegressor(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=5, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regresors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y,
                                    cv=cv,
                                    scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor,
                               params,
                               cv=cv,
                               n_jobs=-1,
                               verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y,
                                    cv=cv,
                                    scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_reg = VotingRegressor(estimators=[("GBM", best_models["GBM"]),
                                             ("XGBoost", best_models["XGBoost"]),
                                             ('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])]).fit(X, y)
    cv_results = cross_validate(voting_reg, X, y,
                                cv=5,
                                scoring=["neg_mean_squared_error", "neg_root_mean_squared_error",
                                         "neg_mean_absolute_error"])
    print(f"MSE: {cv_results['test_neg_mean_squared_error'].mean()}")
    print(f"RMSE: {cv_results['test_neg_root_mean_squared_error'].mean()}")
    print(f"MAE: {cv_results['test_neg_mean_absolute_error'].mean()}")
    return voting_reg


################################################
# Pipeline Main Function
################################################

def main():
    df = load_data("hitters.csv")
    X, y = hitters_data_prep(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    base_models(X_train, y_train)
    best_models = hyperparameter_optimization(X_train, y_train)
    #voting_clf = voting_regressor(best_models, X_train, y_train)
    best_models["XGBoost"].fit(X_train, y_train)
    y_predicted = best_models["XGBoost"].predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    print(rmse)
    # os.chdir(r"C:\Users\pc\PycharmProjects\DSMLBC\hafta8-machine_learning2")
    # joblib.dump(voting_clf, "voting_clf_hitters.pkl")
    #print("Voting_clf has been created")
    #plot_importance(best_models["RF"], X_train, len(X_train))
    #return voting_clf


if __name__ == "__main__":
    main()
