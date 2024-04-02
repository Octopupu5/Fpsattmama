import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from optuna import Trial
from sklearn.model_selection import cross_val_score, KFold
from optuna import create_study

def process_table() -> pd.DataFrame:
    df = pd.read_excel("final_dataset.xlsx")
    df.drop(columns = [
        "full_name",
        "citizenship",
        "current_club",
        "link",
        "club_link",
        "price_history",
        "club_league"
    ], inplace = True)
    
    def social_media(cell):
        if cell is np.nan:
            return 0
        return 1

    def decimal(cell):
        return float(cell.replace(",", "")) if type(cell) is str else cell


    df["social_media"] = df["social_media"].apply(social_media)
    df["current_price"] = df["current_price"].apply(decimal)
    df["minutes_played"] = df["minutes_played"].apply(decimal)
    df["club_price"] = df["club_price"].apply(decimal)
    df["followers"] = df["followers"].apply(decimal)

    # invalid_index = []
    # for i, row in df.iterrows():
    #     if type(row["price_history"]) is float:
    #         invalid_index.append(i)
    #         continue
    #     arr = row["price_history"].split(";")
    #     if len(arr) <= 1:
    #         invalid_index.append(i)
    #         continue
    #     previous_stat = arr[-2]
    #     stats = json.loads(previous_stat.replace("'", '"'))
    #     if stats["value"] == "-":
    #         invalid_index.append(i)
    #         continue
    #     df.loc[i, "previous_price"] = stats["value"]
    
    # df.drop(index = invalid_index, inplace = True)
    
    # def price(cell):
    #     cell = cell[1:]
    #     if cell[-1] == 'k':
    #         return float(cell[:-1]) * 1_000
    #     else:
    #         return float(cell[:-1]) * 1_000_000

    # df["previous_price"] = df["previous_price"].apply(price)

    return df


def RegressionOnPriceTrainUsingGridSearch(df_model : pd.DataFrame):
    df = df_model.copy(deep = True)
    if "current_price" not in df.columns:
        raise "Nothing to learn on"

    # df["target"] = (df["current_price"] > df["previous_price"]).astype("int64")
    # df.drop(columns = ["previous_price"], inplace = True)
    X = df.drop(columns = ["current_price"])
    y = df["current_price"]

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 0, shuffle = True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pipe = Pipeline(
        steps = [
            ("scaler", StandardScaler()),
            ("variance", VarianceThreshold(0.01)),
            ("selection", SelectFromModel(Lasso(5.0))),
            ("regressor", Lasso(5.0)),
        ]
    )

    param_grid = {
        "variance__threshold": [0.002, 0.0035, 0.005, 0.0075, 0.009, 0.01, 0.012],
        "selection__estimator__alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        "regressor__alpha": [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0],
    }

    grid_search = GridSearchCV(pipe, param_grid, cv = 5)

    grid_search.fit(X_train, y_train)
    pipe_best = grid_search.best_estimator_

    return pipe_best

def WriteRegressionUsingGridSearch(df : pd.DataFrame):
    lr_model = RegressionOnPriceTrainUsingGridSearch(df)
    filename = "regression_model_grid.sav"
    joblib.dump(lr_model, filename)


def RegressionOnPriceTrainUsingOptunaSearch(df_model : pd.DataFrame, loss : str):
    X = df_model.drop(columns = ["current_price"])
    y = df_model["current_price"]

    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 0, shuffle = True
    )

    def instantiate_lasso_regression(trial : Trial) -> Lasso:
        params = {
            "alpha" : trial.suggest_float("alpha", 0.01, 50)
        }
        return Lasso(**params)

    def instantiate_thresholder(trial : Trial) -> VarianceThreshold:
        params = {
            "threshold" : trial.suggest_float("threshold", 0.0001, 0.1)
        }
        return VarianceThreshold(**params)

    def instantiate_model(trial : Trial) -> Pipeline:
        pipe = Pipeline(
            steps = [
                ("scaler", StandardScaler()),
                ("variance", instantiate_thresholder(trial = trial)),
                ("selection", SelectFromModel(instantiate_lasso_regression(trial = trial))),
                ("regressor", instantiate_lasso_regression(trial = trial))
            ]
        )
        return pipe

    def objective(trial : Trial, loss : str, X : pd.DataFrame, y : np.ndarray | pd.Series, random_state : int = 42) -> float:
        loss_functions = {
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        }
        model = instantiate_model(trial = trial)
        kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)
        scores = cross_val_score(model, X, y, scoring = loss_functions[loss], cv = kf)
        return np.min([np.mean(scores), np.median(scores)])

    study = create_study(study_name = "Lasso_optimization", direction = "maximize")
    study.optimize(lambda trial : objective(trial, loss, X_train, y_train), n_trials = 400)
    model = instantiate_model(trial = study.best_trial)
    return model

def WriteRegressionUsingOptunaSearch(df : pd.DataFrame, loss : str):
    lr_model = RegressionOnPriceTrainUsingOptunaSearch(df, loss)
    filename = "regression_model_optuna.sav"
    joblib.dump(lr_model, filename)