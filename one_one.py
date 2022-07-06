import pandas as pd
import cufflinks as cf
import shap

from utils import preprocess, readJSON, get_logger

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import optuna
from collections import Counter
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

logger = get_logger(logger_name=__name__, filename='./log/one_one.log')
X, y = preprocess(path='./input/心总表.xlsx', sheet_name='总表')
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
le = LabelEncoder()
y = le.fit_transform(y_resampled)
data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y, columns=['证名'])], axis=1)
Accuracy = []
Precision = []
Recall = []
F1 = []
for i in tqdm(range(8)):
    for j in tqdm(range(8), leave=False):
        if i != j:
            i_resampled = data.drop(data[data['证名'] != i].index)
            j_resampled = data.drop(data[data['证名'] != j].index)
            i_resampled['证名'] = 0
            j_resampled['证名'] = 1
            data_resampled = pd.concat([i_resampled, j_resampled], axis=0)
            data_x = data_resampled.drop(labels='证名', axis=1)
            data_x.reset_index(drop=True, inplace=True)
            data_y = data_resampled['证名']
            data_y.reset_index(drop=True, inplace=True)
            train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42)


            def obj_xgb(trial):
                dtrain = xgb.DMatrix(train_x, label=train_y)
                dvalid = xgb.DMatrix(valid_x, label=valid_y)
                param = {
                    'verbosity': 0,
                    'objective': 'binary:logistic',
                    'tree_method': 'exact',
                    'booster': 'gbtree',
                    # 'booster':trial.suggest_categorical('booster',['gbtree','gblinear','dart']),
                    'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                    'max_depth': trial.suggest_int("max_depth", 3, 9, step=2),
                    'min_child_weight': trial.suggest_int("min_child_weight", 2, 10),
                    'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
                    'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                    'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
                }
                bst = xgb.train(param, dtrain)
                preds = bst.predict(dvalid)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                accuracy = accuracy_score(valid_y, preds)
                return accuracy


            study = optuna.create_study(direction="maximize", study_name='xgb_optuna')
            study.optimize(obj_xgb, n_trials=100, timeout=600)
            trial = study.best_trial
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            total_topN = {}
            key = 1
            for train_index, test_index in kf.split(data_x, data_y):
                dtrain = xgb.DMatrix(data_x.loc[train_index], label=data_y.loc[train_index])
                dvalid = xgb.DMatrix(data_x.loc[test_index], label=data_y.loc[test_index])
                model = xgb.train(study.best_trial.params, dtrain)
                preds = model.predict(dvalid)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                accuracy = accuracy_score(data_y.loc[test_index], preds)
                precision = precision_score(data_y.loc[test_index], preds)
                recall = recall_score(data_y.loc[test_index], preds)
                f1 = f1_score(data_y.loc[test_index], preds)
                Accuracy.append(accuracy)
                Precision.append(precision)
                Recall.append(recall)
                F1.append(f1)
                explainer = shap.TreeExplainer(model)
                shap_values_XGBoost_train = explainer.shap_values(data_x.loc[train_index])
                SHAP_XGBoost = pd.DataFrame(data=shap_values_XGBoost_train,
                                            columns=data_x.loc[train_index].columns).mean(axis=0).sort_values(
                    ascending=False)
                SHAP_topN = list(SHAP_XGBoost[SHAP_XGBoost.values > 0].items())
                topN = [readJSON('./input/id2feature.json')[i[0]] for i in SHAP_topN]
                total_topN[key] = topN
                topN_detail = [(readJSON('./input/id2feature.json')[i[0]], round(i[1], 3)) for i in SHAP_topN]
                logger.info(
                    f'{i + 1}\t{j + 1}\t{key}\t{round(accuracy, 3)}\t{round(precision, 3)}\t{round(recall, 3)}\t{round(f1, 3)}\t\t{topN}\t\t{topN_detail}')
                key = key + 1
            uniqueFeature = set(total_topN[1])
            featureFrequency = Counter([])
            for item in total_topN:
                uniqueFeature = uniqueFeature & set(total_topN[item])
                featureFrequency = featureFrequency + Counter(total_topN[item])
            logger.info(f'{i + 1}\t{uniqueFeature}')
            logger.info(f'{i + 1}\t{sorted(featureFrequency.items(),key=lambda pair:pair[1],reverse=True)}')
            logger.info(f'{i + 1}\t{round(np.mean(Accuracy), 3)}\t{round(np.mean(Precision), 3)}\t'
                        f'{round(np.mean(Recall), 3)}\t{round(np.mean(F1), 3)}')
