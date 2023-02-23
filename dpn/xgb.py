from loguru import logger
from sklearn.model_selection import StratifiedKFold
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from joblib import dump
import argparse
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


id2feature = {
    0:'非DPN组',
    1:'DPN组'
}
logger.add('./log/xgb/{time}.log')
parse = argparse.ArgumentParser(description='XGBoost 参数')
parse.add_argument('--TRAIN_XGB',type=bool,default=True,help='是否训练XGBoost')
parse.add_argument('--TUNE_TIMES',type=int,default=20,help='调参次数')
parse.add_argument('--RANDOM_STATE',type=int,default=43,help='随机参数')

args = parse.parse_args()
TRAIN_XGB = args.TRAIN_XGB
random_state = args.RANDOM_STATE
TUNE_TIMES = args.TUNE_TIMES

logger.info(f'TRAIN_XGB={TRAIN_XGB} TUNE_TIMES={TUNE_TIMES} RANDOM_STATE={TUNE_TIMES}')

dpn_resampled = pd.read_excel('../output/dpn/data/dpn_resampled.xlsx')
X_resampled  = dpn_resampled.drop(columns='分组')
y_resampled = dpn_resampled['分组']

zh2en = {
    '年龄':'Age' ,
    '病程':'Diabetes duration',
    '活化部分凝血活酶时间（APTT）':'Activeated partial thromboplasting time (APTT)',
    '总胆固醇':'total cholesterol(TC)',
    '尿酸':'Uric acid(UA)',
    '血红蛋白':'Hemoglobin (Hb)',
    'C2/C0':'C2/C0',
    'NLR':'NLR',
    '白蛋白':'Albumin(ALB)',
    '尿素': 'UREA',
    '胰岛素抵抗指数':'HOMA-IR',
    '肌酐':'Creatinine(Cr)',
    '24h尿蛋白定量':'24-hour urinary protein quantity',
    '总胆红素':'Total bilirubin',
    '糖化血红蛋白':'glycated hemoglobin A1C (HbA1c)',
    '尿蛋白定量':'Urine protein quantity'
}
def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=43)
    param = {
        'verbosity': 0,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'tree_method': 'exact',
        'n_estimators':trial.suggest_int('n_estimators',100,500,step=50),
        'max_depth': trial.suggest_int("max_depth", 8,20,step=2),
        'grow_policy': trial.suggest_categorical("grow_policy", ['depthwise', 'lossguide']),
        'learning_rate': trial.suggest_float("learning_rate", 1e-8, 1, log=True),
        'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'random_state': 43
    }
    clf_xgb = xgb.XGBClassifier(**param).fit(train_x, train_y)
    preds = clf_xgb.predict(valid_x)
    return accuracy_score(valid_y, preds)
if TRAIN_XGB:
    for i in tqdm(range(TUNE_TIMES)):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        logger.info(f'第{i+1}/{TUNE_TIMES}调参结束，最佳结果为:{study.best_value}')
        logger.info(f'调参结束，最佳参数为:{study.best_params}')
fixed_params = {
        'verbosity': 0,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'tree_method': 'exact',
        'n_estimators':500,
        'max_depth': 8,
        'grow_policy': 'lossguide',
        'learning_rate': 0.16988838433253076,
        'gamma': 0.8731716765537924,
        'reg_lambda':  0.0013501913428471217,
        'reg_alpha': 0.0003173545927786465,
        'subsample':  0.8207157400822733,
        'colsample_bytree': 0.6604573433687222,
        'random_state': 43
}
Accuracy = []
Precision = []
Recall = []
F1 = []
if TRAIN_XGB:
    clf_xgb = xgb.XGBClassifier(**study.best_params)
else:
    clf_xgb = xgb.XGBClassifier(**fixed_params)
logger.warning(clf_xgb.get_params())
clf_xgb.fit(X_resampled,y_resampled)
kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=43)
for train_index, test_index in kf.split(X_resampled,y_resampled):
    clf_xgb.fit(X_resampled.loc[train_index], y_resampled[train_index])
    preds = clf_xgb.predict(X_resampled.loc[test_index])
    accuracy = accuracy_score(y_resampled[test_index], preds)
    Accuracy.append(accuracy)
    precision = precision_score(y_resampled[test_index], preds)
    Precision.append(precision)
    recall = recall_score(y_resampled[test_index], preds)
    Recall.append(recall)
    f1 = f1_score(y_resampled[test_index], preds)
    F1.append(f1)
    logger.warning(f'{round(np.mean(accuracy), 3)}\t{round(np.mean(precision), 3)}\t'
                f'{round(np.mean(recall), 3)}\t{round(np.mean(f1), 3)}')
logger.warning(f'accuracy\t\tmean:{round(np.mean(Accuracy), 3)}\tstd:{round(np.std(Accuracy), 3)}')
logger.warning(f'precision\t\tmean:{round(np.mean(Precision), 3)}\tstd:{round(np.std(Precision), 3)}')
logger.warning(f'recall\t\tmean:{round(np.mean(Recall), 3)}\tstd:{round(np.std(Recall), 3)}')
logger.warning(f'f1\t\tmean:{round(np.mean(F1), 3)}\tstd:{round(np.std(F1), 3)}')
if round(np.mean(Accuracy), 3) >=0.8:
    from joblib import dump
    logger.info('准确率>=0.8,保存模型...')
    dump(clf_xgb,'./output/dpn/model/'+'XGB_'+str(round(np.mean(Accuracy), 3))+'.joblib')
else:
    logger.info('准确率<0.8,继续调参...')

