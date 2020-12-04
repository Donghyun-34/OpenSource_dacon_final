import gc
from matplotlib import rcParams, pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import re
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import plot_model, to_categorical
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import warnings

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':

    # 데이터 경로 지정
    data_dir = Path('./data')
    feature_dir = Path('./build/feature')
    val_dir = Path('./build/val')
    tst_dir = Path('./build/tst')
    sub_dir = Path('./build/sub')

    dirs = [feature_dir, val_dir, tst_dir, sub_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 데이터 불러오기
    trn_file = data_dir / 'train.csv'
    tst_file = data_dir / 'test_x.csv'
    sample_file = data_dir / 'sample_submission.csv'

    target_col = 'author'
    n_fold = 30
    n_class = 5
    seed = 5

    algo_name = 'mta'
    feature_name = 'stacking_final3'
    model_name = f'{algo_name}_{feature_name}'

    feature_file = feature_dir / f'{feature_name}.csv'
    p_val_file = val_dir / f'{model_name}.val.csv'
    p_tst_file = tst_dir / f'{model_name}.tst.csv'
    sub_file = sub_dir / f'{model_name}.csv'

    model_names = ['lstm_glove', 'lstm_glove2', 'lstm_glove3', 'lstm_glove4', 'lstm_glove5',
                   'mta_emb', 'mta_emb2', 'mta_emb3', 'mta_emb4',
                   'mta_stacking_lstm_1', 'mta_stacking_lstm_2', 'mta_stacking_lstm_3',
                   'lr_tfidf', 'lr_tfidf2',
                   'cnn_emb', 'cnn_emb2',
                   'mta_attension', 'mta_attension2', 'mta_attension3', 'mta_attension4', 'mta_attension5',
                   'mta_attension6', 'mta_attension7', 'mta_attension8',
                   'mta_stacking', 'mta_stacking2', 'mta_stacking3', 'mta_stacking4', 'mta_stacking5', 'mta_stacking6',
                   'mta_stacking7', 'mta_stacking8', 'mta_stacking9', 'mta_stacking10',
                   'mta_stacking11', 'mta_stacking12', 'mta_stacking13', 'mta_stacking14', 'mta_stacking15',
                   'mta_stacking16', 'mta_stacking17', 'mta_stacking18', 'mta_stacking19', 'mta_stacking20',
                   'mta_stacking21', 'mta_stacking22', 'mta_stacking23',
                   'mta_stackingdart', 'mta_stackingdart2', 'mta_stackingdart3', 'mta_stackingdart4',
                   'mta_stackingrf',
                   'mta_stacking_final', 'mta_stacking_final2']
    trn = []
    tst = []
    feature_names = []
    for model in model_names:
        trn.append(np.loadtxt(val_dir / f'{model}.val.csv', delimiter=','))
        tst.append(np.loadtxt(tst_dir / f'{model}.tst.csv', delimiter=','))
        feature_names += [f'{model}_class0', f'{model}_class1', f'{model}_class2', f'{model}_class3', f'{model}_class4']

    trn = np.hstack(trn)
    tst = np.hstack(tst)

    train = pd.read_csv(trn_file, index_col=0)
    y = train['author'].values

    # 하이퍼 파라미터 튜닝
    params = {
        "objective": "multiclass",
        "n_estimators": 16608,
        "subsample_freq": 1,
        "n_jobs": -1,
    }

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "num_leaves": hp.choice("num_leaves", [40, 127, 255, 512, 1024, 2048]),
        "colsample_bytree": hp.quniform("colsample_bytree", .5, .9, 0.1),
        "subsample": hp.quniform("subsample", .5, .9, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [100, 250, 500, 1024]),
        "max_depth": hp.choice("max_depth", [128, 256, 516, 1024, 2048])
    }

    X_trn, X_val, y_trn, y_val = train_test_split(trn, y, test_size=.3, random_state=seed)


    def objective(hyperparams):
        model = lgb.LGBMClassifier(**params, **hyperparams)
        model.fit(X=X_trn, y=y_trn,
                  eval_set=[(X_val, y_val)],
                  eval_metric="multi_logloss",
                  early_stopping_rounds=5,
                  verbose=False)
        score = model.best_score_["valid_0"]["multi_logloss"]

        return {'loss': score, 'status': STATUS_OK, 'model': model}


    trials = Trials()
    best = fmin(fn=objective, space=space, trials=trials,
                algo=tpe.suggest, max_evals=10, verbose=1)
    hyperparams = space_eval(space, best)
    n_best = trials.best_trial['result']['model'].best_iteration_
    params.update(hyperparams)
    print(params)
    # 하이퍼 파라미터 튜닝 끝

    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

    p_val = np.zeros((trn.shape[0], n_class))
    p_tst = np.zeros((tst.shape[0], n_class))
    for i, (i_trn, i_val) in enumerate(cv.split(trn, y), 1):
        print(f'training model for CV #{i}')
        clf = lgb.LGBMClassifier(**params)
        clf.fit(trn[i_trn], y[i_trn],
                eval_set=[(trn[i_val], y[i_val])],
                eval_metric='multiclass',
                verbose=3,
                early_stopping_rounds=20)

        p_val[i_val, :] = clf.predict_proba(trn[i_val])
        p_tst += clf.predict_proba(tst) / n_fold

        clear_session()
        gc.collect()

    print(f'Accuracy (CV): {accuracy_score(y, np.argmax(p_val, axis=1)) * 100:8.4f}%')
    print(f'Log Loss (CV): {log_loss(pd.get_dummies(y), p_val):8.4f}')

    np.savetxt(p_val_file, p_val, fmt='%.6f', delimiter=',')
    np.savetxt(p_tst_file, p_tst, fmt='%.6f', delimiter=',')

    sub = pd.read_csv(sample_file, index_col=0)
    print(sub.shape)
    print(sub.head())

    sub[sub.columns] = p_tst
    print(sub.head())

    sub.to_csv(sub_file)