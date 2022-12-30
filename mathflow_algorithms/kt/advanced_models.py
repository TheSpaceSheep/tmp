from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
import time
import numpy as np
import pandas as pd
import os
import glob
import json
import pywFM
import datetime


class DIRT:
    pass

class Das3h:
    """Das3h model.
    The Das3h model is similar to the Dash model extend along multi-dimensions. For instance, the decay
    parameters can be different along KCs and each item can call multiple KCs in Das3h"""
    
    def __init__(self, q_mat):
        self.name = "das3h"
        self.diff = None
        self.ability = None
        self.theta = None
        self.q_mat = load_npz(q_mat)
        pass
    
    def split_and_fit(self, X_file, df, d, 
            generalization="strongest", C=1.0, iter=300,
            grid_search = False, feature_grouping = False, jsonfile=None,
            users=True, items=True, skills=True, wins=True, fails=False,
            attempts=True, tw_kc=True):
        
        today = datetime.datetime.now()
        
        X = csr_matrix(load_npz(X_file))
        X = X[:,3:]
        # X = X[:270000,:] #TO REMOVE, ONLY FOT TESTING (approx 10% of samples)

        y = X[:,0].toarray().flatten()
        qmat = self.q_mat
        
        params = {
            'task': 'classification',
            'num_iter': iter,
            'rlog': True,
            'learning_method': 'mcmc',
            'k2': d
        }
        
        df = pd.read_csv(df)
        # df = df.iloc[:270000,:] #TO REMOVE, ONLY FOT TESTING
        print("Splitting data...")
        self.split_data(df, nb_folds=5)
        EXPERIMENT_FOLDER = os.path.join(f"data/{self.name}", "results")
        
        if grid_search:
            dict_of_auc = defaultdict(lambda: [])
            dict_of_rmse = defaultdict(lambda: [])
            dict_of_nll = defaultdict(lambda: [])
            dict_of_acc = defaultdict(lambda: [])
            list_of_elapsed_times = []

        # Define array of grouping variables
        if feature_grouping:
            with open(jsonfile) as json_file:
                config = json.load(json_file)
            arr_of_grouping = []
            group_id = 0
            if users:
                arr_of_grouping.extend([group_id for i in range(config["n_users"])])
                group_id += 1
            if items:
                arr_of_grouping.extend([group_id for i in range(config["n_items"])])
                group_id += 1
            if skills:
                arr_of_grouping.extend([group_id for i in range(config["n_skills"])])
                group_id += 1
            if wins:
                if tw_kc: # we group all win features together, regardless of the tw
                    arr_of_grouping.extend([group_id for i in range(5*config["n_skills"])])
                    group_id += 1
                else:
                    arr_of_grouping.extend([group_id for i in range(config["n_skills"])])
                    group_id += 1
            if fails: # to change if we allow for fails + tw
                arr_of_grouping.extend([group_id for i in range(config["n_skills"])])
                group_id += 1
            if attempts:
                if tw_kc: # we group all attempt features together, regardless of the tw
                    arr_of_grouping.extend([group_id for i in range(5*config["n_skills"])])
                    group_id += 1
                else:
                    arr_of_grouping.extend([group_id for i in range(config["n_skills"])])
                    group_id += 1
            arr_of_grouping = np.array(arr_of_grouping)
 
        for i, folds_file in tqdm(enumerate(sorted(glob.glob(f"data/{self.name}/strongest/folds/test_fold*.npy")))):
            if not os.path.isdir(os.path.join(EXPERIMENT_FOLDER, str(i))):
                os.makedirs(os.path.join(EXPERIMENT_FOLDER, str(i)))
            dt = time.time()
            test_ids = np.load(folds_file)
            train_ids = list(set(range(X.shape[0])) - set(test_ids))
            
            X_train = X[train_ids,1:]
            y_train = y[train_ids]
            X_test = X[test_ids,1:]
            y_test = y[test_ids]
            
            if grid_search:
                if d == 0:
                    for c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
                        print('fitting for c=...'.format(c))
                        estimators = [
                            ('maxabs', MaxAbsScaler()),
                            ('lr', LogisticRegression(solver="saga", max_iter=iter, C=c))
                        ]
                        pipe = Pipeline(estimators)
                        pipe.fit(X_train, y_train)
                        y_pred_test = pipe.predict_proba(X_test)[:, 1]
                        dict_of_auc[c].append(roc_auc_score(y_test, y_pred_test))
                        dict_of_rmse[c].append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
                        dict_of_nll[c].append(log_loss(y_test, y_pred_test))
                        dict_of_acc[c].append(accuracy_score(y_test, np.round(y_pred_test)))
                    list_of_elapsed_times.append(np.around(time.time() - dt,3))
                else:
                    for meta_value in ["no grouping","feature grouping"]:
                        if meta_value == "no grouping":
                            grouping = None
                        else:
                            grouping = arr_of_grouping
                        print('fitting with {}...'.format(meta_value))
                        transformer = MaxAbsScaler().fit(X_train)
                        fm = pywFM.FM(**params)
                        model = fm.run(transformer.transform(X_train), y_train,
                                    transformer.transform(X_test), y_test, meta=grouping)
                        y_pred_test = np.array(model.predictions)
                        dict_of_auc[meta_value].append(roc_auc_score(y_test, y_pred_test))
                        dict_of_rmse[meta_value].append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
                        dict_of_nll[meta_value].append(log_loss(y_test, y_pred_test))
                        dict_of_acc[meta_value].append(accuracy_score(y_test, np.round(y_pred_test)))
                    list_of_elapsed_times.append(np.around(time.time() - dt,3))
            else:
                if d == 0:
                    print('fitting...')
                    estimators = [
                        ('maxabs', MaxAbsScaler()),
                        # ('lr', SGDClassifier(loss="log", max_iter=iter))
                        ('lr', LogisticRegression(solver="saga", max_iter=iter, C=C))
                    ]
                    pipe = Pipeline(estimators)
                    pipe.fit(X_train, y_train)
                    y_pred_test = pipe.predict_proba(X_test)[:, 1]
                else:
                    if feature_grouping:
                        grouping = arr_of_grouping
                    else:
                        grouping = None
                    transformer = MaxAbsScaler().fit(X_train)
                    fm = pywFM.FM(**params)
                    model = fm.run(transformer.transform(X_train), y_train,
                                transformer.transform(X_test), y_test, meta=grouping)
                    y_pred_test = np.array(model.predictions)
                    model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(i), 'rlog.csv'))
                    
                # print(y_test)
                # print(y_pred_test)
                print("------------------------------------")
                ACC = accuracy_score(y_test, np.round(y_pred_test))
                print('acc : ', ACC)
                AUC = roc_auc_score(y_test, y_pred_test)
                print('auc : ', AUC)
                NLL = log_loss(y_test, y_pred_test)
                print('nll : ', NLL)
                RMSE = np.sqrt(mean_squared_error(y_test,y_pred_test))
                print('rmse : ', RMSE)
                print("\n")
                
                elapsed_time = np.around(time.time() - dt,3)
                # Save experimental results
                with open(os.path.join(EXPERIMENT_FOLDER, str(i), 'results.json'), 'w') as f:
                    f.write(json.dumps({
                        'date': str(today),
                        'args': self.split_and_fit.__code__.co_varnames,
                        'metrics': {
                            'ACC': ACC,
                            'AUC': AUC,
                            'NLL': NLL,
                            'RMSE': RMSE
                        },
                        'elapsed_time': elapsed_time
                    }, indent=4))

        if grid_search:
            list_of_hp = []
            list_of_mean_metrics = []
            for hp in dict_of_auc.keys():
                list_of_hp.append(hp)
                list_of_mean_metrics.append(np.mean(dict_of_auc[hp]))
            optimal_hp = list_of_hp[np.argmax(list_of_mean_metrics)]
            print("Optimal set of HP found: {}".format(optimal_hp))
            print("Overall AUC : {}".format(np.around(np.mean(dict_of_auc[optimal_hp]),3)))
            print("Overall RMSE : {}".format(np.around(np.mean(dict_of_rmse[optimal_hp]),3)))
            print("Overall NLL : {}".format(np.around(np.mean(dict_of_nll[optimal_hp]),3)))

            for i in range(len(list_of_elapsed_times)):
                with open(os.path.join(EXPERIMENT_FOLDER, str(i), 'results.json'), 'w') as f:
                    f.write(json.dumps({
                        'date': str(today),
                        'args': self.split_and_fit.__code__.co_varnames,
                        'metrics': {
                            'ACC': dict_of_acc[optimal_hp][i],
                            'AUC': dict_of_auc[optimal_hp][i],
                            'NLL': dict_of_nll[optimal_hp][i],
                            'RMSE': dict_of_rmse[optimal_hp][i]
                        },
                        'elapsed_time': list_of_elapsed_times[i],
                        'optimal_hp': optimal_hp
                    }, indent=4))
        
    def split_data(self, df, nb_folds=5):
        start = time.time()
        all_users = df["user_id"].unique()
        kfold = KFold(nb_folds, shuffle=True)
        for i, (train, test) in enumerate(kfold.split(all_users)):
            path = f"data/{self.name}/strongest/folds"
            if not os.path.isdir(path):
                os.makedirs(path)
            list_of_test_ids = []
            for user_id in all_users[test]:
                list_of_test_ids += list(df.query('user_id == {}'.format(user_id)).index)
            np.save(path+'/test_fold{}.npy'.format(i), np.array(list_of_test_ids))
        end = time.time()
        print(f"It took {abs(start-end)}s to split data")