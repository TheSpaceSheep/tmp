import numpy as np
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

class MIRT:
    def __init__(self, dim):
        self.name = "mirt"
        self.dim = dim
        self.diff = None
        self.facility = None
        self.ability = None
    
    def fit(self, data, verbose=0, method="mhrm", nb_iterations=300):
        """Fit the MIRT model.

        Args:
            data (pd.DataFrame): Dataset
            method (str, optional): method to fit the model. Theorical descriptions can be found in the 
            protected MathFlow Notion. Defaults to "mhrm".
        """
        N = data.shape[0]
        I = data.shape[1]
        nb_samples = 50
        nb_burned = 5
        
        if method == "mhrm":
            def add_bias(X):
                N, _ = X.shape
                return np.column_stack((X, np.ones(N)))
                
            def expo(Th, X, sgn=1):
                X1 = add_bias(X)
                return np.exp(sgn * X1.dot(Th.T))
            
            def loglikelihood(Th, X, Y):
                exp_XT = expo(Th, X)
                ll = ((Y - 1) * np.log(1 + exp_XT)
                        + Y  * (np.log(exp_XT) - np.log(1 + exp_XT)))
                return ll
            
            def score(Th, X, Y):
                exp_XT = expo(Th, X)
                X1 = add_bias(X)
                C = (Y + (Y - 1) * exp_XT) / (1 + exp_XT)
                return C.T.dot(X1)

            def hessian(Th, X):
                exp_XT = expo(Th, X)
                X1 = add_bias(X)
                XiXiT = np.einsum('ij,ki->ijk', X1, X1.T)
                Lambda = -exp_XT / (1 + exp_XT) ** 2
                return np.tensordot(Lambda.T, XiXiT, axes=1)
            
            def proba1(Th, X):
                exp_minusXT = expo(Th, X, -1)
                return 1 / (1 + exp_minusXT)
            
            def phi(dim, x):
                return multivariate_normal.pdf(x, mean=np.zeros(dim), cov=np.eye(dim))

            def logacceptance(dim, Th, X):
                # print('hop', loglikelihood(Th, X).sum(axis=1).shape, np.log(phi(X)).shape)
                return loglikelihood(Th, X, data).sum(axis=1) + np.log(phi(dim, X))

            def impute(dim, Th):
                def iterate(X):
                    old = logacceptance(dim, Th, X)
                    E = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=N)
                    new = logacceptance(dim, Th, X + E)
                    sample = np.random.random(N)
                    X += np.diag(np.log(sample) < new - old).dot(E)
                    return X
                Xs = [np.zeros((N, dim))]
                for _ in range(nb_samples):
                    Xs.append(iterate(Xs[-1]))
                return Xs[nb_burned:]
            
            def compute_all_errors(dim, Th):
                Xs = impute(dim, Th)
                X = sum(X for X in Xs) / len(Xs)
                p = proba1(Th, X)
                print('Train RMSE:', ((p - data) ** 2).mean() ** 0.5)
                print('Train NLL:', -np.log(1 - abs(p - data)).mean())
                print('Train accuracy:', (np.round(p) == data).mean())

            G = np.zeros((I, self.dim + 1, self.dim + 1))
            Th = np.random.random((I, self.dim + 1))
            for k in tqdm(range(1, nb_iterations)):
                # STEP 1: Imputation
                Xs = impute(self.dim, Th)
                ll = np.mean([loglikelihood(Th, X, data).sum() for X in Xs])
                # STEP 2a: Approximation of score
                s = sum([score(Th, X, data) for X in Xs]) / len(Xs)
                # STEP 2b: Approximation of hessian
                H = -sum([hessian(Th, X) for X in Xs]) / len(Xs)
                gamma = 1 / k
                G += gamma * (H - G)
                Ginv = np.stack(np.linalg.inv(G[i, :, :]) for i in range(I))
                Th += gamma * np.einsum('ijk,ik->ij', Ginv, s)
            if verbose : 
                compute_all_errors(self.dim, Th)
            self.diff = Th[:,:self.dim]
            self.facility = Th[:, self.dim]
    
    def test(self, data):
        df_valid_train, df_valid_test, diff_train, diff_test, facility_train, facility_test = train_test_split(data.T, self.diff, self.facility, test_size=0.2, shuffle=True)
        df_valid_train = df_valid_train.T
        idxs = []
        for i, row in df_valid_train.iterrows():
            if row.nunique()==1:
                idxs.append(i)
                df_valid_train.drop(i, inplace=True)
        df_valid_test = df_valid_test.T
        for i in idxs:
            df_valid_test.drop(i, inplace=True)
            
        def estimate_ability(diff, facility, results):
            assert type(self.diff) != type(None), "Please fit the model first using model.fit"
            clf = LogisticRegression(solver='newton-cg', C=1, intercept_scaling=facility).fit(diff, results)
            ability = clf.coef_[0]
            return ability
        
        def estimate_answers(diff, facility, ability):
            probas = []
            for i in range(facility.shape[0]):
                probas.append(np.exp(sum([ability[k]*diff[i,k] for k in range(len(ability))])+facility[i])/(1+np.exp(sum([ability[k]*diff[i,k] for k in range(len(ability))])+facility[i])))
            return probas
        
        all_thetas = [estimate_ability(diff_train, facility_train, df_valid_train.iloc[i,:].values) for i in range(df_valid_train.shape[0])]
        y_true, y_pred = [],[]
        for n in range(df_valid_test.shape[0]):
            for i in range(df_valid_test.shape[1]):
                y_true.append(df_valid_test.iloc[n,i])
            y_pred += estimate_answers(diff_test, facility_test, all_thetas[n])
            y_pred = [round(item) for item in y_pred]
        logloss = log_loss(y_true, y_pred)
        y_pred = list(np.around(np.array(y_pred)))
        return logloss, accuracy_score(y_true, y_pred)
    
    def estimate_ability(self, results):
        """Estimate the ability of a student given his/her results and the difficulty parameters
        Args:
            results (np.ndarray): Dichotomic results of a student 

        Returns:
            theta (int): Global estimated ability of the student
        """
        assert type(self.diff) != type(None), "Please fit the model first using model.fit"
        clf = LogisticRegression(solver='newton-cg', C=1, intercept_scaling=self.facility).fit(self.diff, results)
        ability = clf.coef_[0]
        return ability
    
    def estimate_answers(self, ability):
        """Estimate the answers of a student given the ability, difficulty and facility parameters
        Args:
            ability (int): Student's global ability

        Returns:
            probas (list): Probabilities of success on each item
        """
        probas = []
        for i in range(self.facility.shape[0]):
            probas.append(np.exp(sum([ability[k]*self.diff[i,k] for k in range(len(ability))])+self.facility[i])/(1+np.exp(sum([ability[k]*self.diff[i,k] for k in range(len(ability))])+self.facility[i])))
        return probas
    
    def get_diff(self):
        return self.diff
    
    def get_ability(self):
        return self.ability
    
    def get_name(self):
        return self.name






class GenMA:
    def __init__(self, dim, q_matrix):
        self.name = "genma"
        self.dim = dim
        self.diff = None
        self.facility = None
        self.q_matrix = q_matrix
        self.ability = None
    
    def fit(self, data, verbose=0, method="mhrm", nb_iterations=300):
        """Fit the MIRT model.

        Args:
            data (pd.DataFrame): Dataset
            method (str, optional): method to fit the model. Theorical descriptions can be found in the 
            protected MathFlow Notion. Defaults to "mhrm".
        """
        N = data.shape[0]
        I = data.shape[1]
        nb_iterations = 100
        nb_samples = 50
        nb_burned = 5
        
        if method == "mhrm":
            def add_bias(X):
                N, _ = X.shape
                return np.column_stack((X, np.ones(N)))
                
            def expo(Th, X, sgn=1):
                X1 = add_bias(X)
                return np.exp(sgn * X1.dot(Th.T))
            
            def loglikelihood(Th, X, Y):
                exp_XT = expo(Th, X)
                ll = ((Y - 1) * np.log(1 + exp_XT)
                        + Y  * (np.log(exp_XT) - np.log(1 + exp_XT)))
                return ll
            
            def score(Th, X, Y):
                exp_XT = expo(Th, X)
                X1 = add_bias(X)
                C = (Y + (Y - 1) * exp_XT) / (1 + exp_XT)
                return C.T.dot(X1)

            def hessian(Th, X):
                exp_XT = expo(Th, X)
                X1 = add_bias(X)
                XiXiT = np.einsum('ij,ki->ijk', X1, X1.T)
                Lambda = -exp_XT / (1 + exp_XT) ** 2
                return np.tensordot(Lambda.T, XiXiT, axes=1)
            
            def proba1(Th, X):
                exp_minusXT = expo(Th, X, -1)
                return 1 / (1 + exp_minusXT)
            
            def phi(dim, x):
                return multivariate_normal.pdf(x, mean=np.zeros(dim), cov=np.eye(dim))

            def logacceptance(dim, Th, X):
                # print('hop', loglikelihood(Th, X).sum(axis=1).shape, np.log(phi(X)).shape)
                return loglikelihood(Th, X, data).sum(axis=1) + np.log(phi(dim, X))

            def impute(dim, Th):
                def iterate(X):
                    old = logacceptance(dim, Th, X)
                    E = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=N)
                    new = logacceptance(dim, Th, X + E)
                    sample = np.random.random(N)
                    X += np.diag(np.log(sample) < new - old).dot(E)
                    return X
                Xs = [np.zeros((N, dim))]
                for _ in range(nb_samples):
                    Xs.append(iterate(Xs[-1]))
                return Xs[nb_burned:]
            
            def compute_all_errors(dim, Th):
                Xs = impute(dim, Th)
                X = sum(X for X in Xs) / len(Xs)
                p = proba1(Th, X)
                print('Train RMSE:', ((p - data) ** 2).mean() ** 0.5)
                print('Train NLL:', -np.log(1 - abs(p - data)).mean())
                print('Train accuracy:', (np.round(p) == data).mean())

            G = np.zeros((I, self.dim + 1, self.dim + 1))
            Th = np.random.random((I, self.dim + 1))
            for k in tqdm(range(1, nb_iterations)):
                # STEP 1: Imputation
                Xs = impute(self.dim, Th)
                ll = np.mean([loglikelihood(Th, X, data).sum() for X in Xs])
                # STEP 2a: Approximation of score
                s = sum([score(Th, X, data) for X in Xs]) / len(Xs)
                # STEP 2b: Approximation of hessian
                H = -sum([hessian(Th, X) for X in Xs]) / len(Xs)
                gamma = 1 / k
                G += gamma * (H - G)
                Ginv = np.stack(np.linalg.inv(G[i, :, :]) for i in range(I))
                Th += gamma * np.einsum('ijk,ik->ij', Ginv, s)
            if verbose : 
                compute_all_errors(self.dim, Th)
            self.diff = Th[:,:self.dim]
            self.facility = Th[:, self.dim]
    
    def test(self, data):
        df_valid_train, df_valid_test, diff_train, diff_test, facility_train, facility_test, q_matrix_train, q_matrix_test = train_test_split(data.T, self.diff, self.facility, self.q_matrix, test_size=0.2, shuffle=True)
        df_valid_train = df_valid_train.T
        idxs = []
        for i, row in df_valid_train.iterrows():
            if row.nunique()==1:
                idxs.append(i)
                df_valid_train.drop(i, inplace=True)
        df_valid_test = df_valid_test.T
        for i in idxs:
            df_valid_test.drop(i, inplace=True)
            
        def estimate_ability(diff, facility, results):
            assert type(self.diff) != type(None), "Please fit the model first using model.fit"
            clf = LogisticRegression(solver='newton-cg', C=1, intercept_scaling=facility).fit(diff, results)
            ability = clf.coef_[0]
            return ability
        
        def estimate_answers(diff, q_matrix, facility, ability):
            probas = []
            for i in range(facility.shape[0]):
                probas.append(np.exp(sum([ability[k]*diff[i,k]*q_matrix[i,k] for k in range(len(ability))])+facility[i])/(1+np.exp(sum([ability[k]*diff[i,k]*q_matrix[i,k] for k in range(len(ability))])+facility[i])))
            return probas
        
        all_thetas = [estimate_ability(diff_train, facility_train, df_valid_train.iloc[i,:].values) for i in range(df_valid_train.shape[0])]
        y_true, y_pred = [],[]
        for n in range(df_valid_test.shape[0]):
            for i in range(df_valid_test.shape[1]):
                y_true.append(df_valid_test.iloc[n,i])
            y_pred += estimate_answers(diff_test, q_matrix_test, facility_test, all_thetas[n])
            y_pred = [round(item) for item in y_pred]
        logloss = log_loss(y_true, y_pred)
        y_pred = list(np.around(np.array(y_pred)))
        return logloss, accuracy_score(y_true, y_pred)
    
    def estimate_ability(self, results):
        """Estimate the ability of a student given his/her results and the difficulty parameters
        Args:
            results (np.ndarray): Dichotomic results of a student 

        Returns:
            theta (int): Global estimated ability of the student
        """
        assert type(self.diff) != type(None), "Please fit the model first using model.fit"
        clf = LogisticRegression(solver='newton-cg', C=1, intercept_scaling=self.facility).fit(self.diff, results)
        ability = clf.coef_[0]
        return ability
    
    def estimate_answers(self, ability):
        """Estimate the answers of a student given the ability, difficulty and facility parameters
        Args:
            ability (int): Student's global ability

        Returns:
            probas (list): Probabilities of success on each item
        """
        probas = []
        for i in range(self.facility.shape[0]):
            probas.append(np.exp(sum([ability[k]*self.diff[i,k]*self.q_matrix[i,k] for k in range(len(ability))])+self.facility[i])/(1+np.exp(sum([ability[k]*self.diff[i,k]*self.q_matrix[i,k] for k in range(len(ability))])+self.facility[i])))
        return probas

    def get_diff(self):
        return self.diff
    
    def get_ability(self):
        return self.ability
    
    def get_name(self):
        return self.name
