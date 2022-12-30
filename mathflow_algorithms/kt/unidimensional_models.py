import numpy as np
from scipy.optimize import fsolve, minimize
import time
from sklearn.linear_model import LogisticRegression
from scipy.optimize import least_squares
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

class Rasch:
    """Rasch model.
    The Rasch model is a particular case of the Item Response Theory (IRT) with one parameter where the
    discrimination parameters are the same for each question (equal to one). Only works for dichotomous
    data.
    """
    
    def __init__(self):
        self.name = "rasch"
        self.diff = None
        self.ability = None
        pass
    
    def fit(self, data, verbose=0, method="prox", x0=None):
        """Fit the Rasch model.

        Args:
            data (pd.DataFrame): Dataset
            method (str, optional): method to fit the model. Theorical descriptions can be found in the 
            protected MathFlow Notion. Defaults to "prox".
        """
        N = data.shape[0]
        I = data.shape[1]
        
        if method == "prox": # Works with missing data
            # Usefull pre-calculations
            S_items = data.sum().values
            N_items = data.notnull().sum().values
            R_users = data.sum(axis=1).values
            N_users = data.notnull().sum(axis=1).values
            # Gaussian parameters
            mu_items = np.zeros(I)
            sigma_items = np.zeros(I)
            mu_users = np.zeros(N)
            sigma_users = np.zeros(N)
            # Trajectories
            all_items_difficulties = []
            all_users_abilities = []
            condition = True
            iter=0
            itermax = 1e4
            start = time.time()
            while condition and iter < itermax:
                # Step 1 : Compute items_difficulties and substract by the mean so that mu_users = 0 at all times
                items_difficulties = mu_items - np.sqrt(1 + (sigma_items**2)/2.9) * np.log(S_items/(N_items-S_items))
                items_difficulties = items_difficulties - np.mean(items_difficulties)
                # Step 2 : Compute the new mu_users (normally null) and sigma_users
                for n in range(mu_users.shape[0]):
                    mu_users[n] = sum([int(data.iloc[n,i] != np.nan)*items_difficulties[i] for i in range(I)])/N_users[n]
                    sigma_users[n] = np.sqrt(sum([int(data.iloc[n][i] != np.nan)*items_difficulties[i]**2 for i in range(I)])/(N_users[n]-1))
                # Step 3 : Compute users_abilities
                users_abilities = mu_users + np.sqrt(1 + (sigma_users**2)/2.9) * np.log(R_users/(N_users-R_users))
                # Step 4 : Compute the new mu_items and sigma_items
                for i in range(mu_items.shape[0]):
                    mu_items[i] = sum([int(data.iloc[n,i] != np.nan)*users_abilities[n] for n in range(N)])/N_items[i]
                    sigma_items[i] = np.sqrt(sum([int(data.iloc[n][i] != np.nan)*users_abilities[n]**2 for n in range(N)])/(N_items[i]-1))
                # Step 5 : Check that the variation of each variable is great enough
                if len(all_items_difficulties) > 0 and len(all_users_abilities) > 0 :
                    if verbose:
                        print(f"Vectorial distance (items) with precedent : {np.linalg.norm(all_items_difficulties[-1] - items_difficulties)}")
                        print(f"Vectorial distance (users) with precedent : {np.linalg.norm(all_users_abilities[-1] - users_abilities)}")
                    condition = np.linalg.norm(all_items_difficulties[-1] - items_difficulties) > 1e-5 and np.linalg.norm(all_users_abilities[-1] - users_abilities) > 1e-5
                all_items_difficulties.append(items_difficulties)
                all_users_abilities.append(users_abilities)
                iter+=1
            global_std = np.sqrt((N_items*(1 + (sigma_items**2)/2.9))/(S_items*(N_items-S_items)))
            end = time.time()
            if verbose:
                print("-------------------------------")
                print("The Rasch model converged in {} iterations : {:.2f}s".format(iter, abs(start-end)))
                print(f"Difficulty parameters : {all_items_difficulties[-1]}")
                print(f"The standard deviation for this estimation is : {global_std}")
                print("-------------------------------")
            self.diff = all_items_difficulties[-1]
            self.ability = all_users_abilities[-1]
            
        elif method == "gls": # Datum-by-datum subsection gaussian least squares
            # Works with missing data
            # Very long ! (3h40 on 500x20 matrix)
            def F(x):
                return np.array([(data.iloc[n,i] - np.exp(x[n+I]-x[i])/(1+np.exp(x[n+I]-x[i]))) 
                                 for n in range(N) for i in range(I)])
            if x0 != None:
                init = x0
            else : init = np.zeros(N+I)
            start = time.time()
            res = least_squares(F, init)
            res = res.x[:I]-mean(res.x[:I])
            end = time.time()
            if verbose:
                print("The Rasch model converged in {:.2f}s".format(abs(start-end)))
                print(f"Difficulty parameters : {res}")
            self.diff = res
            
        elif method == "gls2":
            def proba(diff, ability):
                return np.exp(ability-diff)/(1+np.exp(ability-diff))
            def func(x):
                difficulty_params = [sum([(2*(data.iloc[n,i] - proba(x[n+I],x[i]))*
                                          (proba(x[n+I],x[i])*(1-proba(x[n+I],x[i]))))
                                          for n in range(N)]) for i in range(I)]
                ability_params = [sum([(2*(proba(x[n+I],x[i]) - data.iloc[n,i])*
                                          (proba(x[n+I],x[i])*(1-proba(x[n+I],x[i]))))
                                          for i in range(I)]) for n in range(N)]
                return difficulty_params + ability_params
            if x0 is not None:
                init = x0
            else : init = np.zeros(N+I)
            start = time.time()
            root = fsolve(func, init)
            res = root[:I] - np.mean(root[:I])
            end = time.time()
            if verbose:
                print("The Rasch model converged in {:.2f}s".format(abs(start-end)))
                print(f"Difficulty parameters : {res}")
            self.diff = res
            self.ability = root[I:]
        
        elif method == "gls3":
            def proba(diff, ability):
                return np.exp(ability-diff)/(1+np.exp(ability-diff))
            def objective(x):
                return sum([(data.iloc[n,i]-proba(x[n+I],x[i]))**2 for i in range(I) for n in range(N)])
            def constraint1(x):
                return sum(x[:I])
            def constraint2(x):
                return sum(x[I:])
            start = time.time()
            if x0 is not None:
                init = x0
            else : init = np.zeros(N+I)
            con1 = {'type': 'eq', 'fun': constraint1}
            con2 = {'type': 'eq', 'fun': constraint2}
            cons = ([con1,con2])
            sol = minimize(objective, x0=x0, constraints=cons)
            end = time.time()
            if verbose :
                print("The Rasch model converged in {:.2f}s".format(abs(start-end)))
                print(f"Difficulty parameters : {res}")
            self.diff = sol.x[:I]
            self.ability = sol.x[I:]
            
        elif method == "min-chi-squared": # Datum-by-datum subsection minimum chi-squared
            # Works with missing data
            def proba(diff, ability):
                return np.exp(ability-diff)/(1+np.exp(ability-diff))
            def F(x):
                return np.array([((1/np.sqrt(proba(x[n+I],x[i])*(1-proba(x[n+I],x[i]))))*
                                 (data.iloc[n,i] - proba(x[n+I],x[i])))
                                 for n in range(N) for i in range(I)])
            init = np.zeros(N+I)
            start = time.time()
            res = least_squares(F, init)
            res = res.x[:I]-mean(res.x[:I])
            end = time.time()
            if verbose:
                print("The Rasch model converged in {:.2f}s".format(abs(start-end)))
                print(f"Difficulty parameters : {res}")
            self.diff = res
        
        elif method == "pair": # Datum-by-datum subsection pairwise
            pass
        
        elif method == "cmle": #
            pass
        
        elif method == "jmle":
            
            def func(x):
                sol = []
                for i in range(I): # Items
                    sol.append(sum(np.exp(x[n]-x[i])/(1+np.exp(x[n]-x[i])) for n in range(I,I+N)) - 
                    data[data.columns[i]].sum())
                for n in range(I,I+N): # Users
                    sol.append(sum(np.exp(x[n]-x[i])/(1+np.exp(x[n]-x[i])) for i in range(I)) - 
                    data.iloc[n-I].sum())
                return sol
            
            root = fsolve(func, np.array([0.5 for _ in range(N+I)]))
            if self.verbose:
                print("Discrimination parameters : ", root[:I])
            self.diff = root[:I]
        
        elif method == "log-linear":
            pass
        
        elif method == "mmle":
            pass
        
        elif method == "i2at":
            pass
    
    def test(self, data):
        df_valid_train, df_valid_test, diff_train, diff_test = train_test_split(data.T, self.diff, test_size=0.2, shuffle=True)
        df_valid_train = df_valid_train.T
        idxs = []
        for i, row in df_valid_train.iterrows():
            if row.nunique()==1:
                idxs.append(i)
                df_valid_train.drop(i, inplace=True)
        df_valid_test = df_valid_test.T
        for i in idxs:
            df_valid_test.drop(i, inplace=True)
            
        def estimate_ability(diff, results):
            assert type(diff) != type(None), "Please fit the model first using model.fit"
            clf = LogisticRegression(solver='newton-cg', C=1, fit_intercept=True).fit(diff.reshape(-1,1), results)
            theta = clf.intercept_[0]
            return theta
        
        def estimate_answers(diff, ability):
            probas = []
            for i in range(diff.shape[0]):
                probas.append(np.exp(ability-diff[i])/(1+np.exp(ability-diff[i])))
            return probas
        
        all_thetas = [estimate_ability(diff_train, df_valid_train.iloc[i,:].values) for i in range(df_valid_train.shape[0])]
        y_true, y_pred = [],[]
        for n in range(df_valid_test.shape[0]):
            for i in range(df_valid_test.shape[1]):
                y_true.append(df_valid_test.iloc[n,i])
            y_pred += estimate_answers(diff_test,all_thetas[n])
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
        clf = LogisticRegression(solver='newton-cg', C=1, fit_intercept=True).fit(self.diff.reshape(1,-1), results)
        ability = clf.intercept_[0]
        return ability
    
    def estimate_answers(self, ability):
        """Estimate the answers of a student given the ability, difficulty and facility parameters
        Args:
            ability (int): Student's global ability

        Returns:
            probas (list): Probabilities of success on each item
        """
        probas = []
        for i in range(self.diff.shape[0]):
            probas.append(np.exp(ability-self.diff[i])/(1+np.exp(ability-self.diff[i])))
        return probas

    def get_diff(self):
        return self.diff
    
    def get_ability(self):
        return self.ability
    
    def get_name(self):
        return self.name






class TwoParamsIRT:
    pass






class ThreeParamsIRT:
    pass
