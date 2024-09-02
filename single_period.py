# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
import json
import random 

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from dimod import Integer, Binary
from dimod import quicksum 
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler, LeapHybridCQMSampler 

import yfinance as yf


class SinglePeriod: 
    """Define and solve a  single-period portfolio optimization problem.
    """
    def __init__(self, cryptos=('BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'TRX', 'ADA', 'SHIB', 'LINK', 'DOT'), budget=1000, 
                 bin_size=None, gamma=None, file_path='data/crypto_basic_data.csv', 
                 dates=None, model_type='CQM', alpha=0.005, baseline='^GSPC', 
                 sampler_args=None, t_cost=0.01, verbose=True):
        """Class constructor. 

        Args:
            cryptos (list of str): List of cryptos.
            budget (int): Portfolio budget.
            bin_size (int): Maximum number of intervals for each crypto. 
            gamma (float or int or list or tuple): Budget constraint penalty coefficient(s).
                If gamma is a tuple/list and model is DQM, grid search will be done; 
                otherwise, no grid search.   
            file_path (str): Full path of CSV file containing crypto data. 
            dates (list of str): Pair of strings for start date and end date. 
            model_type (str): CQM or DQM.
            alpha (float or int or list or tuple): Risk aversion coefficient. 
                If alpha is a tuple/list and model is DQM, grid search will be done; 
                otherwise, no grid search.   
            baseline (str): Crypto baseline for rebalancing model. 
            sampler_args (dict): Sampler arguments. 
            t_cost (float): transaction cost; percentage of transaction dollar value. 
            verbose (bool): Flag to enable additional output. 
        """
        self.cryptos = list(cryptos) 
        self.budget = budget 
        self.init_budget = budget 
        self.gamma_list = []
        self.file_path = file_path
        self.dates = dates 
        self.model_type = model_type
        self.alpha_list = []
        self.baseline = [baseline] 
        self.verbose = verbose 
        self.t_cost = t_cost
        self.init_holdings = {s:0 for s in self.cryptos}
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = alpha[0]
            self.alpha_list = list(alpha)
        elif isinstance (alpha, (int, float)):
            self.alpha = alpha
        else:
            raise TypeError("Use integer or float for alpha, or a list or tuple of either type.")
        
        if gamma:
            if isinstance(gamma, (list, tuple)):
                self.gamma = gamma[-1]
                self.gamma_list = list(gamma)
            elif isinstance(gamma, (int, float)):
                self.gamma = gamma 
            else:
                raise TypeError("Use integer or float for gamma, or a list or tuple of either type.")
        else: 
            self.gamma = 100

        if bin_size:
            self.bin_size = bin_size
        else:
            self.bin_size = 10
       
        self.model = {'CQM': None, 'DQM': None}

        self.sample_set = {}
        if sampler_args:
            self.sampler_args = json.loads(sampler_args) 
        else:
            self.sampler_args = {}

        self.sampler = {'CQM': LeapHybridCQMSampler(**self.sampler_args),
                        'DQM': LeapHybridDQMSampler(**self.sampler_args)}

        self.solution = {}

        self.precision = 2
       
    def load_data(self, file_path='', dates=None, df=None, num=0):
        """Load the relevant crypto data from file, dataframe, or Yahoo!. 

        Args:
            file_path (string): Full path of csv file containing crypto price data
                for the single period problem.
            dates (list): [Start_Date, End_Date] to query data.
            df (dataframe): Table of crypto prices.   
            num (int): Number of cryptos to be randomnly generated. 
        """
        if df is not None:
            print("\nLoading data from DataFrame...")
            self.df = df 
            self.cryptos = df.columns.tolist()
        elif dates or self.dates: 
            if dates:
                self.dates = dates 
            """
            print(f"\nLoading live data from the web from Yahoo! finance",
                  f"from {self.dates[0]} to {self.dates[1]}...")

            # Generating randomn list of cryptos 
            if num > 0: 
                if (self.dates[0] < '2017-01-01'):
                    raise Exception(f"Start date must be >= '2017-01-01' " 
                                    f"when using option 'num'.") 
                symbols_df = pd.read_csv('data/crypto_symbols.csv')
                self.cryptos = random.sample(list(symbols_df.loc[:,'Symbol']), num)

            # Read in daily data; resample to monthly
            panel_data = yf.download(self.cryptos, start=self.dates[0], end=self.dates[1])
            
            # Check if panel_data is a Series or DataFrame
            if isinstance(panel_data, pd.Series):
                panel_data = panel_data.to_frame()
                panel_data.columns = ['Adj Close']
            
            # Resample the data to monthly using 'ME' and forward fill
            panel_data = panel_data.resample('ME').last()
            
            # Initialize DataFrame with the same index as panel_data and columns for each crypto
            self.df_all = pd.DataFrame(index=panel_data.index, columns=self.cryptos)
            
            # Populate self.df_all with the 'Adj Close' prices
            for i in self.cryptos:
                if 'Adj Close' in panel_data.columns:
                    if isinstance(panel_data['Adj Close'], pd.Series):
                        self.df_all[i] = panel_data['Adj Close']
                    elif i in panel_data['Adj Close'].columns:
                        self.df_all[i] = panel_data['Adj Close'][i]
                else:
                    print(f"Warning: 'Adj Close' data not found for {i}")
                       
            # Identify dates with NaNs
            for column in self.df_all.columns:
                nan_dates = self.df_all[self.df_all[column].isna()].index
                if len(nan_dates) > 0:
                    print(f"NaN dates for {column}: {nan_dates}")
                        
            # Check for columns with NaN values and drop them if necessary
            nan_columns = self.df_all.columns[self.df_all.isna().any()].tolist()
            if nan_columns:
                print("The following tickers are dropped due to invalid data: ", nan_columns)
                self.df_all = self.df_all.dropna(axis=1)
                if len(self.df_all.columns) < 2:
                    raise Exception("There must be at least 2 valid crypto tickers.")
                self.cryptos = list(self.df_all.columns)
            
            # Read in baseline data; resample to monthly
            index_df = yf.download(self.baseline, start=self.dates[0], end=self.dates[1])
            index_df = index_df.resample('ME').last()
            
            # Initialize DataFrame for baseline data
            self.df_baseline = pd.DataFrame(index=index_df.index)
            
            # Populate self.df_baseline with the 'Adj Close' prices
            for i in self.baseline:
                if 'Adj Close' in index_df.columns:
                    self.df_baseline[i] = index_df['Adj Close']
                else:
                    print(f"Warning: 'Adj Close' data not found for {i}")
            
            # Assign self.df_all to self.df
            self.df = self.df_all
            """
        else:
            print("\nLoading data from provided CSV file...")
            if file_path:
                self.file_path = file_path

            self.df = pd.read_csv(self.file_path, index_col=0)

        self.init_holdings = {s:0 for s in self.cryptos}

        self.max_num_shares = (self.budget/self.df.iloc[-1]).astype(int)
        if self.verbose:
            print("\nMax shares we can afford with a budget of", self.budget)
            print(self.max_num_shares.to_string())

        self.shares_intervals = {}
        for crypto in self.cryptos:
            if self.max_num_shares[crypto]+1 <= self.bin_size:
                self.shares_intervals[crypto] = list(range(self.max_num_shares[crypto] + 1))
            else:
                span = (self.max_num_shares[crypto]+1) / self.bin_size
                self.shares_intervals[crypto] = [int(i*span) 
                                        for i in range(self.bin_size)]

        self.price = self.df.iloc[-1]
        self.monthly_returns = self.df[list(self.cryptos)].pct_change().iloc[1:]
        self.avg_monthly_returns = self.monthly_returns.mean(axis=0)
        self.covariance_matrix = covariance_matrix = self.monthly_returns.cov()

        # convert any NaNs in the covariance matrix to 0s
        covariance_matrix.replace(np.nan, 0)

    def build_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Build and store a CQM. 
        This method allows the user a choice of 3 problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state. 
        """
        # Instantiating the CQM object 
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model 
        x = {s: Integer("%s" %s, lower_bound=0, 
                        upper_bound=self.max_num_shares[s]) for s in self.cryptos}

        # Defining risk expression 
        risk = 0
        for s1, s2 in product(self.cryptos, self.cryptos):
            coeff = (self.covariance_matrix[s1][s2] * self.price[s1] * self.price[s2])
            risk = risk + coeff*x[s1]*x[s2]

        # Defining the returns expression 
        returns = 0
        for s in self.cryptos: 
            returns = returns + self.price[s] * self.avg_monthly_returns[s] * x[s]

        # Adding budget and related constraints
        if not init_holdings:
            init_holdings = self.init_holdings
        else:
            self.init_holdings = init_holdings

        if not self.t_cost:  
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.cryptos])
                            <= self.budget, label='upper_budget')
            cqm.add_constraint(quicksum([x[s]*self.price[s] for s in self.cryptos])
                            >= 0.997*self.budget, label='lower_budget')
        else:
            # Modeling transaction cost 
            x0 = init_holdings

            y = {s: Binary("Y[%s]" %s) for s in self.cryptos}

            lhs = 0 
            for s in self.cryptos:
                lhs = lhs + 2*self.t_cost*self.price[s]*x[s]*y[s] \
                          + self.price[s]*(1 - self.t_cost)*x[s] \
                          - 2*self.t_cost*self.price[s]*x0[s]*y[s] \
                          - self.price[s]*(1 - self.t_cost)*x0[s]
                          
            cqm.add_constraint( lhs <= self.budget, label='upper_budget')
            cqm.add_constraint( lhs >= self.budget - 0.003*self.init_budget, 
                                label='lower_budget')

            # indicator constraints 
            for s in self.cryptos:
                cqm.add_constraint(x[s] - x0[s]*y[s] >= 0, 
                                   label=f'indicator_constraint_gte_{s}')
                cqm.add_constraint(x[s] - x[s]*y[s] <= x0[s], 
                                   label=f'indicator_constraint_lte_{s}')

        if max_risk: 
            # Adding maximum risk constraint 
            cqm.add_constraint(risk <= max_risk, label='max_risk')

            # Objective: maximize return 
            cqm.set_objective(-1*returns)
        elif min_return:
            # Adding minimum returns constraint
            cqm.add_constraint(returns >= min_return, label='min_return') 

            # Objective: minimize risk 
            cqm.set_objective(risk)
        else: 
            # Objective: minimize mean-variance expression 
            cqm.set_objective(self.alpha*risk - returns)

        cqm.substitute_self_loops()

        self.model['CQM'] = cqm 

    def solve_cqm(self, max_risk=None, min_return=None, init_holdings=None):
        """Solve CQM.  
        This method allows the user to solve one of 3 cqm problem formulations: 
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk 
            3) min risk s.t. return >= min_return  

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            init_holdings (float): Initial holdings, or initial portfolio state. 

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format 
                e.g., solution = {'cryptos': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        self.build_cqm(max_risk, min_return, init_holdings)

        self.sample_set['CQM'] = self.sampler['CQM'].sample_cqm(self.model['CQM'], 
                                                                label="Example - Portfolio Optimization")
        n_samples = len(self.sample_set['CQM'].record)

        feasible_samples = self.sample_set['CQM'].filter(lambda d: d.is_feasible) 

        if not feasible_samples: 
            raise Exception("No feasible solution could be found for this problem instance.")
        else:
            best_feasible = feasible_samples.first

            solution = {}

            solution['cryptos'] = {k:int(best_feasible.sample[k]) for k in self.cryptos}

            solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['cryptos'])

            spending = sum([self.price[s]*max(0, solution['cryptos'][s] - self.init_holdings[s]) for s in self.cryptos])
            sales = sum([self.price[s]*max(0, self.init_holdings[s] - solution['cryptos'][s]) for s in self.cryptos])

            transaction = self.t_cost*(spending + sales)

            if self.verbose:
                print(f'Number of feasible solutions: {len(feasible_samples)} out of {n_samples} sampled.')
                print(f'\nBest energy: {self.sample_set["CQM"].first.energy: .2f}')
                print(f'Best energy (feasible): {best_feasible.energy: .2f}')  

            print(f'\nBest feasible solution:')
            print("\n".join("{}\t{:>3}".format(k, v) for k, v in solution['cryptos'].items())) 

            print(f"\nEstimated Returns: {solution['return']}")

            print(f"Sales Revenue: {sales:.2f}")

            print(f"Purchase Cost: {spending:.2f}")

            print(f"Transaction Cost: {transaction:.2f}")

            print(f"Variance: {solution['risk']}\n")

            return solution 

    def build_dqm(self, alpha=None, gamma=None):
        """Build DQM.  

        Args:
            alpha (float): Risk aversion coefficient.
            gamma (int): Penalty coefficient for budgeting constraint.
        """
        if gamma:
            self.gamma = gamma

        if alpha:
            self.alpha = alpha

         # Defining DQM 
        dqm = DiscreteQuadraticModel() 

        # Build the DQM starting by adding variables
        for s in self.cryptos:
            dqm.add_variable(len(self.shares_intervals[s]), label=s)

        # Objective 1: minimize variance
        for s1, s2 in product(self.cryptos, self.cryptos):
            coeff = (self.covariance_matrix[s1][s2]
                        * self.price[s1] * self.price[s2])
            if s1 == s2:
                for k in range(dqm.num_cases(s1)):
                    num_s1 = self.shares_intervals[s1][k]
                    dqm.set_linear_case(
                                    s1, k, 
                                    dqm.get_linear_case(s1,k) 
                                    + self.alpha*coeff*num_s1*num_s1)
            else:
                for k in range(dqm.num_cases(s1)):
                    for m in range(dqm.num_cases(s2)):
                        num_s1 = self.shares_intervals[s1][k]
                        num_s2 = self.shares_intervals[s2][m]

                        dqm.set_quadratic_case(
                                s1, k, s2, m, 
                                dqm.get_quadratic_case(s1,k,s2,m)
                                + coeff*self.alpha*num_s1*num_s2) 
                        
        # Objective 2: maximize return
        for s in self.cryptos:
            for j in range(dqm.num_cases(s)):
                dqm.set_linear_case(
                                s, j, dqm.get_linear_case(s,j)
                                - self.shares_intervals[s][j]*self.price[s]
                                * self.avg_monthly_returns[s])

        # Scaling factor to guarantee that all coefficients are integral
        # needed in order to use add_linear_inequality_constraint method 
        factor = 10**self.precision

        min_budget = round(factor*0.997*self.budget)
        budget = int(self.budget)

        terms = [(s, j, int(self.shares_intervals[s][j]
                *factor*self.price[s])) 
                for s in self.cryptos 
                for j in range(dqm.num_cases(s))]

        dqm.add_linear_inequality_constraint(terms, 
                                            constant=0, 
                                            lb=min_budget, 
                                            ub=factor*budget, 
                                            lagrange_multiplier=self.gamma, 
                                            label="budget")

        self.model['DQM'] = dqm 

    def solve_dqm(self):
        """Solve DQM.

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format 
                e.g., solution = {'cryptos': {'BTC': 3, 'ETH': 12}, 'risk': 10, 'return': 20}
        """
        if not self.model['DQM']:
            self.build_dqm()

        self.sample_set['DQM'] = self.sampler['DQM'].sample_dqm(self.model['DQM'], 
                                                                label="Example - Portfolio Optimization")

        solution = {}

        sample = self.sample_set['DQM'].first.sample 
        solution['cryptos'] = {s:self.shares_intervals[s][sample[s]] for s in self.cryptos}
        
        solution['return'], solution['risk'] = self.compute_risk_and_returns(solution['cryptos'])

        spending = sum([self.price[s]*solution['cryptos'][s] for s in self.cryptos])

        print(f'\nDQM -- solution for alpha == {self.alpha} and gamma == {self.gamma}:')
        print(f"\nShares to buy:")

        print("\n".join("{}\t{:>3}".format(k, v) for k, v in solution['cryptos'].items())) 

        print(f"\nEstimated returns: {solution['return']}")

        print(f"Purchase Cost: {spending:.2f}")

        print(f"Variance: {solution['risk']}\n")


        return solution 

    def dqm_grid_search(self):
        """Execute parameter (alpha, gamma) grid search for DQM.
        """
        alpha = self.alpha_list 
        gamma = self.gamma_list 

        data_matrix = np.zeros((len(alpha), len(gamma)))
        
        if self.verbose:
            print("\nGrid search results:")
            
        for i in range(len(alpha)):
            for j in range(len(gamma)):

                alpha_i = alpha[i]
                gamma_j = gamma[j]

                self.build_dqm(alpha_i, gamma_j)

                # Solve the problem using the DQM solver
                solution = self.solve_dqm()

                data_matrix[i,j] = solution['return'] / np.sqrt(solution['risk'])

        n_opt = np.argmax(data_matrix)

        self.alpha = alpha[n_opt//len(gamma)]
        self.gamma = gamma[n_opt - (n_opt//len(gamma)) * len(gamma)]

        print(f"DQM Grid Search Completed: alpha={self.alpha}, gamma={self.gamma}.-")

    def compute_risk_and_returns(self, solution):
        """Compute the risk and return values of solution.
        """
        variance = 0.0
        for s1, s2 in product(solution, solution):
            variance += (solution[s1] * self.price[s1] 
                        * solution[s2] * self.price[s2]  
                        * self.covariance_matrix[s1][s2])

        est_return = 0
        for crypto in solution:
            est_return += solution[crypto]*self.price[crypto]*self.avg_monthly_returns[crypto]

        return round(est_return, 2), round(variance, 2)

    def run(self, min_return=0, max_risk=0, num=0, init_holdings=None): 
        """Execute sequence of load_data --> build_model --> solve.

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            num (int): Number of cryptos to be randomnly generated. 
            init_holdings (float): Initial holdings, or initial portfolio state. 
        """
        self.load_data(num=num)
        if self.model_type=='CQM': 
            print(f"\nCQM run...")
            self.solution['CQM'] = self.solve_cqm(min_return=min_return, 
                                                  max_risk=max_risk, 
                                                  init_holdings=init_holdings)
        else:
            print(f"\nDQM run...")
            if len(self.alpha_list) > 1 or len(self.gamma_list) > 1:
                print("\nStarting DQM Grid Search...")
                self.dqm_grid_search()

            self.build_dqm()
            self.solution['DQM'] = self.solve_dqm()
