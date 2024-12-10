import os
import pandas as pd
import numpy as np
import random
import time
import math
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import comb

################# Public Functions #################
def ErlangLoss(Lambda, Mu, N = None):
    # The Erlangloss Model is exactly the same as MMN0 and this returns the whole probability distribution
    # Solves the Erlang loss system
    if N == 0: # if there is 0 unit. The block probability is 1
    	return [1]
    if N is not None: # If there is a size N, we constitute the Lambda and Mu vectors manually 
        Lambda = np.ones(N)*Lambda
        Mu = Mu*(np.arange(N)+1)
    else: # if the Lambdas and Mus are already given in vector form then no need to do anything
        N = len(Lambda)
    LoM = [1] + [l/m for l,m in zip(Lambda, Mu)]
    Prod = [np.prod(LoM[0:i]) for i in range(1,N+2)]
    P_n = Prod/sum(Prod)
    return P_n

def Loss_f(a, n): # This is the sfunction for loss probability that is used to calculate P_b
    return a**n/math.factorial(n)

def Get_Effective_Lambda(L, Mu, N):
    # Get the effective lambda that gives the desired offered load
    # Using Newton's Method
    rho_ = 0
    # rho = L * (1+L/(N*(N-L)))
    rho = 1 # this is not used. Just an initialization that will be replaced by rho_ in the iteration
    step = 0
    while np.abs(rho-rho_) > 0.001:
        rho = rho_
        l = rho * Mu
        B = ErlangLoss(l, Mu, N)[-1] # the loss probability for the loss system
        #print('B', B)
        rho_ = rho - (rho*(1-B) - L)/(1-B-(N-rho+rho*B)*B)
        # print(rho)
        step += 1
        if step > 100:
            break
    Lambda_eff = rho * Mu
    return Lambda_eff

class Two_State_Hypercube():
    def __init__(self, data_dict = None):
        # initilize data stored in self.data_dict
        self.keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 't_mat', 'pre_list', 'pol']
        self.rho_hyper, self.rho_approx, self.rho_simu = None, None, None # initialize the utilizations to be None 
        self.prob_dist = None
        self.data_dict = dict.fromkeys(self.keys, None) 
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys:
                    self.data_dict[k] = v
        self.G = None # G for approximation
        self.Q = None # Q for approximation
        self.r = None # r for approximation
        self.P_b = None # This is to differentiate from 3-state case
        self.q_nj = None # This is the dispatch probability using Linear-alph algorithm. This is shared by both 2-state and 3-state algorithms. 

    def Update_Parameters(self, **kwargs): 
        # update any parameters passed through kwargs
        for k, v in kwargs.items(): 
            if k in self.keys:
                self.data_dict[k] = v
                if k == 'pre_list': # reset G if pre_list changes
                    self.G = None
    
    def Random_Pref(self, seed=9001):
        # random preference list
        random.seed(seed) # Set Random Seed
        N, K = self.data_dict['N'], self.data_dict['K']
        # Shuffle the IDs as each preference list
        pre_list = np.array([random.sample(list(range(N)),N) for _ in range(K)])
        self.Update_Parameters(pre_list = pre_list)
    
    def Random_Fraction(self, seed=9001):
        '''
            Obtain random frac. 
        '''
        np.random.seed(seed)
        K = self.data_dict['K']
        frac_j = np.random.random(size=K)
        frac_j /= sum(frac_j)
        self.data_dict['frac_j'] = frac_j
        return frac_j

    def Random_Time_Mat(self, t_min = 1, t_max = 10, seed = 9001):
        np.random.seed(seed)
        N, K = self.data_dict['N'], self.data_dict['K']
        t_mat = np.random.uniform(low=t_min, high=t_max, size=(K,N))
        self.Update_Parameters(t_mat = t_mat)
        return t_mat

    def Myopic_Policy(self, source='t_mat'):
        # Obtain the policy to dispatch the cloest available unit given the time matrix
        N, K = self.data_dict['N'], self.data_dict['K']
        if source == 't_mat':
            t_mat = self.data_dict['t_mat']
            pre_list = t_mat.argsort(axis=1)
            self.Update_Parameters(pre_list = pre_list)
        elif source == 'pre':
            pre_list = self.data_dict['pre_list']
        else:
            print('Wrong source!')
        policy = np.zeros([2**N, K],dtype=np.int_)
        for s in range(2**N):
            for j in range(K):
                pre = pre_list[j]
                for n in range(N):
                    if not s >> pre[n] & 1: # n th choice is free
                        policy[s, j] = pre[n]
                        break
        self.data_dict['pol'] = policy

    def Cal_Trans(self):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Calculate the 
        N_state = 2**N
        A = np.zeros([N_state, N_state])
        # Calculate upward transtition
        for s in range(N_state-1): # The last state will not transition to other states by a arrival
            pol_s = pol[s]
            for j in range(K):
                dis = pol_s[j]
                A[s, s+2**dis] += Lambda * frac_j[j]
        # Calculate downward transtition
        for s in range(1,N_state): # The first state will not transition
            bin_s = bin(s)
            len_bin = len(bin_s)
            i = 0
            while bin_s[len_bin-1-i] != 'b':
                if bin_s[len_bin-1-i] == '1':
                    A[s,s-2**i] = Mu
                i += 1
        return A

    def Solve_Hypercube(self, update_rho = True):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Get the transition Matrix 
        A = self.Cal_Trans()
        # Solve for the linear systems
        transition = A.T - np.diag(A.T.sum(axis=0))
        #print (np.linalg.det(transition))
        #print ('Eigenvalues of Transition Matrix:',np.linalg.eig(transition)[0])
        transition[-1] = np.ones(2**N)
        b = np.zeros(2**N)
        b[-1] = 1
        start_time = time.time() # staring time
        prob_dist = np.linalg.solve(transition,b)
        # print("------ Hypercube run %s seconds ------" % (time.time() - start_time))
        if update_rho: # store the utilizations
            statusmat = [("{0:0"+str(N)+"b}").format(i) for i in range(2**N)]
            busy = [[N-1-j for j in range(N) if i[j]=='1'] for i in statusmat]
            rho = [sum([prob_dist[j] for j in range(2**N) if i in busy[j]]) for i in range(N)]
            self.rho_hyper = rho
        self.prob_dist = prob_dist
        return prob_dist
    
    def Get_MRT_Hypercube(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pol', 'frac_j', 't_mat']
        N, K, pol, frac_j, t_mat = [self.data_dict.get(key) for key in keys]
        prob_dist = self.prob_dist 
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for n in range(N): # The last state has value 0
            q_nj[:,n] = frac_j * np.dot(prob_dist[:-1], pol[:-1,:]==n) # here we don't need last state so take :-1
        q_nj /= (1-prob_dist[-1])
        self.q_nj = q_nj # store these values in the class
        MRT = np.sum(q_nj*t_mat)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        return MRT, MRT_j

    def Get_LateCalls_Hypercube(self, threshold): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pol', 'frac_j', 't_mat']
        N, K, pol, frac_j, t_mat = [self.data_dict.get(key) for key in keys]
        prob_dist = self.prob_dist 
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for n in range(N): # The last state has value 0
            q_nj[:,n] = frac_j * np.dot(prob_dist[:-1], pol[:-1,:]==n) # here we don't need last state so take :-1
        q_nj /= (1-prob_dist[-1])
        self.q_nj = q_nj # store these values in the class
        tau = 1.75
        late_calls_matrix = t_mat > threshold - tau
        late_calls = np.sum(q_nj*late_calls_matrix)
        return late_calls

    # For Approximation
    def Cal_P_n(self):
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        P_n = ErlangLoss(Lambda, Mu, N)
        return P_n

    def Cal_Q(self, P_n = None):
        keys = ['N', 'Lambda', 'Mu']
        N, Lambda, Mu = [self.data_dict.get(key) for key in keys]
        if self.G is None:
            self.G = [[np.where(self.data_dict['pre_list'][:,i] == j)[0] for i in range(N)] for j in range(N)] 
        Q = np.zeros(N)
        if P_n is None:
            P_n = self.Cal_P_n()
        N = len(P_n) - 1
        # r = Lambda/(Mu*N) * (1-P_n[-1]) # two ways of calculating r. This one is wrong for general cases where P_N is not from erlang loss
        r = np.dot(P_n,np.arange(N+1))/N

        for j in range(N):
            Q[j] = sum([math.factorial(k)/math.factorial(k-j) * math.factorial(N-j)/math.factorial(N)* (N-k)/(N-j) * P_n[k] for k in range(j,N)])/ (r**(j) * (1-r))
        self.Q = Q
        self.r = r
        return Q

    def Larson_Approx(self, epsilon=0.0001): # this already integrates the linear-alpha with two places that have alpha in the code
        keys = ['N', 'Lambda', 'Mu', 'frac_j', 'pre_list']
        N, Lambda, Mu, frac_j, pre_list = [self.data_dict.get(key) for key in keys]
        try:
            alpha = self.alpha
        except:
            # print('Two state!')
            alpha = 0

        use_effective_lambda = True # This part is not in the paper now. 
        # Step 0: Initialization
        self.Cal_Q() # This calculates Q, r, and G
        r = self.r # average fraction of busy time for each unit in the system. Calculated in Cal_Q
        # print(r)
        if self.P_b is not None:
            # print('All pooled')
            r = Lambda/(N*Mu)*(1-self.P_b) # P_b is the block probability. The probability that all units are busy

        rho_i = np.full(N,r) # utilization of each unit i. Initialization
        rho_i_ = np.zeros(N) # temporary utilizations to store new value at each step
        n = 0
        # Step 1: Iteration
        start_time = time.time()
        while True:
            n += 1 # increase step by 1
            rho_total = (rho_i + (1-rho_i)*alpha)
            ######################
            # Use the effective lambda to get the most accurate P_n and Q for each iteration (This helps a lot)
            if use_effective_lambda:
                if self.P_b is not None: 
                    L = rho_total.sum()
                    Lambda_eff = Get_Effective_Lambda(L, Mu, N)
                    P_n = ErlangLoss(Lambda_eff, Mu, N)
                    self.Cal_Q(P_n) # when use this Q, the old Q is overwritten and does not work
            ######################
            for i in range(N): # for each unit 
                value = 1 # 
                for k in range(N): # for each order
                    prod_g_j = 0 # Product term for each sum term
                    for j in self.G[i][k]:
                        prod_g_j += Lambda*frac_j[j]*self.Q[k]* np.prod(rho_total[pre_list[j,:k]]) # if alpha = 0, this is just rho in paranthasis
                    value += (1/Mu) * prod_g_j  # There should be a 1/mu here. because we don't assume it to be 1. 
                rho_i_[i]= (1-((1-rho_i)*alpha)[i])*(1-1/value) # again when alpha = 0, this is 1 in the paranthesis
            # Step 2: Normalize. 
            # Here the normalizing factor only takes on 1 service because we assume adding the other does not change much. 
            Gamma = rho_i_.sum()/(r*N)  # r is only used here for normalization
            rho_i_ /= Gamma
            # Step 3: Convergence Test
            if abs(rho_i_ - rho_i).max() < epsilon:
                # print ('Program stop in',n,'iterations in ', (time.time() - start_time), 'secs')
                # print(rho_i_)
                self.rho_approx = rho_i_
                return rho_i_
            else: # go to next step
                rho_i = np.array(rho_i_)
                rho_i_ = np.zeros(N)

    def Get_MRT_Approx(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            # print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        # print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        # print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        MRT = np.sum(q_nj*t_mat)
        return MRT, MRT_j

    def Get_ambulance_util(self): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        MRT_j = np.sum(q_nj*t_mat,axis = 1)/np.sum(q_nj, axis=1)
        MRT = np.sum(q_nj*t_mat)
        return q_nj

    def Get_LateCalls_Approx(self, threshold): # Method 1 of getting response time as in Larson
        keys = ['N', 'K', 'pre_list', 'frac_j', 't_mat', 'Mu']
        N, K, pre_list, frac_j, t_mat, Mu = [self.data_dict.get(key) for key in keys]
        try: # if self.alpha exists, it is three state. rho is the total rho
            rho = self.rho_total_approx
            print('Three state! Rho total is:', rho)
        except: # two state. rho is the normal approximate rho
            print('Two state!')
            rho = self.rho_approx
        
        P_n = self.Cal_P_n()
        # self.Cal_Q(P_n) ### 
        print('average rho:', self.r, np.mean(rho))
        Q = self.Q
        # This is the average response time for each state s
        q_nj = np.zeros([K, N])
        for j in range(K):
            pre_j = pre_list[j]
            for n in range(N):
                q_nj[j,pre_j[n]] = Q[n]*np.prod(rho[pre_j[:n]])*(1-rho[pre_j[n]])
            #print(q_nj[j,:].sum())
            q_nj[j,:] *= (1-P_n[-1])/q_nj[j,:].sum() # normalization
            q_nj[j,:] *= frac_j[j]

        print('sum of q_nj', q_nj.sum())
        # q_nj /= (1-P_n[-1])
        q_nj /= q_nj.sum() # same as divide by (1-P_allbusy)
        self.q_nj = q_nj # store these values in the class
        # print('q_nj', q_nj)
        tau = 1.75
        late_calls_matrix = t_mat > threshold - tau
        late_calls = np.sum(q_nj*late_calls_matrix)
        return late_calls

    def Get_Percent_Pref_Dispatch(self):# get percentage of percentage responsed by most preferred unit
        pre_list = self.data_dict['pre_list']
        preferred_units_j = pre_list[:,0] # mmost preferred units for each node j
        percent_pref_response = np.sum(self.q_nj[np.arange(len(self.q_nj)), preferred_units_j]) # get the corresponsing fraction for each node and sum up 
        # print('percent_pref_response',percent_pref_response)
        return percent_pref_response