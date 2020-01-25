from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
       
        obs_index = self.obs_dict[Osequence[0]]
        for s in range(S):
            alpha[ s, 0 ] = self.pi[s] * self.B[ s, obs_index]

        for t in range(1, L):
            obs_index = self.obs_dict[Osequence[t]]
            for s in range(S):
                alpha[s, t] = self.B[s, obs_index] * sum([alpha[s2, t-1] * self.A[s2, s] for s2 in range(S)])

        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        
        for s in range(S):
            beta[ s, L - 1 ] = 1.0

        for t in range(L - 2, -1, -1):
            obs_index = self.obs_dict[Osequence[t+1]]
            for s in range(S):
                beta[s, t] = sum([ self.A[s, s2] * self.B[s2, obs_index] * beta[s2, t+1] for s2 in range(S) ])

        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        
        L = len(Osequence)
        alpha=self.forward(Osequence)                
        sum_arr=alpha.sum(axis=0)
        prob=float(sum_arr[L-1])
                
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)               
        a = np.multiply(alpha, beta)
        prob = a / a.sum(axis=0, keepdims=1)
        
        ###################################################
        return prob
    
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
                        
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)  
        a = np.multiply(alpha, beta)
        norm = a.sum(axis=0)[:-1]
        
        
        for i in range(S):
            for j in range(S):
                for k in range(L-1):
                    obs_index = self.obs_dict.get(Osequence[k+1])
                    prob[i,j,k] = alpha[i,k] * beta[j,k+1] * self.A[i,j] * self.B[j, obs_index] 
                    #print(prob[i,j,k])
                prob[i, j, :] = prob[i, j, :] / norm
        
        
        ###################################################
        print(prob)
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        
        obs_count = len(Osequence)
        state_count = len(self.state_dict)
        decode, backtrack = np.zeros((state_count, obs_count)), np.zeros((state_count, obs_count), dtype=np.int)
        backtrack[0] = 0;
        
        start_state = self.pi.reshape((state_count, 1))
        
        
        
        for s in range(0, state_count):
            if Osequence[0] in self.obs_dict:
                decode[s, 0] = start_state[s] * self.B[s, self.obs_dict[Osequence[0]]]
            else:
                decode[s, 0] = start_state[s] * 0.000006
                
            
         
        for t in range(1, obs_count):
            for s in range(0, state_count):
                prev_prob = decode[:, t - 1] * self.A[:, s]
                decode[s, t] = np.amax(prev_prob)
                backtrack[s, t] = np.argmax(prev_prob)
                if Osequence[t] in self.obs_dict:
                    decode[s, t] = decode[s, t] * self.B[s, self.obs_dict[Osequence[t]]]
                else:
                    decode[s, t] = decode[s, t] * 0.000006
                    
                
        
        result = np.zeros(obs_count, dtype=np.int);
        result[obs_count - 1] = decode[:, obs_count - 1].argmax()
        for t in range(obs_count - 1, 0, -1):
            result[t - 1] = backtrack[result[t], t]
            
        index_list=result.tolist()
        
        inv_map = {v: k for k, v in self.state_dict.items()}

        for index in index_list:
            path.append(inv_map.get(index))

        ###################################################
        return path
