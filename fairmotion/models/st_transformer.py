import numpy as np
import torch
from torch import nn
import random
import pickle
import math

class SpatioTemporalTransformer(nn.Module):

    def __init__(self, M=3, D=32, feedforward_dim=64, L=3, p=0.1):

        super(SpatioTemporalTransformer, self).__init__()


        self.N = 24 # 24 joints
        self.T = 120 # 120 frames needed for each training instance 
        self.M = M # aa preprocessing has dim=3 
        self.D = D # joint embedding dimension
        self.H = 2 # number of heads 
        self.F = int(self.D/self.H) # dimension of each head 
        self.feedforward_dim = feedforward_dim # hidden layer dimension in feedforward network
        self.L = L # number of attention layers
        self.p = p # dropout probability

        # joint embeddings: project all joints into D dimensions
        # linear layers for each joint. There are 120 joints in total. 
        self.E_layers = nn.ModuleList([nn.Linear(self.M, self.D) for i in range(self.N)])
        self.dropout = nn.Dropout(self.p)

        # initialize positional encoding: T=120 entries. Each entry is of shape [8]
        self.positional_encoding = [torch.zeros(self.D) for t in range(self.T)]
        for t, encoding in enumerate(self.positional_encoding):
            for i in range(self.D):
                denominator = pow(10000, 2*i/self.D)
                if i % 2 == 0:
                    encoding[i] = math.sin(t/denominator)
                else:
                    encoding[i] = math.cos(t/denominator)

        # temporal attention: Each of the 24 joints has a weight matrix. Weight is shared across different timesteps
        # 2 heads are implemented
        self.temporal_Q_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.N)])
        self.temporal_K_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.N)])
        self.temporal_V_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.N)])

        self.softmax = nn.Softmax(dim=2) # each row sums to 1
        self.mask = torch.zeros(self.T, self.T)
        self.mask = self.mask - float('inf')
        self.mask = torch.triu(self.mask, diagonal=1)

        self.temporal_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.D, num_heads=self.H) for i in range(self.N)])
        # layer normalization over just the joint embedding dimension. Shape is batch_size x T x N x D
        self.attention_norm = nn.LayerNorm(self.D) 

        # spatio attention: Each of the 120*24=2880 joints in all the time frames has a different weight matrix for spatio Q
        # for spatio K and V, each of the 120 timesteps has a weight matrix. Weight is shared across different joints 
        # self.spatio_Q_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.N*self.T)])
        self.spatio_Q_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.T)])
        self.spatio_K_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.T)])
        self.spatio_V_layers = nn.ModuleList([nn.Linear(self.D, self.D) for i in range(self.T)])

        self.spatio_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.D, num_heads=self.H) for i in range(self.T)])

        # feedforward network
        self.feedforward_l1 = nn.Linear(self.D, self.feedforward_dim)
        self.relu = nn.ReLU()
        self.feedforward_l2 = nn.Linear(self.feedforward_dim, self.D)

        # output projection layer
        self.output_projection_layers = nn.ModuleList([nn.Linear(self.D, self.M) for i in range(self.N)])

    def init_weights(self):
        pass

    def forward(self, X, tgt=None, max_len=None, teacher_forcing_ratio=None):
        """
        :param X: tensor of shape (batch_size, T, N*M)
        :returns outputs: tensor of shape (batch_size, T, N*M)
        """
        X = torch.transpose(X,1,2)

        # joint embeddings: project all joints into D dimensions
        n, t = 0, 0 
            # joint embeddings for each joint. 120 frames and each frame has 24 joints. 120*24=2880 joints in total which is len(self.E)
            # each joint is batch_size x D = batch_size x 8, which is self.E[i].shape
            # dropout before passing to the attention layers after after positioning encoding
        E = [] 
        while n < self.N and t < self.T:
            j_t_n = X[:,self.M*n:self.M*(n+1),t]

            # pass through linear layer and add positional encoding here
            e_t_n = self.dropout(self.E_layers[n](j_t_n) + self.positional_encoding[t])
            E.append(e_t_n)
            # loop through 24 joints in each time step first, before incrementing the time step.
            if n == self.N-1:
                t += 1
                n = 0
            else:
                n += 1 

        # L attention layers
        for l in range(self.L):
            # temporal attention 
                # temporal_E: There are 24 distinct joints. len(self.temporal_E)=N=24
                # Each joint has 120 frames and each joint has dim=8. Therefore self.temporal_E[i].shape = batch_size x 120 x 8 
            temporal_E = []
            for n in range(self.N):
                temporal_E_n_stg = [E[n+t*self.N] for t in range(self.T)]
                temporal_E_n = torch.stack(temporal_E_n_stg,1)
                temporal_E.append(temporal_E_n)
            
                # calculate Q,K,V: len(self.temporal_Q/K/V)=24
                # self.temporal_Q/K/V[i] has shape of batch_size x T x F which is batch_size x 120 x 4
            temporal_Q = [self.temporal_Q_layers[i](temporal_E[i]) for i in range(self.N)]
            temporal_K = [self.temporal_K_layers[i](temporal_E[i]) for i in range(self.N)]
            temporal_V = [self.temporal_V_layers[i](temporal_E[i]) for i in range(self.N)]

                # calculate temporal attention: len(self.temporal_attention_head) = N = 24
                # self.temporal_summary_stg[i] shape is batch_size x T x D = batch_size x 120 x 8
            temporal_summary_stg = [torch.transpose(self.temporal_attention_layers[i](torch.transpose(temporal_Q[i],0,1), torch.transpose(temporal_K[i],0,1), torch.transpose(temporal_V[i],0,1), attn_mask=self.mask)[0],0,1) for i in range(self.N)]

                # self.temporal_summary needs to stack to become tensor of shape batch_size x T x N x D = batch_size x 120 x 24 x 8
            temporal_summary = self.dropout(torch.stack(temporal_summary_stg,2))
                # self.temporal_E_full shape is batch_size x T x N x D. Same as self.temporal_summary
                # dropout before adding them up and layer norm
            temporal_E_full = torch.stack(temporal_E,2)
            temporal_summary_norm = self.attention_norm(temporal_E_full+temporal_summary)

            # spatial attention
                # There are 120 frames. len(self.spatial_E)=T=120
                # Each frame has 24 joints and each joint has dim=8. Therefore self.spatial_E[i].shape = batch_size x N x D = batch_size x 24 x 8
            spatio_E = []
            # spatio_Q = []
            for t in range(self.T):
                spatial_E_t_stg = [E[n+t*self.N] for n in range(self.N)]
                spatial_E_t = torch.stack(spatial_E_t_stg,1)
                spatio_E.append(spatial_E_t)

                # # calculate spatio Q in this loop as well. len(spatio_Q)=T=120. spatio_Q[i] shape is batch_size x N x F = batch_size x 24 x 4
                # spatio_Q_stg = [self.spatio_Q_layers[n+t*self.N](E[n+t*self.N]) for n in range(self.N)]
                # spatio_Q_t = torch.stack(spatio_Q_stg,1)
                # spatio_Q.append(spatio_Q_t)

                # calculate spatio K and V
                # self.spatio_Q/K/V has length of T=120 and each entry has batch_size x N x F = batch_size x 24 x 4
            spatio_Q = [self.spatio_Q_layers[i](spatio_E[i]) for i in range(self.T)]
            spatio_K = [self.spatio_K_layers[i](spatio_E[i]) for i in range(self.T)]
            spatio_V = [self.spatio_V_layers[i](spatio_E[i]) for i in range(self.T)]

                # calculate spatio attention: len(self.spatio_attention_head) = T = 120
                # self.spatio_summary_stg shape is batch_size x N x D = batch_size x 24 x 8
            spatio_summary_stg = [torch.transpose(self.spatio_attention_layers[i](torch.transpose(spatio_Q[i],0,1), torch.transpose(spatio_K[i],0,1), torch.transpose(spatio_V[i],0,1))[0],0,1) for i in range(self.T)]  

                # self.spatio_summary needs to stack to become tensor of shape batch_size x T x N x D = batch_size x 120 x 24 x 8            
            spatio_summary = self.dropout(torch.stack(spatio_summary_stg,1))
                # self.spatio_E_full shape is batch_size x T x N x D. Same as self.spatio_summary
                # dropout before adding them up and layer norm
            spatio_E_full = torch.stack(spatio_E,1)
            spatio_summary_norm = self.attention_norm(spatio_E_full+spatio_summary)

            # aggregation of temporal and spatio normalized summary: batch_size x T x N x D = batch_size x 120 x 24 x 8
            temporal_spatio_summary = temporal_summary_norm + spatio_summary_norm

            # feedforward hidden layer: only change dimension of D, and not T or N. Output is still batch_size x T x N x D
                # also inserts dropout here 
            feedforward_output = self.dropout(self.feedforward_l2(self.relu(self.feedforward_l1(temporal_spatio_summary))))
                # add feedforward_output with temporal_spatio_summary and normalize
            feedforward_norm = self.attention_norm(feedforward_output+temporal_spatio_summary)

            # do not need this if this is already the final layer
            batch_size = feedforward_output.shape[0]
            if l < self.L-1:
                # reshape (batch_size x T x N x D) into (batch_size x T x N*D) then into (batch_size x N*D x T)
                feedforward_reshaped = torch.transpose(torch.reshape(torch.flatten(feedforward_norm), (batch_size, self.T, self.N*self.D)),1,2)

                # reformat feedforward_output to become self.E which is len 120*24=2880 and contains e_t_n of shape batch_size x D
                # this will be passed back to the attention layer 
                n, t = 0, 0 
                E = [] 
                while n < self.N and t < self.T:
                    e_t_n = feedforward_reshaped[:,self.D*n:self.D*(n+1),t]
                    E.append(e_t_n)
                    # loop through 24 joints in each time step first, before incrementing the time step.
                    if n == self.N-1:
                        t += 1
                        n = 0
                    else:
                        n += 1 

        # output projection layer: N linear layers, each representing a joint. 
            # self.feedforward_norm shape is batch_size x T x N x D. Get batch_size x T x D for each joint
            # Pass through linear layer to get batch_size x T x M. Stack to get batch_size x T x N x M = batch_size x 120 x 24 x 3
        output_projection_stg = [self.output_projection_layers[n](feedforward_norm[:,:,n,:]) for n in range(self.N)]
        output_projection = torch.stack(output_projection_stg,2)
            # reformat (batch_size x T x N x M) into (batch_size x T x N*M)  = batch_size x 120 x 72 
        output = torch.reshape(torch.flatten(output_projection),(batch_size,self.T,self.N*self.M))
            # final residual connection
        return output + torch.transpose(X,1,2)


# if __name__ == "__main__":
#     # load the pickle file 
#     # file_handler = open('/Users/krystal/Desktop/small_preprocessed/aa/test.pkl', 'rb')
#     # data = pickle.load(file_handler)
#     # file_handler.close()
#     # data = data[0]
#     # print(data[0].shape) # (120, 72)
#     # print(len(data)) # 32

#     batch_size = 32
#     T = 120
#     N = 24 # number of joints
#     M = 3
#     D = 8
#     X = torch.rand(batch_size, T, N*M) # [10, 120, 72]

#     import time 
#     a = time.time()
    
#     test = SpatioTemporalTransformer()
#     # pytorch_total_params = sum(p.numel() for p in test.parameters())
#     # print(pytorch_total_params)
#     result = test(X) 
#     # print(len(result))
#     # print(result[0].shape)

#     criterion = nn.MSELoss()
#     loss = criterion(result, X)
#     loss.backward()
#     optimizer = torch.optim.SGD(test.parameters(), lr=0.001, momentum=0.9)
#     optimizer.step()
#     b = time.time() # 81.35835313796997 for batch_size=64, M=3, D=32, feedforward_dim=64, L=3, p=0.1. 6.7 hours
#     # if simplify spatio_Q, then 79.21889805793762 for batch_size=64, M=3, D=32, feedforward_dim=64, L=3, p=0.1
#     # 37.1215 for batch_size=32, M=3, D=32, feedforward_dim=64, L=3, p=0.1. 6.2hours
#     # if no batch_first and add transpose, then 39.071s for batch_size=32, M=3, D=32, feedforward_dim=64, L=3, p=0.1. 6.5hours
#     time_taken = b-a
#     print(time_taken)

#     time_taken = b-a
#     print(time_taken) # 98.77776312828064 for batch_size=64, M=3, D=32, feedforward_dim=64, L=3, p=0.1. Takes 8+ hours. 
#     # 45.550 for batch_size=32, M=3, D=32, feedforward_dim=64, L=3, p=0.1. Takes 7.5h 

#     # print(result.shape)
