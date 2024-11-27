import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
class PositionEncodding(nn.Module):
     def __init__(self,d_model:int,seqn_len:int,dropout:float)->None:

        super().__init__()
        self.d_model = d_model
        self.seqn_len = seqn_len
        self.Dropout = nn.Dropout(dropout)

        ## Creating the matrix of length sqqn_len and d_model
        pe = torch.zeros(seqn_len,d_model)

        ## creating the sequeens length matrix of shape (seqn_len , 1)
        # position = torch.arange(0,seqn_len,dtype = torch.float).unsqueeze(1) 
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term) 

        for pos in range(seqn_len):
            for i in range(0,d_model,2):
                #pe[pos,i] = math.sin(pos / (10000 ** ((2*i)/d_model))) 
                #pe[pos , i+1]  = math.sin(pos / (10000 ** (2*(i+1)/d_model)))
                pe[pos, i] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        #pe.unsqueeze(0).transpose(0,1) 
        pe = pe.unsqueeze(0)

        ##saving the parameter in register buffer that not updated during the training
        
        self.register_buffer("pe", pe)       

     def forward(self , x):
         x = x+ (self.pe[:,:x.shape[1],:]).requires_grad_(False)
         return self.Dropout(x)



'''



''''

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()
    



class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
    



class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights




class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int , d_ff:int, dropout=0.1)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self , x):
        ## (batch_size , seq_len , d_model) --> (batch_size , seq_len , d_ff) --> (batch_size , seq_len , d_model)
        return  self.dropout(self.linear_2(nn.ReLU()(self.linear_1(x)))) 


class ResidualConnection(nn.Module): 
    def __init__(self,d_model:int, dropout_rate=0.1)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        

    def forward(self , x, sublayer):
        return  x + self.dropout(sublayer(self.norm(x)))
    





class EncoderBlock(nn.Module):
    def __init__(self ,d_model:int, attention_block :MultiHeadAttention, feed_forward_block : FeedForward , dropout : 0.1 )->None:
        super().__init__()

        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList(ResidualConnection(d_model,dropout) for _ in range(2))

    def forward(self , X ):
        ## srs_mask this mask we want to  apply to input of encoder
        #we  need this b/c we want to hide the interection of padding word with other word
        mask = None

        X = self.residual_block[0](X , lambda X : self.attention_block(X, X , X , mask))

        X = self.residual_block[1](X , lambda X : self.feed_forward_block(X))

        return X



class Encoder(nn.Module):

    def __init__(self,features:int , layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self , X , mask=None):
        for layer in self.layers:
            X = layer(X , mask)
        return self.norm(X)


class ProjecLayer(nn.Module):
    def __init__(self,d_model:int , seq_size:int)->None:
        super().__init__()
        self.proj = nn.Linear(d_model , seq_size)

    def forward(self , X):
        ## (batch_size , seq_len , d_model)    -->> (batch_size , seq_len , vocb_size) projection on the vocab_size
        return torch.log_softmax(self.proj(X),dim=-1)



class AuxiliaryGenerator(nn.Module):
    def __init__(self, input_dim):
        super(AuxiliaryGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.fc(x)        





'''



## New define architecture 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        **kwargs
    ):
        super(EncoderLayer, self).__init__()

       

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        # if self.diagonal_attention_mask:
        #     mask_time = torch.eye(self.d_time).to(self.device)
        # else:
        mask_time = None

        residual = enc_input
        
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()

        
            

class AuxiliaryGenerator(nn.Module):
    def __init__(self, input_dim):
        super(AuxiliaryGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.fc(x) 
    



    
