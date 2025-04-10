from dataclasses import dataclass
import torch.multiprocessing as multiprocessing
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    # max_batch_size: int = 32
    max_batch_size: int = 1
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor,
        # wq: Optional[torch.Tensor] = None,
        # wk: Optional[torch.Tensor] = None,
        # wv: Optional[torch.Tensor] = None,
        # wo: Optional[torch.Tensor] = None,
    ):
        print("Runnning self-attention forward waiting for weight...")
        print(x)
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)
        print (f"batch_size : {batch_size} , seq_len : {seq_len} , start_pos : {start_pos}") 
        print(self.wq)

        print(self.wq.weight)
        print(self.wk.weight)
        print(self.wv.weight)
        
        # print(wq.weight)
        # print(wk.weight)
        # print(wv.weight)
        
        # print(f"x : {x} , wq : {self.wq.weight} , wk : {self.wk.weight} , wv : {self.wv.weight}")
        
        print("calculating q,k,v")
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)
        
        # # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        # xq = wq(x)
        # # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        # xk = wk(x)
        # # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        # xv = wv(x)


        print("calculating q,k,v")
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # print("calculated q,k,v")
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # print("calculating q k with rotary embedding")
        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # print("replacing k v in kv cache")
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # print("loaded k v from kv cache")
        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # print("starting selfattention execution..")
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # print("completed selfattention")
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        # swish = F.silu(self.w1(x))
        # # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        # x_V = self.w3(x)
        # # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        # x = swish * x_V
        # # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        # x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, state_dict: Optional[dict] = None):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.args=args
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # self.attention.share_memory()
        # self.feed_forward.share_memory()

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.processCreated = False
        self.state_dict = state_dict
        self.forward_proc_create()
        

    def forward_proc_create(self):
        # if self.processCreated:
        #     print("Processes already created")
        #     return
        
        attention_dict = {k.replace("attention.", ""): v for k, v in self.state_dict.items() if k.startswith("attention.")}
        feed_forward_dict = {k.replace("feed_forward.", ""): v for k, v in self.state_dict.items() if k.startswith("feed_forward.")}


        # self.layers = [self.apply_attention_norm, self.apply_self_attention, self.apply_ffn_norm, self.apply_feed_forward]
        # self.args = [self.attention_norm, self.state_dict, self.ffn_norm, self.feed_forward]
        # self.layer_args = [(self.args, attention_dict), (self.feed_forward, feed_forward_dict)]
        self.layer_args = [(self.attention, attention_dict), (self.feed_forward, feed_forward_dict)]
        # self.attention.share_memory()
        # self.feed_forward.share_memory()
        # self.processCreated = True
        # self.layers = [self.apply_attention_norm, self.apply_ffn_norm]

        # self.layers = [self.apply_self_attention, self.apply_feed_forward]
        self.layers = [self.apply_self_attention]

        self.len_layers = len(self.layers)
        self.pipes = [multiprocessing.Pipe() for i in range(self.len_layers + 1)]
        # main -> l1 -> l2 -> l3 -> l4 -> main
        self.processes = []
        # print(self.pipes)

        for layer_idx in range(self.len_layers):
            # self.args[layer_idx].share_memory()

            # print(f"pipes for {self.layers[layer_idx]}: {self.pipes[layer_idx]} -> {self.pipes[layer_idx]}")
            # t = multiprocessing.Process(target=self.layers[layer_idx], args=(self.pipes[layer_idx + 1][1], self.pipes[layer_idx][0], self.args[layer_idx]))
            t = multiprocessing.Process(target=self.layers[layer_idx], args=(self.pipes[layer_idx + 1][1], self.pipes[layer_idx][0], self.layer_args[layer_idx]))
            # t = threading.Thread(target=self.layers[layer_idx], args=(self.pipes[layer_idx + 1][1], self.pipes[layer_idx][0]))
            self.processes.append(t)
            # print("created process for layer : ", self.layers[layer_idx].__name__, "with pname : ", t.name)
            # args=(self.pipes[layer_idx + 1][1], self.pipes[layer_idx][0])
            # self.layers[layer_idx](args[0], args[1])            
            t.start()
            print("started process for layer : ", t.name, "with pid : ", t.pid)

    def forward_proc_runner(self, x: torch.Tensor, meta_d):       
        # Main process sends initial message to the first worker
        self.pipes[0][1].send((x, meta_d))
        # Main process receives final message from the last worker
        out, meta_d = self.pipes[self.len_layers][0].recv()
        # print(f'Main process received: {x, meta_d}')
        # print("#################################")
        return out
    
    def exit_processes(self, ):
        len_layers = len(self.layers)
        print("Terminating processes...")
        for idx in range(len_layers):
            self.pipes[idx+1][1].send(('exit', {}))

        # Join all processes
        # for p in self.processes:
        #     p.join()
            
        # Close all pipes
        for pipe in self.pipes:
            pipe[0].close()
            pipe[1].close()


    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        print(self.attention.wq.weight)
        # self.forward_proc_create()
        meta_d = {'start_pos': start_pos, 'freqs_complex': freqs_complex}
        return self.forward_proc_runner(x, meta_d=meta_d)
    
    def apply_attention_norm(self, send_conn, recv_conn, norm_layer):
        # self.attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        while True:
            print("waiting for input at apply_attention_norm")
            x,meta_d = recv_conn.recv()
            print("got for input at apply_attention_norm")
            if x == "exit":
                send_conn.send((x, meta_d))
                # recv_conn.close()
                # send_conn.close()
                exit()
            else:
                x = norm_layer.forward(x).detach()
                send_conn.send((x, meta_d))

    def apply_self_attention(self, send_conn, recv_conn, layer_args):
    # def apply_self_attention(self, send_conn, recv_conn, state_dict):
        print("in apply_self_attention")

        attention = layer_args[0]
        state_dict = layer_args[1]
        
        print(attention.wq)

        print("printing weights")

        while True:
            print("waiting for input at apply_self_attention")
            x, meta_d = recv_conn.recv()
            print("got for input at apply_self_attention")
            print("in apply_self_attention a")

            # for name, param in attnetion_layer.named_parameters():
            #     print(name, param.data)

            print("in apply_self_attention b")
            # if x == "exit":
            #     send_conn.send((x, meta_d))
            #     exit()
            # else:
            #     x = (x + attention.forward(x, meta_d['start_pos'], meta_d['freqs_complex'])).detach()
            #     # print("calculated apply_self_attention")
            #     send_conn.send((x, meta_d))

    def apply_ffn_norm(self, send_conn, recv_conn, norm_layer):        
        # self.ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        while True:
            print("waiting for input at apply_ffn_norm")
            x, meta_d= recv_conn.recv()
            print("got for input at apply_ffn_norm")
            if x == "exit":
                send_conn.send((x, meta_d))
                # recv_conn.close()
                # send_conn.close()
                exit()
            else:
                x = norm_layer.forward(x).detach()
                send_conn.send((x, meta_d))

    def apply_feed_forward(self, send_conn, recv_conn, feed_forward_layer):
        feed_forward = feed_forward_layer[0]
        state_dict = feed_forward_layer[1]
        # print(state_dict['wq.weight'])
        # print(nn.Parameter(state_dict['wq.weight'].float()))

        # attention.load_state_dict(state_dict)
        feed_forward.w1.weight = nn.Parameter(state_dict['w1.weight'])
        feed_forward.w2.weight = nn.Parameter(state_dict['w2.weight'])
        feed_forward.w3.weight = nn.Parameter(state_dict['w3.weight'])
        # self.feed_forward = FeedForward(self.args)
        while True:
            print("waiting for input at apply_feed_forward")
            x, meta_d= recv_conn.recv()
            print("got for input at apply_feed_forward")
            if x == "exit":
                send_conn.send((x, meta_d))
                # recv_conn.close()
                # send_conn.close()
                exit()
            else:
                x = (x + feed_forward.forward(x)).detach()
                send_conn.send((x, meta_d))

    
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs, state_dict_layers: Optional[dict] = None):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        # for layer_id in range(args.n_layers):
        #     self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args, state_dict_layers[layer_id]))
        # return 

    def forward(self, tokens: torch.Tensor, start_pos: int):
        print("in Transformer forward")
        print(self.layers[0].attention.wq.weight)
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers

        # for layer_id in range(self.args.n_layers):
        #     # self.layers.append(EncoderBlock(self.args))
        #     layer = EncoderBlock(self.args)
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        
        # for layer in self.layers:
        #     layer('exit', start_pos, freqs_complex)

        # print(f"completed execution and terminated processes Output : {output}")
        return output
    def exit_model(self):
        for layer in self.layers:
            layer.exit_processes()
        print("Terminated all processes")