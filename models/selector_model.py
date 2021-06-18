import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SelectorModel(nn.Module):
    def __init__(self, mlp_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('roberta-base')

        self.in_dim = 768
        self.hidden_dim = 768
        self.mlp_size = mlp_size
        self.selector = LSTMSelector(in_size=self.in_dim, hidden_dim=self.hidden_dim, mlp_size=self.mlp_size)
    
    def forward(self, ctx, target, target_len, ctx_len, n_step):
        target_emb = torch.stack(
            [self.encode(target[i]) for i in range(target.size(0))], dim=0
        )
        ctx_emb = torch.stack(
            [self.encode(ctx[i]) for i in range(ctx.size(0))], dim=0
        )
        outputs, dist = self.selector(ctx_emb, target_emb, ctx_len, target_len, n_step)
        return outputs, dist
    
    def encode(self, input_ids):
        return self.encoder(input_ids)[0][:, 0]



class LSTMSelector(nn.Module):
    def __init__(self, in_size, hidden_dim, mlp_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_size

        self.fn_activate = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.lstm_cell = nn.LSTMCell(in_size, hidden_dim)
        self.mlp = nn.ModuleList(
                    [nn.Linear(in_size+hidden_dim, mlp_size, bias=False),
                    self.drop_out,
                    self.fn_activate,
                    nn.Linear(mlp_size, 1, bias=False)]
        )        
    
    def forward(self, ctx_emb: torch.Tensor, target_emb: torch.Tensor, ctx_len, target_len, n_step):
        outputs = []
        dists = []

        lstm_in = target_emb # bs x in_dim
        bs = lstm_in.size()[0]
        ns = ctx_emb.size()[1]
        h_0 = torch.zeros((bs, self.hidden_dim))
        c_0 = torch.zeros((bs, self.hidden_dim))
        lstm_state = (h_0, c_0)
        dim = self.hidden_dim + self.in_dim
        mask = torch.ones((bs, ns))
        for _ in n_step:
            h, c = self.lstm_cell(lstm_in, lstm_state)

            ctx_emb = torch.cat([h.unsqueeze(1).expand((-1, ns, -1)), ctx_emb], dim=2) # bs x ns x (in_dim + hidden_dim)
            ctx_emb = self.drop_out(ctx_emb)
            sc = torch.sigmoid(self.mlp(ctx_emb)) # bs x ns
            sc = sc * mask

            if self.training:
                prob = F.softmax(sc, dim=-1)
                C = torch.distributions.Categorical(probs=prob)
                out = C.sample() # bs x 1: index of selected sentence in this step
                dists.append(C)
                outputs.append(out)
            else:
                out = sc.max(dim=-1)[1] # bs x 1: index of selected sentence in this step
                outputs.append(out)
            for i in range(len(out)):
                mask[i, out[i]] *= 0
            
            lstm_in = torch.gather(ctx_emb, dim=1, index=out.unsqueeze(1).unsqueeze(2).expand(bs, 1, self.in_dim))
            lstm_in = lstm_in.squeeze(1)
            lstm_state = (h, c)
        
        return outputs, dists


