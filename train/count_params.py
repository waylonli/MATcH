import torch
import torch.nn as nn
from transformers import BertModel

class Similarity(nn.Module):
    def __init__(self,
                 d_model, deep):
        super(Similarity, self).__init__()

        out_bi = d_model if deep else 1
        self.bi = nn.Bilinear(d_model, d_model, out_bi)
        self.proj = None
        if deep:
            self.proj = nn.Sequential(nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, svecs, pvecs):
        if self.proj is None:
            return self.bi(svecs, pvecs)
        v = self.bi(svecs, pvecs)
        return self.proj(v)
class TransfoEnc(nn.Module):
    def __init__(self,
                n_src_vocab, len_max_seq, d_word_vec,
                n_layers, n_head, d_k, d_v,
                d_model, d_inner, dropout=0.1, scratchbert=False, mathbert=False):

        super(TransfoEnc, self).__init__()

        # self.encoder = SelfAttentiveEncoder(n_src_vocab, len_max_seq, d_word_vec,
        #                                     n_layers, n_head, d_k, d_v,
        #                                     d_model, d_inner, dropout=dropout)
        assert not (scratchbert and mathbert)
        if scratchbert:
            self.encoder = torch.nn.parallel.DataParallel(BertModel.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/train_from_scratch/model_files"))
        elif mathbert:
            self.encoder = torch.nn.parallel.DataParallel(BertModel.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/baseline/MathBERT-custom"))
        else:
            raise Exception("No encoder specified")
        self.sim = Similarity(d_model, deep=False)

    def forward(self, input):
        """
        Args:
            input: list of torch.long tensors
        Returns:
            res: tensor of size (batch, out_dim)
        """
        input = [i[:self.encoder.max_length] for i in input]

        padded_input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
        pos = torch.arange(1, len(padded_input[0]) + 1).repeat(len(padded_input), 1).to(padded_input.device)

        output = self.encoder(padded_input, pos)

        # output is a tuple of length 1 or 2 (2 if return attention vectors)
        res = output[0]

        #res = res[:,0,:]
        return torch.max(res, dim=1)[0]

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


count_parameters(torch.load("./maximin_zero_anno_both_global_global/model"))