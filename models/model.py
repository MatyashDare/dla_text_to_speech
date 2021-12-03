import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, model_size, dk):
        super().__init__()
        self.Qlayer = nn.Linear(model_size, dk)
        self.Klayer = nn.Linear(model_size, dk)
        self.Vlayer = nn.Linear(model_size, dk)
        self.dk = dk

    def forward(self, x, attention_mask):
        Q = self.Qlayer(x)
        K = self.Klayer(x)
        V = self.Vlayer(x)
        QK = F.softmax(torch.matmul(Q, K.permute(0, 2, 1)) / self.dk ** 0.5 + attention_mask, dim=-1)
        return torch.matmul(QK, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, dk, n_heads):
        super().__init__()
        self.head = nn.ModuleList()
        for i in range(n_heads):
            self.head.append(Attention(model_size, dk))
        self.tail = nn.Linear(n_heads * dk, model_size)

    def forward(self, x, attention_mask):
        tmp = []
        for module in self.head:
            tmp.append(module(x, attention_mask))
        out = torch.cat(tmp, dim=-1)
        out = self.tail(out)
        return out


class InterLayer(nn.Module):
    def __init__(self, model_size, inter_size, kernel_size, activation, dropout_p):
        super().__init__()
        print(model_size, inter_size, kernel_size)
        self.inter = nn.Sequential(nn.Conv1d(model_size, inter_size, kernel_size=kernel_size, padding='same'),
                                          activation(),
                                          nn.Conv1d(inter_size, model_size, kernel_size=1, padding='same'))
        self.layer_norm = nn.LayerNorm(model_size)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        add_att = self.dropout(self.inter(x))
        x = (x + add_att).permute(0, 2, 1)
        x = self.layer_norm(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, model_size,
                 inter_size,
                 inter_kernel_size,
                 activation,
                 n_heads,
                 dk,
                 dropout_p):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(model_size, dk, n_heads)
        self.context_linear = nn.Linear(n_heads * dk, model_size)
        self.intermediate_size = inter_size
        self.intermediate_layer = InterLayer(model_size,
                                                    inter_size,
                                                    inter_kernel_size,
                                                    activation,
                                                    dropout_p)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(model_size)
        self.d = torch.nn.Dropout(dropout_p)

    def forward(self, x, attention_mask):
        # (bs, seq len, hidden size)
        print('! x.shape', x.shape)
        print(' !attention_mask.shape', attention_mask.shape)
        print('! self.multi_head_attention', self.multi_head_attention(x, attention_mask).shape)
        attention_output = self.multi_head_attention(x, attention_mask)
        add_att = self.d(self.context_linear(attention_output))
        print('add_att!', add_att.shape)
        x = self.layer_norm(x + add_att)
        x = self.intermediate_layer(x)
        return x, attention_mask



class TransformerModel(nn.Module):
    def __init__(self, n_layers,
                       model_size,
                       inter_size,
                       iter_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       dropout_p):
        super().__init__()
        self.n_layers = n_layers
        args = [model_size,
                inter_size,
                iter_kernel_size,
                activation,
                n_heads,
                size_per_head,
                dropout_p]
        self.layers = nn.ModuleList([TransformerLayer(*args) for i in range(n_layers)])

    def forward(self, x, attention_mask):
        for i in range(self.n_layers):
            x, attention_mask = self.layers[i](x, attention_mask)
        return x


class SinCosPE(nn.Module):
    def __init__(self,
                 emb_size,
                 maxlen):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embeddings):
        return embeddings + self.pos_embedding[:embeddings.shape[0], :]


def duplicate_by_duration(encoder_result, durations):
    bs = encoder_result.shape[0]
    encoder_result = encoder_result.reshape(-1, encoder_result.shape[2])
    melspec_len = durations[0].sum()
    assert(torch.all(durations.sum(1) == melspec_len))
    durations = durations.flatten()
    durations_cumsum = durations.cumsum(0)
    mask1 = torch.arange(bs * melspec_len)[None, :] < (durations_cumsum[:, None])
    mask2 = torch.arange(bs * melspec_len)[None, :] >= (durations_cumsum - durations)[:, None]
    mask = (mask2 * mask1).float()
    encoder_result = mask.T @ encoder_result
    encoder_result = encoder_result.reshape(bs, int(melspec_len.item()), encoder_result.shape[1])
    return encoder_result


class FastSpeechModel(nn.Module):
    def __init__(self, vocab_size,
                       max_len,
                       n_layers,
                       output_size,
                       model_size,
                       inter_size,
                       iter_kernel_size,
                       activation,
                       n_heads,
                       size_per_head,
                       dropout_p):
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU
        args = [n_layers,
                model_size,
                inter_size,
                iter_kernel_size,
                activation,
                n_heads,
                size_per_head,
                dropout_p]
        self.tokens_positions = SinCosPE(model_size, max_len)
        self.frames_positions = SinCosPE(model_size, max_len)
        self.embedding_layer = nn.Embedding(vocab_size, model_size)
        self.encoder = TransformerModel(*args)
        self.decoder = TransformerModel(*args)
        self.output_layer = nn.Linear(model_size, output_size)

    def forward(self, batch):
        tokens = batch["tokens"]
        melspec = batch["melspec"]
        melspec_length = batch["melspec_length"]
        tokens_length = batch["token_lengths"]
        duration_multipliers = batch["duration_multipliers"]

        tokens_embeddings = self.embedding_layer(tokens)
        tokens_embeddings = tokens_embeddings + self.tokens_positions(tokens_embeddings)
        print('tokens_embeddings.shape',tokens_embeddings.shape)

        attention_mask = (torch.arange(tokens_embeddings.shape[1])[None, :] > tokens_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        print('attention_mask.shape', attention_mask[:, :, None].shape)
        encoder_result = self.encoder(tokens_embeddings, attention_mask[:, :, None])

        input_to_decoder = duplicate_by_duration(encoder_result, duration_multipliers)
        mask = (torch.arange(input_to_decoder.shape[1])[None, :] <= melspec_length[:, None]).float()
        input_to_decoder = input_to_decoder * mask[:, :, None]
        attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :] > melspec_length[:, None]).float()
        attention_mask[attention_mask == 1] = -torch.inf
        
        output = self.decoder(input_to_decoder, attention_mask)

        output = self.output_layer(output)
        output = output.permute(0, 2, 1)
        return output
        # output_attention_mask = (torch.arange(input_to_decoder.shape[1])[None, :] <= melspec_length[:, None]).float()
        # return output * output_attention_mask[:, None, :]
