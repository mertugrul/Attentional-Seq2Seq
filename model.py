SEED = 1234
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dropout_val, bidirectional, device):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_val = dropout_val
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)#, padding_idx=0)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=(0 if self.num_layers == 1 else self.dropout),
                          bidirectional=self.bidirectional, batch_first=True)

    def forward(self, x, x_lens, hidden=None):

        x_padded = pad_sequence(x, batch_first=True, padding_value=0)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = x_lens.sort(0, descending=True)
        x_padded = x_padded[perm_idx]

        embedded = self.embedding(x_padded)

        x_packed = pack_padded_sequence(embedded, seq_lengths, batch_first=True)
        #print(x_packed.data.shape)
        #print(x_packed.batch_sizes)

        output, hidden = self.gru(x_packed)
        if self.bidirectional:
            output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])

        output, _ = pad_packed_sequence(output, batch_first=True)

        #print('output.shape:', output.shape)
        #print('hidden.shape:', hidden.shape)
        ## TO DO support multi-layer rnns.
        #hidden = hidden.view(self.num_layers, 2, output.size(0), self.hidden_size)  # 2 for bidirectional

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class Attention(nn.Module):
    def __init__(self, att_type, hidden_size):
        super(Attention, self).__init__()
        self.att_type = att_type
        self.hidden_size = hidden_size
        if self.att_type == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        if self.att_type == 'bahdanau':
            self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
            self.Ua = nn.Linear(self.hidden_size, self.hidden_size)

    # encoder_output.shape (B, T, D), hidden.shape (B, 1, D)
    def concat_score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)))
        energy = torch.sum(self.v * energy, dim=2)
        return energy  #(B, T)

    # encoder_output.shape (B, T, D), hidden.shape (B, 1, D)
    def bahdanau_score(self, hidden, encoder_outputs):
        energy = F.tanh((self.Wa(hidden.expand(-1, encoder_outputs.size(1), -1)) + self.Ua(encoder_outputs)))
        energy = torch.sum(self.v * energy, dim=2)
        return energy  #(B, T)

    # hidden.shape (B, 1, D), encoder_output.shape (B, T, D)
    def dot_score(self, hidden, encoder_outputs):
        energy = torch.sum(hidden.expand(-1, encoder_outputs.size(1), -1) * encoder_outputs, dim=2)
        return energy  #(B, T)


    def forward(self, hidden, encoder_outputs):
        hidden = torch.transpose(hidden, 0, 1)  #(B, 1, D)
        if self.att_type == 'concat':
            energies = self.concat_score(hidden, encoder_outputs)
        elif self.att_type == 'bahdanau':
            energies = self.bahdanau_score(hidden, encoder_outputs)
        elif self.att_type == 'dot':
            energies = self.dot_score(hidden, encoder_outputs)

        return F.softmax(energies, dim=1).unsqueeze(1)  #(B, 1, T)


class Decoder(nn.Module):
    def __init__(self, attn_type, output_size, hidden_size, embedding_size, num_layers, dropout_val, bidirectional, device):
        super(Decoder, self).__init__()
        self.attn_type = attn_type
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_val = dropout_val
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)  # , padding_idx=0)
        self.dropout = nn.Dropout(self.dropout_val)
        self.attn = Attention(self.attn_type, self.hidden_size)
        self.gru = nn.GRU(self.embedding_size + self.hidden_size, self.hidden_size, self.num_layers, dropout=(0 if self.num_layers == 1 else self.dropout),
                          bidirectional=self.bidirectional, batch_first=True)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)


    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        ## Apply attention here
        attn_weights = self.attn(last_hidden, encoder_outputs)  #(B, 1, T)
        context = torch.bmm(attn_weights, encoder_outputs)  #(B, 1, D)
        rnn_input = torch.cat([embedded, context], 2)
        print('decoder rnn input shape:', rnn_input.shape)
        ## Apply Decoder RNN here
        #rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        print('decoder rnn output shape:', rnn_output.shape)
        rnn_output = rnn_output.squeeze(1)  # (B, 1, D) --> (B, D)
        context = context.squeeze(1)   #(B, 1, D) --> (B, D)
        output = self.out(torch.cat([rnn_output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights





class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, trg, trg_lens, teacher_forcing_ratio=0.5):

        batch_size = len(trg)
        max_len = len(trg[0])
        trg_vocab_size = self.decoder.output_size

        ## To keep the predicted outputs
        outputs = Variable(torch.zeros(max_len, batch_size, trg_vocab_size, device=self.device))

        ## Get encoder outputs
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)

        ## Decoder operations
        decoder_input = torch.ones(10, 1, device=self.device, dtype=torch.long) * 1 ## SOS_token
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]

        max_len = decoder_input.size(0)

        print('encoder_outputs.shape:', encoder_outputs.shape)
        print('encoder_hidden.shape:', encoder_hidden.shape)
        print('decoder_input.shape:', decoder_input.shape)
        print('decoder_hidden.shape:', decoder_hidden.shape)

        for i in range(1, max_len):
            rnn_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            print('rnn_output.shape', rnn_output.shape)
            print('decoder_hidden.shape', decoder_hidden.shape)
            #break