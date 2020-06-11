SEED = 1234
import random
import numpy as np
import utils
from dataset import Dataset
from io import open
import unicodedata
import string
import re

import torch
from torch.utils import data
from operator import itemgetter

from model import Encoder, Decoder, Seq2Seq

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sentence2Tensor(lang, sentence):
    tokens = [lang.word2index[word] for word in sentence.split(' ')]
    tokens.append(lang.EOS_token)
    return torch.tensor(tokens, dtype=torch.long, device=device).view(-1, 1)

def pair2Tensor(pair):
    input_tensor = sentence2Tensor(input_lang, pair[0])
    target_tensor = sentence2Tensor(output_lang, pair[1])
    return (input_tensor, target_tensor)

def senteces2Tensors(lang, sentences):
    outputs = []
    outputs_lens = []
    for sentence in sentences:
        tokens = [lang.word2index[word] for word in sentence.split(' ')]
        tokens.append(lang.EOS_token)
        #tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).view(-1, 1)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        outputs.append(tokens_tensor)
        outputs_lens.append(tokens_tensor.size()[0])
    return outputs, torch.LongTensor(outputs_lens, device=device)





if __name__ == "__main__":
    input_lang, output_lang, pairs = utils.prepareData('eng', 'fra', True)
    print(random.choice(pairs))

    train_ratio = 0.7
    test_ratio = 0.1
    # Parameters
    params = {'batch_size': 10,
              'shuffle': False,
              'num_workers': 6}
    max_epochs = 100
    train_src, train_trg, train_ids = ([pairs[i][0] for i in range(int(len(pairs)*train_ratio))],
                                       [pairs[i][1] for i in range(int(len(pairs)*train_ratio))],
                                       [i for i in range(int(len(pairs)*train_ratio))])
    test_src, test_trg, test_ids = ([pairs[i][0] for i in range(len(pairs) - int(len(pairs)*test_ratio), len(pairs))],
                                    [pairs[i][1] for i in range(len(pairs) - int(len(pairs)*test_ratio), len(pairs))],
                                    [i for i in range(len(pairs) - int(len(pairs)*test_ratio), len(pairs))])

    # Generators
    training_set = Dataset(train_ids)
    training_generator = data.DataLoader(training_set, **params)

    test_set = Dataset(test_ids)
    test_generator = data.DataLoader(test_set, **params)

    encoder = Encoder(input_size=input_lang.n_words, hidden_size=16, embedding_size=5, num_layers=1, dropout_val=0.2,
                      bidirectional=False, device=device)

    decoder = Decoder(attn_type='concat', output_size=output_lang.n_words, hidden_size=16, embedding_size=5, num_layers=1, dropout_val=0.2,
                      bidirectional=False, device=device)

    model = Seq2Seq(encoder, decoder, device)

    for batch in training_generator:
        #print(batch)
        batch_src, batch_trg = list(itemgetter(*batch)(train_src)), list(itemgetter(*batch)(train_trg))
        batch_src_tokens, batch_src_lens = senteces2Tensors(input_lang, batch_src)
        batch_trg_tokens, batch_trg_lens = senteces2Tensors(output_lang, batch_trg)

        #print(type(batch_src_tokens),type(batch_src_tokens))
        #print('len(batch_src_tokens):', len(batch_src_tokens), 'len(batch_src_lens):', len(batch_src_lens))
        #print('len(batch_trg_tokens):', len(batch_trg_tokens), 'len(batch_trg_lens):', len(batch_trg_lens))

        #print(batch_src_tokens)
        #print(batch_src_lens)
        #enc_hidden = encoder.initHidden()

        model(batch_src_tokens, batch_src_lens, batch_trg_tokens, batch_trg_lens)
        break




