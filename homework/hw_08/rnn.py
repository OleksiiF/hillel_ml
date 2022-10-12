import os

import torch.nn.functional as F
import string

import numpy as np
import torch

from torch import nn, device, cuda
from torch.utils.data import DataLoader, TensorDataset



class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        with open('text.txt', 'rb') as fh:
            text = self.__get_cleaned_text(fh.read())

        self.chars = tuple(set(text))
        self.int_to_vocab = dict(enumerate(self.chars))
        self.vocab_to_int = {ch: index for index, ch in self.int_to_vocab.items()}
        self.encoded = [self.vocab_to_int[word] for word in text]

        self.seq_length = 8
        self.batch_size = 256

        self.pun_chars = {
            '.': '||period||',
            ',': '||comma||',
            '"': '||quotation_mark||',
            ';': '||semicolon||',
            '!': '||exclamation_mark||',
            '?': '||question_mark||',
            '(': '||left_parentheses||',
            ')': '||right_Parentheses||',
            '-': '||dash||',
            '\n': '||return||'
        }

    def __get_cleaned_text(self, data):

        white_chars = string.ascii_letters + ''.join(self.pun_chars.values())
        data = ''.join([
            char for char in data.decode().lower().strip()
            if char in white_chars
        ])

        for key, token in self.pun_chars.items():
            data = data.replace(key, f' {token} ')

        return data.split() + ['<PAD>']

    def get_dataloader(self):
        batch_num = len(self.encoded) // self.batch_size
        batch_words = self.encoded[: (batch_num * self.batch_size)]

        feature, target = [], []
        target_len = len(batch_words[:-self.seq_length])

        for i in range(0, target_len):
            feature.append(batch_words[i: i + self.seq_length])
            target.append(batch_words[i + self.seq_length])

        target_tensors = torch.from_numpy(np.array(target))
        feature_tensors = torch.from_numpy(np.array(feature))

        data = TensorDataset(feature_tensors, target_tensors)

        data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return data_loader


class RNN(nn.Module):
    def __init__(self, dataset, embedding_dim=256, hidden_dim=512, n_layers=3, dropout=0.3):
        super().__init__()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_size = len(dataset.vocab_to_int)
        self.vocab_size = len(dataset.vocab_to_int)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        self.train_loader = dataset.get_dataloader(
            dataset.seq_length,
            dataset.batch_size
        )

        # Model Layers
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_size)

        self.epoch = 25
        self.lr = 0.0003
        self.show_every_n_batches = 500

    def forward(self, nn_input, hidden):
        batch_size = nn_input.size(0)
        nn_input = nn_input.long()

        embed_out = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embed_out, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)

        lstm_out = lstm_out.view(batch_size, -1, self.output_size)
        lstm_output = lstm_out[:, -1]

        return lstm_output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return(
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        )


    def forward_back_prop(self, rnn, optimizer, criterion, inp, target, hidden):
        inp, target = inp.cuda().to(self.device), target.cuda().to(self.device)

        hidden = tuple([i.data for i in hidden])

        rnn.zero_grad()
        out, hidden = rnn(inp, hidden)

        loss = criterion(out, target)
        loss.backward()

        clip = 5

        nn.utils.clip_grad_norm_(rnn.parameters(), clip)

        optimizer.step()

        return loss.item(), hidden


    def train_rnn(self):
        batch_losses = []
        self.train()
        print("Training for %d epoch(s)..." % self.n_epochs)
        for epoch_i in range(1, self.n_epochs + 1):
            hidden = self.init_hidden(self.batch_size)

            for batch_i, (inputs, labels) in enumerate(self.train_loader, 1):
                n_batches = len(self.train_loader.dataset) // self.batch_size
                if (batch_i > n_batches):
                    break
                loss, hidden = self.forward_back_prop(self, self.optimizer, self.criterion, inputs, labels, hidden)

                batch_losses.append(loss)
                if batch_i % self.show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                        epoch_i, self.n_epochs, np.average(batch_losses)))
                    batch_losses = []

        return self

    def generate(self, prime_word, predict_len=100):
        self.eval()
        prime_id = self.dataset.vocab_to_int[prime_word]
        current_seq = np.full((1, self.seq_length), '<PAD>')
        current_seq[-1][-1] = prime_id
        predicted = [self.dataset.int_to_vocab[prime_id]]

        for _ in range(predict_len):
            current_seq = torch.LongTensor(current_seq).to(self.device)
            hidden = self.init_hidden(current_seq.size(0))
            output, _ = self(current_seq, hidden)
            p = F.softmax(output, dim=1).data
            p = p.to(self.device)
            top_k = 5
            p, top_i = p.topk(top_k)
            top_i = top_i.numpy().squeeze()
            p = p.numpy().squeeze()
            word_i = np.random.choice(top_i, p=p / p.sum())
            word = self.dataset.int_to_vocab[word_i]
            predicted.append(word)
            current_seq = current_seq.cpu().numpy()
            current_seq = np.roll(current_seq, -1, 1)
            current_seq[-1][-1] = word_i

        gen_sentences = ' '.join(predicted)
        # for key, token in self.dataset.pun_chars.items():
        #     ending = ' ' if key in ['\n', '(', '"'] else ''
        #     gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
        #
        # gen_sentences = gen_sentences.replace('\n ', '\n')
        # gen_sentences = gen_sentences.replace('( ', '(')

        return gen_sentences


dataset = Dataset()
rnn = RNN(dataset=dataset)
trained_rnn = rnn.train_rnn()
save_filename = os.path.splitext(os.path.basename('./rnn_trained'))[0] + '.pt'
torch.save(trained_rnn, save_filename)
print('Model Trained and Saved')



generated_script = rnn.generate('dog')
print(generated_script)
