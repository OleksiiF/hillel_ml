import os
import string

import numpy as np
from torch import nn, device, cuda, from_numpy, optim, LongTensor, save, load
import torch.nn.functional as torch_func
from torch.utils.data import DataLoader, TensorDataset, Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self):
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

        with open('text.txt', 'rb') as fh:
            text = self.__get_cleaned_text(fh.read().decode())

        self.chars = tuple(set(text))
        self.int_to_vocab = dict(enumerate(self.chars))
        self.vocab_to_int = {ch: index for index, ch in self.int_to_vocab.items()}
        self.encoded = [self.vocab_to_int[word] for word in text]

        self.seq_length = 8
        self.batch_size = 256

    def __get_cleaned_text(self, data):
        white_chars = string.ascii_letters + ''.join(self.pun_chars.keys()) + ' '
        data = ''.join([
            char for char in data.lower().strip()
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

        target_tensors = from_numpy(np.array(target))
        feature_tensors = from_numpy(np.array(feature))

        data = TensorDataset(feature_tensors, target_tensors)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        return data_loader


class RNN(nn.Module):
    def __init__(self, dataset, embedding_dim=256, hidden_dim=512, n_layers=3, dropout=0.3, lr=0.0003, n_epochs=10):
        super().__init__()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
        # dataset
        self.dataset = dataset
        self.train_loader = dataset.get_dataloader()
        self.output_size = len(dataset.vocab_to_int)
        self.vocab_size = len(dataset.vocab_to_int)
        # rnn
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim).to(self.device)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(hidden_dim, self.output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # logging
        self.show_every_n_batches = 100

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

    def forward_back_prop(self, inp, target, hidden):
        inp = inp.cuda().to(self.device)
        target =  target.cuda().to(self.device)
        hidden = tuple([i.data for i in hidden])

        self.zero_grad()
        out, hidden = self(inp, hidden)

        loss = self.criterion(out, target.type(LongTensor).to(self.device))
        loss.backward()

        clip = 5
        nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.optimizer.step()

        return loss.item(), hidden

    def train_rnn(self, save_model=False):
        batch_losses = []
        self.train()
        print(f"Training for {self.n_epochs} epoch(s)")
        for epoch_i in range(1, self.n_epochs + 1):
            hidden = self.init_hidden(self.dataset.batch_size)

            for batch_i, (inputs, labels) in enumerate(self.train_loader, 1):
                n_batches = len(self.train_loader.dataset) // self.dataset.batch_size
                if batch_i > n_batches:
                    break
                loss, hidden = self.forward_back_prop(inputs, labels, hidden)

                batch_losses.append(loss)
                if batch_i % self.show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}'.format(
                        epoch_i, self.n_epochs, np.average(batch_losses)
                    ))
                    batch_losses = []

            if save_model:
                dir = 'rnn_models'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass

                save(self, f"{dir}/model_{epoch_i}.pt")
                print('Model trained and saved')

        return self

    def generate(self, prime_word, predict_len=100):
        self.eval()
        prime_id = self.dataset.vocab_to_int[prime_word]
        current_seq = np.full(
            (1, self.dataset.seq_length),
            self.dataset.vocab_to_int['<PAD>']
        )

        current_seq[-1][-1] = prime_id
        predicted_raw = [prime_word]

        for _ in range(predict_len):
            current_seq = LongTensor(current_seq).to(self.device)
            hidden = self.init_hidden(current_seq.size(0))
            output, _ = self(current_seq, hidden)
            p = torch_func.softmax(output, dim=1).data
            p = p.to(self.device)
            top_k = 5
            p, top_i = p.topk(top_k)
            p = p.cpu()
            top_i = top_i.cpu()
            top_i = top_i.numpy().squeeze()
            p = p.numpy().squeeze()
            word_i = np.random.choice(top_i, p=(p / p.sum()))
            word = self.dataset.int_to_vocab[word_i]
            predicted_raw.append(word)
            current_seq = current_seq.cpu().numpy()
            current_seq = np.roll(current_seq, -1, 1)
            current_seq[-1][-1] = word_i

        predicted = []
        tmp = None
        for token in predicted_raw:
            if token in self.dataset.pun_chars.values() and tmp == token:
                continue

            tmp = token
            predicted.append(token)

        gen_sentences = ' '.join(predicted)

        for key, token in self.dataset.pun_chars.items():
            gen_sentences = gen_sentences.replace(f' {token.lower()}', key)

        gen_sentences = gen_sentences.replace('\n ', '\n')

        return gen_sentences


if __name__ == '__main__':
    dataset = Dataset()
    rnn = RNN(dataset=dataset)
    # rnn = load(rnn.path.format(1))
    rnn.train_rnn(save_model=True)

    generated_text = rnn.generate('well')
    with open('generated_text.txt', 'w+') as fh:
        fh.write(generated_text)
