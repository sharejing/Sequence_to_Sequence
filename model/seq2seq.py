import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

MAX_LENGTH = 10


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)

    def forward(self, input, hidden):
        output, hidden = self.gru(self.embedding(input), hidden)
        return output, hidden

    # TODO: other inits
    def initHidden(self, batch_size):

        return Variable(torch.zeros(1, batch_size, self.hidden_size))


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):

        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        # TODO use transpose of embedding
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        emb = self.embedding(input).unsqueeze(1)
        res, hidden = self.gru(emb, hidden)
        output = self.sm(self.out(res[:, 0]))
        return output, hidden


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):

        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):

        return Variable(torch.zeros(1, 1, self.hidden_size))