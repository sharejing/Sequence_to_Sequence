from torch.autograd import Variable
import torch
import math
import time
import torch.optim as optim
import torch.nn as nn
from preprocess import get_batch, fra, eng


MAX_LENGTH = 10
SOS_token = 0

def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    batch_size, input_length = input_variable.size()
    target_length = target_variable.size()[1]
    encoder_hidden = encoder.initHidden(batch_size).cuda()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).cuda()
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        targ = target_variable[:, di]

        loss += criterion(decoder_output, targ)
        decoder_input = targ

    # 训练一次

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainEpochs(encoder, decoder, n_epochs, print_every=1000,learning_rate=0.01):

    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss().cuda()

    for epoch in range(1, n_epochs + 1):

        training_batch = get_batch(fra, eng)
        input_variable = Variable(torch.LongTensor(training_batch[0])).cuda()
        target_variable = Variable(torch.LongTensor(training_batch[1])).cuda()
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, criterion)

        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch,
                                         epoch / n_epochs * 100, print_loss_avg))