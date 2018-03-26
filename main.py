from model.seq2seq import EncoderRNN, DecoderRNN
from preprocess import input_lang, output_lang
from train import trainEpochs
import torch

hidden_size = 384

encoder = EncoderRNN(input_lang.n_words, hidden_size).cuda()
decoder = DecoderRNN(hidden_size, output_lang.n_words).cuda()

trainEpochs(encoder, decoder, 200000, print_every=500, learning_rate=0.001)

print('大吉大利　encoder_decoder 训练完毕......')

torch.save(encoder.state_dict(), './encoder.pth')
torch.save(decoder.state_dict(), './decoder.pth')