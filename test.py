from model.seq2seq import EncoderRNN, DecoderRNN
from preprocess import input_lang, output_lang
import torch
# from evaluate import evaluateRandomly
from evaluate import evaluateTestSet

hidden_size = 384

encoder = EncoderRNN(input_lang.n_words, hidden_size).cuda()
decoder = DecoderRNN(hidden_size, output_lang.n_words).cuda()

encoder.load_state_dict(torch.load('./encoder.pth'))
decoder.load_state_dict(torch.load('./decoder.pth'))


# evaluateRandomly(encoder, decoder)

evaluateTestSet(encoder, decoder)