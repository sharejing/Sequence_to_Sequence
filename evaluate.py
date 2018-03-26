from preprocess import variableFromSentence, input_lang, output_lang, pairs
from torch.autograd import Variable
import torch

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    input_variable = variableFromSentence(input_lang, sentence).cuda()
    input_length = input_variable.size()[0]

    encoder_hidden = encoder.initHidden(1).cuda()
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([SOS_token])).cuda()
    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        topv, topi = decoder_output.data.topk(1)

        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([ni])).cuda()
    return decoded_words, 0



def evaluateTestSet(encoder, decoder):

    test_data = open('data/test_last_next.txt', 'r')
    iter_test_data = iter(test_data)
    for line in iter_test_data:
        list = line.strip().split('\t')
        output_words, attentions = evaluate(encoder, decoder, list[0])
        output_sentence = ''.join(output_words)
        print('setup = ', list[0].replace(' ', ''))
        print('punch = ', list[1].replace(' ', ''))
        print('model = ', output_sentence)
        print('\n')


# test_ui.py needs it!

# def evaluateTestSet(encoder, decoder, setup):
#
#     output_words, attentions = evaluate(encoder, decoder, setup)
#     output_sentence = ''.join(output_words)
#
#     return output_sentence.replace('SOS', '')