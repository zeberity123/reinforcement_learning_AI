# 2_8_chat_main.py
import numpy as np
import keras
import nltk
chat_util = __import__('2_7_chat_util')


def decode_prediction(seq, vocab, is_question):
    seq = list(seq)

    pos = seq.index(chat_util._EOS_) if chat_util._EOS_ in seq else len(seq)
    seq = seq[:pos]

    result = ' '.join(vocab[i] for i in seq)
    result = result.replace(vocab[chat_util._PAD_], '')
    print('여우 :' if is_question else '왕자 :', result)


def talk_with_bot():
    vocab = chat_util.load_vocab()
    vectors = chat_util.load_vectors()

    questions = vectors[::2]
    answers = vectors[1::2]

    max_len_enc = max([len(v) for v in questions])
    max_len_dec = max([len(v) for v in answers])

    # ----------------------------------------- #

    onehot = np.eye(len(vocab), dtype=np.float32)

    model = keras.models.load_model('chat/chat.keras')

    x_dec = chat_util.add_pad([chat_util._STA_], max_len_dec)
    x_dec = onehot[x_dec]       # (9, 164)
    x_dec = x_dec[np.newaxis]
    print(x_dec.shape)          # (1, 9, 164)

    while True:
        line = input('여우 : ')

        if '끝' in line:
            break

        x1 = nltk.tokenize.regexp_tokenize(line, r'\w+')
        x2 = [vocab.index(t) if t in vocab else chat_util._UNK_ for t in x1]
        x3 = chat_util.add_pad(x2, max_len_enc)

        x_enc = onehot[x3]
        x_enc = x_enc[np.newaxis]

        # 퀴즈
        # 예측해서 결과를 decode_prediction 함수에 전달해서 대화를 나눠보세요
        p = model.predict([x_enc, x_dec], verbose=0)
        # print(p.shape)          # (1, 9, 164)

        p_arg = np.argmax(p, axis=2)
        # print(p_arg.shape)      # (1, 9)

        decode_prediction(p_arg[0], vocab, is_question=False)


talk_with_bot()
