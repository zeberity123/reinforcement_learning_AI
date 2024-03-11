# 2_7_chat_util.py
import csv
import numpy as np
import keras


_PAD_, _STA_, _EOS_, _UNK_ = 0, 1, 2, 3


def load_vocab():
    f = open('chat/vocab.txt', 'r', encoding='utf-8')
    vocab = [w.strip() for w in f]
    # print(vocab)
    f.close()

    return vocab


# 퀴즈
# vectors.txt 파일을 반환하는 함수를 만드세요
def load_vectors():
    f = open('chat/vectors.txt', 'r', encoding='utf-8')
    vectors = [[int(t) for t in w.strip().split(',')] for w in f]
    # print(vectors)
    f.close()

    return vectors


# 퀴즈
# 시퀀스에 대해 패드를 추가해서 최대 길이로 만드세요
# seq: [114, 128, 85, 79]
def add_pad(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]

    # seq를 앞에 두는 이유는 _EOS_ 위치를 찾아서 유효 영역을 잘라내야 해서
    return seq + (max_len - len(seq)) * [_PAD_]


def make_xxy():
    vocab = load_vocab()
    vectors = load_vectors()

    # print(add_pad([114, 128, 85, 79], 7))

    questions = vectors[::2]
    answers = vectors[1::2]

    max_len_enc = max([len(v) for v in questions])
    max_len_dec = max([len(v) for v in answers])

    # ----------------------------------------- #

    onehot = np.eye(len(vocab), dtype=np.float32)

    x_enc, x_dec, y_dec = [], [], []
    for q, a in zip(questions, answers):
        enc_in = add_pad(q, max_len_enc)
        dec_in = add_pad([_STA_] + a, max_len_dec)
        target = add_pad(a + [_EOS_], max_len_dec)

        x_enc.append(onehot[enc_in])
        x_dec.append(onehot[dec_in])
        y_dec.append(target)

    return np.float32(x_enc), np.float32(x_dec), np.float32(y_dec), vocab


def save_model():
    x_enc, x_dec, y_dec, vocab = make_xxy()
    print(x_enc.shape, x_dec.shape, y_dec.shape)    # (52, 9, 164) (52, 9, 164) (52, 9)

    # 인코더
    enc_in = keras.layers.Input(shape=x_enc.shape[1:])
    _, enc_state = keras.layers.SimpleRNN(128, return_state=True)(enc_in)

    # 디코더
    dec_in = keras.layers.Input(shape=x_dec.shape[1:])
    output = keras.layers.SimpleRNN(128, return_sequences=True)(dec_in, initial_state=enc_state)
    output = keras.layers.Dense(len(vocab), activation='softmax')(output)

    model = keras.Model([enc_in, dec_in], output)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit([x_enc, x_dec], y_dec, verbose=2, epochs=1000)
    model.save('chat/chat.keras')


if __name__ == '__main__':
    save_model()



