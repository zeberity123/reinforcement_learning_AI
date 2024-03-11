# 1_3_flask.py
import numpy as np
from flask import Flask, render_template, request
import random
import keras


app = Flask(__name__)

# 퀴즈
# iris 모델을 구축해서
# 크롬으로부터 넘겨받은 피처에 대해 결과를 알려주세요
model = keras.models.load_model('model/iris.keras')
classes = ['setosa', 'versicolor', 'virginica']


@app.route('/iris')
def predict():
    # http://127.0.0.1:5000/iris?s_len=5.1&s_wid=3.5&p_len=1.4&p_wid=0.2
    s_len = float(request.args.get('s_len'))
    s_wid = float(request.args.get('s_wid'))
    p_len = float(request.args.get('p_len'))
    p_wid = float(request.args.get('p_wid'))

    p = model.predict([[s_len, s_wid, p_len, p_wid]], verbose=0)
    p_arg = np.argmax(p, axis=1)
    return render_template('iris.html', result=classes[p_arg[0]])


@app.route('/')
def index():
    return 'hello, flask!!'


# 퀴즈
# 로또 숫자 6개를 반환하는 함수를 만드세요
@app.route('/lotto')
def lotto645():
    a = list(range(1, 46))
    random.shuffle(a)
    return a[:6]


@app.route('/first')
def first():
    lotto = lotto645()
    return render_template('first.html', lotto=lotto)


# http://127.0.0.1:5000/whoru?name=김정훈&age=21
@app.route('/whoru')
def show_me():
    name = request.args.get('name')
    age = request.args.get('age')

    return render_template('who-r-u.html', name=name, age=age)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='192.168.0.96')



