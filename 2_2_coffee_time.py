# 2_2_coffee_time.py
from flask import Flask, render_template, request


# 퀴즈
# 메뉴 주문을 받는 서버를 구축하세요
# 주문 방법: 이름, 메뉴
app = Flask(__name__)
orders = {}


# http://192.168.0.96:5000?name=정훈&menu=카페라떼
@app.route('/')
def order():
    name = request.args.get('name')
    menu = request.args.get('menu')

    orders[name] = menu
    return render_template('orders.html', orders=orders)


if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.96')







