# 2_2_coffee_time.py

from flask import Flask, render_template, request


app = Flask(__name__)


order_list = {}

# 퀴즈
# 메뉴 주문을 받는 서버를 구축하세요
# 주문 방법: 이름, 메뉴

@app.route('/')
def index():
    name = request.args.get("name")
    menu = request.args.get("menu")
    if name is not None:
        order_list[name] = menu
    return render_template('coffee.html', order_list=order_list)


if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.68')
