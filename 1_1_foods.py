# 1_1_foods.py
import re
import requests


# 퀴즈
# 폴리텍 신기술교육원의 식단 정보를 보여주세요
def solution_1():
    url = "https://www.kopo.ac.kr/int/content.do?menu=2520"
    response = requests.get(url)
    # print(response)
    # print(response.text)

    tbody = re.findall(r'<tbody>(.+?)</tbody>', response.text, re.DOTALL)
    # print(tbody)
    # print(len(tbody))
    # print(tbody[0])

    tr = re.findall(r'<tr>(.+?)</tr>', tbody[0], re.DOTALL)
    # print(tr)
    # print(len(tr))

    for item, weekday in zip(tr, ('월', '화', '수', '목', '금', '토', '일')):
        item = item.replace('\r', '')
        span = re.findall(r'<span>(.+?)</span>', item)
        print('[{}요일]'.format(weekday))
        for meal, name in zip(span, ('아침', '점심', '저녁')):
            print(name, ':', meal)


# 퀴즈
# 앞에서 만든 코드를 post 방식으로 수정하고
# 원하는 주(week)의 식단을 보여주세요
def solution_2(date):
    payload = {
        'day': date         # '20240219'
    }

    url = "https://www.kopo.ac.kr/int/content.do?menu=2520"
    response = requests.post(url, data=payload)
    # print(response)
    # print(response.text)

    tbody = re.findall(r'<tbody>(.+?)</tbody>', response.text, re.DOTALL)
    # print(tbody)
    # print(len(tbody))
    # print(tbody[0])

    tr = re.findall(r'<tr>(.+?)</tr>', tbody[0], re.DOTALL)
    # print(tr)
    # print(len(tr))

    for item, weekday in zip(tr, ('월', '화', '수', '목', '금', '토', '일')):
        item = item.replace('\r', '')
        span = re.findall(r'<span>(.*?)</span>', item)
        print('[{}요일]'.format(weekday))
        for meal, name in zip(span, ('아침', '점심', '저녁')):
            print(name, ':', meal)


def solution_3(year, month, day):
    solution_2('{}{:02}{:02}'.format(year, month, day))


# solution_1()
# solution_2('20240305')
solution_3(year=2024, month=3, day=7)

