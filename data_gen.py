import numpy as np
import random
import tqdm
import math
import pandas as pd

import pyjosa

from fractions import Fraction
from decimal import Decimal

import torch
from torch.utils.data import Dataset, DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


class MWPDataset(Dataset):
    def __init__(self, seed=0, max_len=1000):
        self.seed = seed
        self.max_len = max_len

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(self.seed)
        
        self.problems = {'1-1': self.__get_problem01_01__,
                         '1-2': self.__get_problem01_02__,
                         '1-3': self.__get_problem01_03__,
                         '1-4': self.__get_problem01_04__,
                         '2-1': self.__get_problem02_01__,
                         '2-2': self.__get_problem02_02__,
                         '2-3': self.__get_problem02_03__,
                         '4-1': self.__get_problem04_01__,
                         '4-2': self.__get_problem04_02__}
                         
    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        num = len(self.problems)
        np.random.seed(idx)
        p = np.random.randint(num)
        problem_idx = list(self.problems.keys())[p]        
        sample = self.problems[problem_idx]()

        return sample

    def __get_value__(self, min, max):
        return np.random.randint(min, max)

    ''' 이름 하나 임의적으로 추출 '''
    def __get_name__(self):
        # name = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        idx = np.random.randint(0, len(name))
        return name[idx]

    ''' 중복안되게 이름 두개 임의적으로 추출 '''
    def __get_name2__(self):
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        n0, n1 = random.sample(name, 2)
        return n0, n1

    ''' 중복안되게 이름 세개 임의적으로 추출 '''
    def __get_name3__(self):
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        n0, n1, n2 = random.sample(name, 3)
        return n0, n1, n2

    ''' 중복안되게 이름 네개 임의적으로 추출 '''
    def __get_name4__(self):
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        n0, n1, n2, n3 = random.sample(name, 4)
        return n0, n1, n2, n3

    def __get_name4_2__(self):
        name = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
        n0, n1, n2, n3 = random.sample(name, 4)
        return n0, n1, n2, n3

    def __get_ordinal__(self, idx):
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13', '14', '15', '16',
                  '17', '18', '19', '20', '21', '22', '23','24', '25', '26', '27', '28', '29', '30']
		return number[idx]

    def __get_person__(self):
        name = ['학생', '사람']
        idx = np.random.randint(0, len(name))
        return name[idx]

    def __get_sport__(self):
        sport = ['달리기', '수영', '탁구', '볼링', '베드민턴', '자전거', '마라톤']
        idx = np.random.randint(0, len(sport))
        return sport[idx]

    def __get_ball__(self):
        ball = ['공', '당구공', '야구공', '축구공', '농구공', '배구공', '탁구공', '테니스공', '볼링공']
        idx = np.random.randint(0, len(ball))
        return ball[idx]

    def __get_box__(self):
        box = ['상자', '박스']
        idx = np.random.randint(0, len(box))
        return box[idx]

    def __get_odis__(self):
        name = ['홀수', '짝수']
        idx = np.random.randint(0, len(name))
        return name[idx]

    def __get_food__(self):
        food = ['사과', '복숭아', '배', '참외', '포도', '딸기', '바나나', '옥수수', '토마토', '오이', '배추', '무', '과자', '음료수', '주스', '우유', '감', '수박', '귤', '당근', '라면', '사탕', '김밥', '달걀']
        idx = np.random.randint(0, len(food))
        return food[idx]

    def __get_subject__(self):
        subject = ["국어","영어","수학","사회","과학","음악","미술","체육"]
        idx = np.random.randint(0, len(subject))
        return subject[idx]

    def __get_float_value__(self, min, max):
        return np.random.randint(min + .5, max) + (np.random.rand(1)[0] - .5)

    def __get_value_array__(self, min, max, num, dtype='int', frac_prob=0.1):
        np_array = np.random.choice(np.arange(min, max), size=num, replace=False)
        str_array = ''

        if dtype == 'int':
            for idx in range(num - 1):
                str_array += '%d, ' % np_array[idx]
            str_array += '%d' % np_array[-1]
        elif dtype == 'float':
            np_array = np_array + (np.random.rand(num) - .5)
            np_array = np.round_(np_array, 2)
            for idx in range(num - 1):
                # Fraction
                if np.random.rand() <= frac_prob:
                    val = '%.4f' % np_array[idx]
                    val = Fraction(Decimal(val))
                    str_val = '%d/%d, ' % (val.numerator, val.denominator)
                else:
                    str_val = '%.2f, ' % np_array[idx]
                str_array += str_val

            if np.random.rand() <= frac_prob:
                val = '%.2f' % np_array[-1]
                val = Fraction(Decimal(val))
                str_val = '%d/%d' % (val.numerator, val.denominator)
            else:
                str_val = '%.2f' % np_array[-1]

            str_array += str_val

        return str_array, np_array

    def __get_ordinal2__(self, max):
        if max == 2:
            name = ['가장', '1번째로', '2번째로']
        elif max == 3:               
            name = ['가장', '1번째로', '2번째로', '3번째로']
        elif max == 4:                        
            name = ['가장', '1번째로', '2번째로', '3번째로', '4번째로']
        idx = np.random.randint(0, len(name))
        if idx <= 2:
            ord_idx = 0
        elif idx <= 4:
            ord_idx = 1
        elif idx <= 6:
            ord_idx = 2
        elif idx <= 8:
            ord_idx = 3
        return name[idx], ord_idx

    def __get_pos_str__(self, pos):
        if pos == 1:
            name = '한'
        elif pos == 2:
            name = '두'
        elif pos == 3:
            name = '세'
        elif pos == 4:
            name = '네'
        elif pos == 5:
            name = '다섯'
        return name

    def __get_shift__(self):
        if np.random.rand() < .5:
            s1 = '오른'
            s2 = '커'
            op = 'increase'
        else:
            s1 = '왼'
            s2 = '작아'
            op = 'decrease'
        return s1, s2, op

    def __get_operation__(self):
        p = np.random.rand()
        if p < .25:
            str_op = '합은'
            op = 'add'
        elif p < .5:
            str_op = '차는'
            op = 'sub'
        elif p < .75:
            str_op = '곱은'
            op = 'mul'
        else:
            str_op = '평균은'
            op = 'avg'
        return str_op, op

    ''' 유형1 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem01_01__(self):
        v0 = self.__get_value__(0, 100)
        v1 = self.__get_value__(0, 100)

        n0 = self.__get_name__()

        box = self.__get_box__()
        ball = self.__get_ball__()

        eq = '%d + %d' % (v0, v1)
        ans = '%d' % (v0 + v1)

        p = np.random.rand()
        if p < 0.34:
            que = '%s 안에 %d개의 %s이 있습니다. %s가 %d개의 %s을 %s 안에 더 넣었습니다. %s 안에 있는 %s은 모두 몇 개입니까?' \
                  % (box, v0, ball, n0, v1, ball, box, box, ball)
        elif p < 0.67:
            que = '%s 안에 %d개의 %s이 있고, %s가 %d개의 %s을 %s 안에 더 넣었습니다. %s 안에 있는 %s은 모두 몇 개인지 구하시오.' \
                  % (box, v0, ball, n0, v1, ball, box, box, ball)
        else:
            que = '%s 안에 %d개의 %s이 있습니다. %s가 %d개의 %s을 %s 안에 더 넣었다면, %s 안에 있는 %s은 모두 몇 개일까요?' \
                  % (box, v0, ball, n0, v1, ball, box, box, ball)

        return que, eq, ans

    def __get_problem01_02__(self):
        v0 = self.__get_value__(1, 90)
        v1 = self.__get_value__(v0+10, 200)

        odis0 = self.__get_odis__()

        # eq = '%d + %d' % (v0, v1)
        eq = ''
        ans = 0
        for i in range(v0, v1+1):
            if odis0 == '홀수' and i % 2 == 1:
                ans += i
                eq += str(i) + '+'
            elif odis0 == '짝수' and i % 2 == 0:
                ans += i
                eq += str(i) + '+'

        eq = eq[:-1]

        # p = np.random.rand()
        # if p < 0.34:
        que = '%d부터 %d까지의 %s의 합을 구하시오.' \
                % (v0, v1, odis0)
        return que, eq, ans

    def __get_problem01_03__(self):
        v0 = self.__get_value__(1, 10)
        v1 = self.__get_value__(2, 10)

        food = self.__get_food__()

        eq = '%d * %d' % (v0, v1)
        ans = '%d' % (v0 * v1)

        p = np.random.rand()
        if p < 0.34:
            que = pyjosa.replace_josa(
                u"한 상자에는 {0}(이)가 {1}개씩 들어있습니다. {2}개의 상자 안에 있는 {0}(은)는 모두 몇 개일까요?".format(food, v0, v1))
        elif p < 0.67:
            que = pyjosa.replace_josa(
                u"한 상자에 {0}(이)가 {1}개씩 들어있으면, {2}개의 상자 안에 있는 {0}(은)는 모두 몇 개인지 구하시오.".format(food, v0, v1))
        else:
            que = pyjosa.replace_josa(
                u"한 상자에 {0}(이)가 {1}개씩 들어있을 때, {2}개의 상자 안에 있는 {0}(은)는 모두 몇 개일까?".format(food, v0, v1))

        return que, eq, ans

    def __get_problem01_04__(self):
        p = np.random.rand()
        while (1):
            v0 = self.__get_value__(40, 100)
            v1 = self.__get_value__(40, 100)
            v2 = self.__get_value__(40, 100)
            v3 = self.__get_value__(40, 100)
            avg = self.__get_value__(40, 95)
            total = self.__get_value__(15, 50)  # 몇 명?

            n0, n1, n2, n3 = self.__get_name4_2__()

            subject = self.__get_subject__()

            if p < 0.34:  # 인원 2명
                ans = (avg * (total - 2) + v0 + v1) / total
                if int(ans) == ans:
                    eq = '(%d*(%d-2)+%d+%d)/%d' % (avg, total, v0, v1, total)

                    if n0 in ['윤기', '은지', '유나'] and n1 in ['윤기', '은지', '유나']:
                        que = '%s, %s의 %s 점수는 각각 %d점, %d점입니다. 이 둘을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점입니까?' \
                              % (n0, n1, subject, v0, v1, subject, avg, n0, total, subject)
                    elif n0 in ['윤기', '은지', '유나'] and n1 not in ['윤기', '은지', '유나']:
                        que = '%s, %s이의 %s 점수는 각각 %d점, %d점입니다. 이 둘을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점입니까?' \
                              % (n0, n1, subject, v0, v1, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n1 in ['윤기', '은지', '유나']:
                        que = '%s, %s의 %s 점수는 각각 %d점, %d점입니다. 이 둘을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점입니까?' \
                              % (n0, n1, subject, v0, v1, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n1 not in ['윤기', '은지', '유나']:
                        que = '%s, %s이의 %s 점수는 각각 %d점, %d점입니다. 이 둘을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점입니까?' \
                              % (n0, n1, subject, v0, v1, subject, avg, n0, total, subject)

                    break
                else:
                    eq = ''

            elif p < 0.67:  # 인원 3명
                ans = (avg * (total - 3) + v0 + v1 + v2) / total
                if int(ans) == ans:
                    eq = '(%d*(%d-3)+%d+%d+%d)/%d' % (avg, total, v0, v1, v2, total)
                    if n0 in ['윤기', '은지', '유나'] and n2 in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s의 %s 점수는 각각 %d점, %d점, %d점입니다. 이 셋을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점인지 구하시오.' \
                              % (n0, n1, n2, subject, v0, v1, v2, subject, avg, n0, total, subject)
                    elif n0 in ['윤기', '은지', '유나'] and n2 not in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s이의 %s 점수는 각각 %d점, %d점, %d점입니다. 이 셋을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점인지 구하시오.' \
                              % (n0, n1, n2, subject, v0, v1, v2, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n2 in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s의 %s 점수는 각각 %d점, %d점, %d점입니다. 이 셋을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점인지 구하시오.' \
                              % (n0, n1, n2, subject, v0, v1, v2, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n2 not in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s이의 %s 점수는 각각 %d점, %d점, %d점입니다. 이 셋을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점인지 구하시오.' \
                              % (n0, n1, n2, subject, v0, v1, v2, subject, avg, n0, total, subject)
                    break
                else:
                    eq = ''

            else:  # 인원 4명
                ans = (avg * (total - 4) + v0 + v1 + v2 + v3) / total
                if int(ans) == ans:
                    eq = '(%d*(%d-4)+%d+%d+%d+%d)/%d' % (avg, total, v0, v1, v2, v3, total)
                    if n0 in ['윤기', '은지', '유나'] and n3 in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s, %s의 %s 점수는 각각 %d점, %d점, %d점, %d점입니다. 이 넷을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점일까요?' \
                              % (n0, n1, n2, n3, subject, v0, v1, v2, v3, subject, avg, n0, total, subject)
                    elif n0 in ['윤기', '은지', '유나'] and n3 not in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s, %s이의 %s 점수는 각각 %d점, %d점, %d점, %d점입니다. 이 넷을 제외한 학급의 %s 점수 평균은 %d점입니다. %s네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점일까요?' \
                              % (n0, n1, n2, n3, subject, v0, v1, v2, v3, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n3 in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s, %s의 %s 점수는 각각 %d점, %d점, %d점, %d점입니다. 이 넷을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점일까요?' \
                              % (n0, n1, n2, n3, subject, v0, v1, v2, v3, subject, avg, n0, total, subject)
                    elif n0 not in ['윤기', '은지', '유나'] and n3 not in ['윤기', '은지', '유나']:
                        que = '%s, %s, %s, %s이의 %s 점수는 각각 %d점, %d점, %d점, %d점입니다. 이 넷을 제외한 학급의 %s 점수 평균은 %d점입니다. %s이네 학급 인원수가 %d명일 때, 학급 %s 평균 점수는 몇 점일까요?' \
                              % (n0, n1, n2, n3, subject, v0, v1, v2, v3, subject, avg, n0, total, subject)
                    break
                else:
                    eq = ''

        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    ''' 유형2 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem02_01__(self):
        v0 = self.__get_value__(5, 100)
        v1 = self.__get_value__(1, v0 - 2)

        n0 = self.__get_name__()

        person = self.__get_person__()

        eq = '(%d - 1) - %d' % (v0, v1)
        ans = '%d' % (v0 - v1 - 1)

        p = np.random.rand()
        if p < 0.34:
            que = '%d명의 %s들이 한 줄로 줄을 섰습니다. %s 앞에 %d명의 %s들이 서 있습니다. %s의 뒤에 서 있는 %s은 몇 명입니까?' \
                  % (v0, person, n0, v1, person, n0, person)
        elif p < 0.67:
            que = '%d명의 %s들이 한 줄로 줄을 섰고, %s 앞에 %d명의 %s들이 서 있습니다. %s의 뒤에 서 있는 %s은 몇 명일까요?' \
                  % (v0, person, n0, v1, person, n0, person)
        else:
            que = '%d명의 %s들이 한 줄로 줄을 섰습니다. %s 앞에 %d명의 %s들이 서 있다면, %s의 뒤에 서 있는 %s은 몇 명인지 구하시오.' \
                  % (v0, person, n0, v1, person, n0, person)

        return que, eq, ans

    def __get_problem02_02__(self):
        v0 = self.__get_value__(1, 50)
        v1 = v0 + 2

        n0, n1, n2 = self.__get_name3__()

        sport = self.__get_sport__()

        eq = '(%d + %d) / 2' % (v0, v1)
        ans = int('%d' % (int((v0 + v1) / 2)))

        p = np.random.rand()
        if p < 0.25:
            que = '%s 시합에서 %s는 %d등을 했고, %s는 %d등을 했습니다. %s는 %s보다 잘했지만 %s보다는 못했습니다. %s의 등수는 몇 등입니까?' \
                  % (sport, n0, v0, n1, v1, n2, n1, n0, n2)
        elif p < 0.5:
            que = '%s 시합에서 %s는 %d등을 했고, %s는 %d등을 했습니다. %s는 %s보다 잘했지만 %s보다는 못했다면, %s의 등수는 몇 등일까요?' \
                  % (sport, n0, v0, n1, v1, n2, n1, n0, n2)
        elif p < 0.75:
            que = '%s 시합에서 %s는 %d등을 했고, %s는 %d등을 했습니다. %s는 %s보다 못했지만 %s보다는 잘했습니다. %s의 등수는 몇 등인지 구하시오.' \
                  % (sport, n0, v0, n1, v1, n2, n0, n1, n2)
        else:
            que = '%s 시합에서 %s는 %d등을 했고, %s는 %d등을 했습니다. %s는 %s보다 못했지만 %s보다는 잘했다면, %s의 등수는 몇 등입니까?' \
                  % (sport, n0, v0, n1, v1, n2, n0, n1, n2)

        return que, eq, ans

    def __get_problem02_03__(self):
        v0 = self.__get_value__(5, 30)
        v1 = self.__get_value__(1, v0)
        o1 = self.__get_ordinal__(v1 - 1)

        n0 = self.__get_name__()

        person = self.__get_person__()

        eq = '%d - %d + 1' % (v0, v1)
        ans = '%d' % (v0 - v1 + 1)

        p = np.random.rand()
        if p < 0.25:
            que = '키가 작은 %s부터 순서대로 %d명이 한 줄로 서 있습니다. %s가 앞에서부터 %s 번째에 서 있습니다. 키가 큰 사람부터 순서대로 다시 줄을 서면 %s는 앞에서부터 몇 번째에 서게 됩니까?' \
                  % (person, v0, n0, o1, n0)
        elif p < 0.5:
            que = '키가 작은 %s부터 순서대로 %d명이 한 줄로 서 있고, %s가 앞에서부터 %s 번째에 서 있습니다. 키가 큰 사람부터 순서대로 다시 줄을 서면 %s는 앞에서부터 몇 번째에 서게 되는지 구하시오.' \
                  % (person, v0, n0, o1, n0)
        elif p < 0.75:
            que = '키가 큰 %s부터 순서대로 %d명이 한 줄로 서 있습니다. %s가 앞에서부터 %s 번째에 서 있습니다. 키가 작은 사람부터 순서대로 다시 줄을 서면 %s는 앞에서부터 몇 번째에 서게 됩니까?' \
                  % (person, v0, n0, o1, n0)
        else:
            que = '키가 큰 %s부터 순서대로 %d명이 한 줄로 서 있고, %s가 앞에서부터 %s 번째에 서 있습니다. 키가 작은 사람부터 순서대로 다시 줄을 서면 %s는 앞에서부터 몇 번째에 서게 될까요?' \
                  % (person, v0, n0, o1, n0)

        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    ''' 유형4 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem04_01__(self):
        v = self.__get_value__(2, 6)

        float_prob = np.random.rand()
        if float_prob < 0.8:
            str_vs, vs = self.__get_value_array__(0, 100, v)
        else:
            str_vs, vs = self.__get_value_array__(0, 100, v, dtype='float')

        o1, o1_idx = self.__get_ordinal2__(np.min([v, 4]))
        o2, o2_idx = self.__get_ordinal2__(np.min([v, 4]))

        vs_sorted = np.sort(vs)
        v1 = vs_sorted[-o1_idx + 1]
        v2 = vs_sorted[o2_idx]

        op, op_type = self.__get_operation__()

        if float_prob < 0.8:
            if op_type == 'add':
                eq = '%d + %d' % (v1, v2)
                ans = '%d' % (v1 + v2)
            elif op_type == 'sub':
                eq = '%d - %d' % (v1, v2)
                ans = '%d' % (v1 - v2)
            elif op_type == 'mul':
                eq = '%d * %d' % (v1, v2)
                ans = '%d' % (v1 * v2)
            elif op_type == 'avg':
                eq = '(%d + %d) / 2' % (v1, v2)
                ans = '%d' % ((v1 + v2) / 2)
        else:
            if op_type == 'add':
                eq = '%.2f + %.2f' % (v1, v2)
                ans = '%.2f' % (v1 + v2)
            elif op_type == 'sub':
                eq = '%.2f - %.2f' % (v1, v2)
                ans = '%.2f' % (v1 - v2)
            elif op_type == 'mul':
                eq = '%.2f * %.2f' % (v1, v2)
                ans = '%.2f' % (v1 * v2)
            elif op_type == 'avg':
                eq = '(%.2f + %.2f) / 2' % (v1, v2)
                ans = '%.2f' % ((v1 + v2) / 2)

        p = np.random.rand()
        if p < .25:
            que = '%d개의 %s 수가 있습니다. 그 중에서 %s 큰 수와 %s 작은 수의 %s 얼마입니까?' \
                  % (v, str_vs, o1, o2, op)
        elif p < .5:
            que = '전체 %d개의 %s 숫자가 주어졌습니다. 그 중에서 %s 큰 숫자와 %s 작은 숫자의 %s 얼마입니까?' \
                  % (v, str_vs, o1, o2, op)
        elif p < .75:
            que = '총 %d개의 %s 숫자가 놓여있습니다. 그 중에서 %s 큰 수와 %s 작은 수의 %s 얼마입니까?' \
                  % (v, str_vs, o1, o2, op)
        else:
            que = '모두 %d개의 %s 수가 있습니다. 그 중에서 %s 큰 숫자와 %s 작은 숫자의 %s 얼마입니까?' \
                  % (v, str_vs, o1, o2, op)
        return que, eq, ans

    def __get_problem04_02__(self):
        pos = self.__get_value__(1, 2)
        pos_str = self.__get_pos_str__(pos)
        v = self.__get_float_value__(1, 50)
        s1, s2, op = self.__get_shift__()

        if op == 'increase':
            # eq = '%.2f / %d' % (v, (10 ** pos - 1))
            eq = '%.2f / (%d - 1)' % (v, (10 ** pos))
            ans = '%.2f' % (v / (10 ** pos - 1))
        else:
            eq = '%.2f / (%.2f - 1)' % (v, -(10 ** (-pos)))
            ans = '%.2f' % (v / -(10 ** (-pos) - 1))

        p = np.random.rand()

        if p < 1 / 3:
            que = '어떤 소수의 소수점을 %s쪽으로 %s 자리 옮기면 원래보다 %.2f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        elif p < 2 / 3:
            que = '어떤 수의 소수점을 %s쪽으로 %s 자리 옮기면 원래보다 %.2f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        else:
            que = '알 수 없는 소수의 소수점을 %s쪽으로 %s 자리 옮기면 원래보다 %.2f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        return que, eq, ans

    # def __get_problem04_03__(self):
    #     v = self.__get_value__(2, 5)
    #     str_vs, vs = self.__get_value_array__(0, 15, v)
    #     v_str = self.__get_pos_str__(v)
    #
    #     # eq = '%d // math.lcm - / %d' %(v, (10**pos - 1))
    #     # ans = '%.2f' %(v / (10**pos - 1))
    #
    #     p = np.random.rand()
    #     if p  <  1/3:
    #         que = '총 %d개의 수 %s로 나누어떨어지는 %s 자리 수는 모두 몇 개 있습니까?' \
    #               % (v, str_vs, v_str)
    #     elif p < 2/3:
    #         que = '모두 %d개의 수 %s로 나누어 떨어질 수 있는 %s 자리 수는 총 몇 개 있니까?' \
    #               % (v, str_vs, v_str)
    #     else:
    #         que = '%d 개의 수 %s로 나누어 떨어질 수 있는 %s 자리 수는 모두 몇 개 있습니까?' \
    #               % (v, str_vs, v_str)
    #     return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


if __name__ == '__main__':
    mwp_dataset = MWPDataset()
    data_loader = DataLoader(mwp_dataset, batch_size=1, shuffle=False, num_workers=0)

    question_list, equation_list, answer_list = [], [], []
    for batch_idx, samples in tqdm.tqdm(enumerate(data_loader)):
        que, eq, ans = samples
        if torch.is_tensor(ans) == True:
            ans = ans.detach().cpu().numpy()

        question_list.append(que[0])
        equation_list.append(eq[0])
        #answer_list.append(ans.detach().cpu().numpy())
        answer_list.append(ans[0])

    df = pd.DataFrame({'Question': question_list,
                       'Equation': equation_list,
                       #'Answer': answer_list})
                       'Answer' : np.array(answer_list).reshape(-1)})
    df.to_csv('train.csv', index=False, encoding='euc-kr')

