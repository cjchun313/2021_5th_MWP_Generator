import numpy as np
import random
import tqdm
import math
import pandas as pd
import itertools
from math_custom import *
import pyjosa
import re

from fractions import Fraction
from decimal import Decimal

import torch
from torch.utils.data import Dataset, DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


class MWPDataset(Dataset):
    def __init__(self, seed=0, max_len=100):
        self.seed = seed
        self.max_len = max_len

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(self.seed)

        '''
        self.problems = {'1-1': self.__get_problem01_01__,
                         '1-2': self.__get_problem01_02__,
                         '1-3': self.__get_problem01_03__,
                         '1-4': self.__get_problem01_04__,
                         '2-1': self.__get_problem02_01__,
                         '2-2': self.__get_problem02_02__,
                         '2-3': self.__get_problem02_03__,
                         '3-1': self.__get_problem03_01__,
                         '3-2': self.__get_problem03_02__,
                         '4-1': self.__get_problem04_01__,
                         '4-2': self.__get_problem04_02__,
                         '4-3': self.__get_problem04_03__,
                         '6-1': self.__get_problem06_01__,
                         '6-3': self.__get_problem06_03__,
                         '6-4': self.__get_problem06_04__,
                         '8-1': self.__get_problem08_01__,
                         '8-2': self.__get_problem08_02__,
                         '8-3': self.__get_problem08_03__,
                         '9-1': self.__get_problem09_01__,
                         '9-2': self.__get_problem09_02__,
                         '9-3': self.__get_problem09_03__,}
        '''
        self.problems = {'6-1': self.__get_problem06_01__,
                         '6-3': self.__get_problem06_03__,
                         '6-4': self.__get_problem06_04__,
                         '7-3': self.__get_problem07_03__
                         }
        
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

    def __get_value3__(self, min, max):
        num_list = list(range(min, max))
        return random.sample(num_list, 3)

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
    def __get_name2_2__(self):
        name = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
        n0, n1 = random.sample(name, 2)
        return n0, n1

    ''' 중복안되게 이름 세개 임의적으로 추출 '''
    def __get_name3__(self):
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        n0, n1, n2 = random.sample(name, 3)
        return n0, n1, n2
    
    def __get_name3_2__(self):
        name = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
        n0, n1, n2 = random.sample(name, 4)
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
    
    ''' 중복안되게 이름 네개 임의적으로 추출 '''
    def __get_name_n__(self, num):
        name = ['남준이', '석진이', '윤기', '호석이', '지민이', '태형이', '정국이', '민영이', '유정이', '은지', '유나']
        return np.random.choice(name, num, replace = False)

    def __get_name_n_2__(self, num):
        name = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
        return np.random.choice(name, num, replace = False)
    
    def __get_ordinal__(self, idx):
        # number = ['첫', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '열한', '열두', '열세', '열네', '열다섯', \
        #           '열여섯', '열일곱', '열여덟', '열아홉', '스무', '스물한', '스물두', '스물세', '스물네', '스물다섯', '스물여섯', \
        #           '스물일곱', '스물여덟', '스물아홉', '서른']
        number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', \
                  '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
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

    def __get_fruit1__(self):
        name = ['사과', '배', '감', '귤', '포도', '수박', '참외', '복숭아', '딸기', '오렌지', '자두', '체리', '바나나', '레몬', '키위', '살구', '파일애플',
                '석류', '매실']
        n0 = random.sample(name, 1)
        return n0[0]

    def __get_fruit8__(self):
        name = ['사과', '배', '감', '귤', '포도', '수박', '참외', '복숭아', '딸기', '오렌지', '자두', '체리', '바나나', '레몬', '키위', '살구', '파일애플',
                '석류', '매실']
        n0, n1, n2, n3, n4, n5, n6, n7 = random.sample(name, 8)
        return n0, n1, n2, n3, n4, n5, n6, n7

    def __get_animal__(self):
        subject = ['오리', '닭', '토끼', '물고기', '고래', '거위', '개구리', '강아지', '고양이', '비둘기', '병아리', '원숭이', '코끼리', '양', '소', '돼지',
                   '쥐']
        idx = np.random.randint(0, len(subject))
        return subject[idx]

    def my_nCr(self, n, r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)

    def __get_cal__(self):
        cal_list = ['합', '차']
        idx = np.random.randint(0, len(cal_list))
        return cal_list[idx]

    def __get_subject__(self):
        subject = ["국어","영어","수학","사회","과학","음악","미술","체육"]
        idx = np.random.randint(0, len(subject))
        return subject[idx]

    def __get_color__(self):
        color = ['빨간', '주황', '노란', '초록', '파란', '흰', '검은', '보라']
        idx = np.random.randint(0, len(color))
        return color[idx]
    
    def __get_colorname__(self):
        color = ['빨간색', '주황색', '노란색', '초록색', '파란색', '흰색', '검은색', '보라색']
        idx = np.random.randint(0, len(color))
        return color[idx]
    
    def __get_kor_bracket_seq__(self, num):
        kor = ['(가)', '(나)', '(다)', '(라)', '(마)', '(바)', '(사)', '(아)', '(자)', '(차)', '(카)', '(타)', '(파)', '(하)']
        str_val = ''
        for idx in range(num - 1):
            str_val += '%s, ' % kor[idx]
        str_val += '%s' % kor[num - 1]
        return kor[:num], str_val
    
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
                    str_val = '%.02f, ' % np_array[idx]
                str_array += str_val

            if np.random.rand() <= frac_prob:
                val = '%.02f' % np_array[-1]
                val = Fraction(Decimal(val))
                str_val = '%d/%d' % (val.numerator, val.denominator)
            else:
                str_val = '%.02f' % np_array[-1]

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
        if idx <= 1:
            ord_idx = 0
        elif idx == 2:
            ord_idx = 1
        elif idx == 3:
            ord_idx = 2
        elif idx == 4:
            ord_idx = 3
        return name[idx], ord_idx

    def __get_pos_str__(self, pos):
        if pos == 1:
            name = '1'
        elif pos == 2:
            name = '2'
        elif pos == 3:
            name = '3'
        elif pos == 4:
            name = '4'
        elif pos == 5:
            name = '5'
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

    def __get_sample_value__(self, min, max):
        num_list = list(range(min, max))
        return random.sample(num_list, 2)

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



















    ''' 유형3 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem03_01__(self):

        fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7 = self.__get_fruit8__()

        # eq = '%d × %d' % (v0, v1)
        # ans = v0 * v1

        p = np.random.rand()
        if p < 0.2:
            v0 = self.__get_value__(2, 3)
            que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3} 중에서 {4}가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, v0))
            form_que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3} 중에서 n0가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, v0))
            eq = '4C{}'.format(v0)
            form = '4 C n0'
            ans = len(list(itertools.combinations([fr0, fr1, fr2, fr3],v0)))

        elif p < 0.4:
            v0 = self.__get_value__(2, 4)
            que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4} 중에서 {5}가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, v0))
            form_que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4} 중에서 n0가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, v0))
            eq = '5C{}'.format(v0)
            form = '5 C n0'
            ans = len(list(itertools.combinations([fr0, fr1, fr2, fr3, fr4],v0)))

        elif p < 0.6:
            v0 = self.__get_value__(2, 5)
            que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5} 중에서 {6}가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, v0))
            form_que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5} 중에서 n0가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, v0))
            eq = '6C{}'.format(v0)
            form = '6 C n0'
            ans = len(list(itertools.combinations([fr0, fr1, fr2, fr3, fr4, fr5],v0)))

        elif p < 0.8:
            v0 = self.__get_value__(2, 6)
            que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5}, {6} 중에서 {7}가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, fr6,v0))
            form_que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5}, {6} 중에서 n0가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, fr6,v0))
            eq = '7C{}'.format(v0)
            form = '7 C n0'
            ans = len(list(itertools.combinations([fr0, fr1, fr2, fr3, fr4, fr5, fr6],v0)))

        else:
            v0 = self.__get_value__(2, 7)
            que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} 중에서 {8}가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, v0))
            form_que = pyjosa.replace_josa(u"{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7} 중에서 n0가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?".format(fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, v0))
            eq = '8C{}'.format(v0)
            form = '8 C n0'
            ans = len(list(itertools.combinations([fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7],v0)))

        #return que, eq, form, form_que, ans
        return que, eq, ans

    def __get_problem03_02__(self):

        while (1):
            fr0 = self.__get_fruit1__()
            v0 = self.__get_value__(5, 20)

            while (1):
                v1 = self.__get_value__(2, 10)
                if v1 <= v0 + 1:
                    break

            onetwo_list = ['한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열']

            animal0 = self.__get_animal__()

            v2 = self.__get_value__(1, 5)

            # eq = '%d × %d' % (v0, v1)
            # ans = v0 * v1

            p = np.random.rand()
            if p < 0.2:
                change_v1 = onetwo_list[v1 - 1]
                que = pyjosa.replace_josa(
                    u"{0} {1}개를 서로 다른 {2} 마리의 {3}에게 나누어 주려고 합니다. {3}(은)는 적어도 {0} {4}개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 가지 입니까?".format(
                        fr0, \
                        v0, change_v1, animal0, v2))
                form_que = pyjosa.replace_josa(
                    u"{0} n0개를 서로 다른 n1 마리의 {3}에게 나누어 주려고 합니다. {3}(은)는 적어도 {0} n2개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 가지 입니까?".format(
                        fr0, \
                        v0, change_v1, animal0, v2))
                eq = '{}C{}'.format((v0 - v1 * v2 + 1), v1 - 1)
                if (v0 - v1 * v2) > v1 + 1:
                    ans = self.my_nCr((v0 - v1 * v2 + 1), v1 - 1)
                    break

            elif p < 0.4:
                change_v1 = onetwo_list[v1 - 1]
                que = pyjosa.replace_josa(
                    u"{0} {1}개를 서로 다른 {2} 마리의 {3}에게 나누어 주려고 합니다. {3}(은)는 적어도 {0} {4}개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 가지 일까요?".format(
                        fr0, \
                        v0, change_v1, animal0, v2))
                form_que = pyjosa.replace_josa(
                    u"{0} n0개를 서로 다른 n1 마리의 {3}에게 나누어 주려고 합니다. {3}(은)는 적어도 {0} n2개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 가지 일까요?".format(
                        fr0, \
                        v0, change_v1, animal0, v2))
                eq = '{}C{}'.format((v0 - v1 * v2 + 1), v1 - 1)
                if (v0 - v1 * v2) > v1 + 1:
                    ans = self.my_nCr((v0 - v1 * v2 + 1), v1 - 1)
                    break

            elif p < 0.6:
                change_v1 = onetwo_list[v1 - 1]
                que = pyjosa.replace_josa(
                    u"{0} {1}개를 서로 다른 {2} 마리의 {3}에게 나눠줍니다. {3}(은)는 최소 {0} {4}개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                form_que = pyjosa.replace_josa(
                    u"{0} n0개를 서로 다른 n1 마리의 {3}에게 나눠줍니다. {3}(은)는 최소 {0} n2개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                eq = '{}C{}'.format((v0 - v1 * v2 + 1), v1 - 1)
                if (v0 - v1 * v2) > v1 + 1:
                    ans = self.my_nCr((v0 - v1 * v2 + 1), v1 - 1)
                    break

            elif p < 0.8:
                change_v1 = onetwo_list[v1 - 1]
                que = pyjosa.replace_josa(
                    u"{0} {1}개를 서로 다른 {2} 마리의 {3}에게 나눠줍니다. {3}(은)는 적어도 {0} {4}개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                form_que = pyjosa.replace_josa(
                    u"{0} n0개를 서로 다른 n1 마리의 {3}에게 나눠줍니다. {3}(은)는 적어도 {0} n2개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                eq = '{}C{}'.format((v0 - v1 * v2 + 1), v1 - 1)
                if (v0 - v1 * v2) > v1 + 1:
                    ans = self.my_nCr((v0 - v1 * v2 + 1), v1 - 1)
                    break

            else:
                change_v1 = onetwo_list[v1 - 1]
                que = pyjosa.replace_josa(
                    u"{0} {1}개를 서로 다른 {2} 마리의 {3}에게 나눠줍니다. {3}(은)는 적어도 {0} {4}개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                form_que = pyjosa.replace_josa(
                    u"{0} n0개를 서로 다른 n1 마리의 {3}에게 나눠줍니다. {3}(은)는 적어도 {0} n2개는 받습니다. {0}(을)를 나누어 주는 방법은 모두 몇 개 일까요?".format(
                        fr0, v0, change_v1, animal0, v2))
                eq = '{}C{}'.format((v0 - v1 * v2 + 1), v1 - 1)
                if (v0 - v1 * v2) > v1 + 1:
                    ans = self.my_nCr((v0 - v1 * v2 + 1), v1 - 1)
                    break
        form = "( n0 - n1 * n2 + 1 ) C ( n1 - 1)"
        #return que, eq, form, form_que, ans
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
        v1 = vs_sorted[v - 1 - o1_idx]
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
                ans = '%.02f' % ((v1 + v2) / 2)
        else:
            if op_type == 'add':
                eq = '%.02f + %.02f' % (v1, v2)
                ans = '%.02f' % (v1 + v2)
            elif op_type == 'sub':
                eq = '%.02f - %.02f' % (v1, v2)
                ans = '%.02f' % (v1 - v2)
            elif op_type == 'mul':
                eq = '%.02f * %.02f' % (v1, v2)
                ans = '%.02f' % (v1 * v2)
            elif op_type == 'avg':
                eq = '(%.02f + %.02f) / 2' % (v1, v2)
                ans = '%.02f' % ((v1 + v2) / 2)

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
        pos = self.__get_value__(1, 3) # 최대 2자리 이동
        pos_str = str(pos)
        v = self.__get_float_value__(1, 50)
        # refine float value
        v = ((v * 100.) // (10 ** pos - 1)) * (10 ** pos - 1) / 100.
        s1, s2, op = self.__get_shift__()

        if op == 'increase':
            # eq = '%.02f / %d' % (v, (10 ** pos - 1))
            eq = '%.02f / (%d - 1)' % (v, (10 ** pos))
            ans = '%.02f' % (v / (10 ** pos - 1))
        else:
            eq = '%.02f / (%.02f - 1)' % (v, -(10 ** (-pos)))
            ans = '%.02f' % (v / -(10 ** (-pos) - 1))

        p = np.random.rand()

        if p < 1 / 3:
            que = '어떤 소수의 소수점을 %s쪽으로 %s자리 옮기면 원래보다 %.02f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        elif p < 2 / 3:
            que = '어떤 수의 소수점을 %s쪽으로 %s자리 옮기면 원래보다 %.02f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        else:
            que = '알 수 없는 소수의 소수점을 %s쪽으로 %s자리 옮기면 원래보다 %.02f만큼 %s집니다. 원래의 소수를 구하시오.' \
                  % (s1, pos_str, v, s2)
        return que, eq, ans

    def __get_problem04_03__(self):
        v = self.__get_value__(2, 5)
        str_vs, vs = self.__get_value_array__(1, 10, v)
        
        n_digit = self.__get_value__(2, 4)
        n_digit_str = self.__get_pos_str__(n_digit)
    
        eq = '(%d - 1) // LCM([%s]) - (%d - 1) // LCM([%s])' %(10**(n_digit), str_vs, 10**(n_digit - 1), str_vs)
        ans = '%d' %((10**(n_digit) - 1)//lcm(vs) - (10**(n_digit - 1) - 1)//lcm(vs))
    
        p = np.random.rand()
        if p  <  1/3:
            que = '총 %d개의 수 %s로 나누어떨어지는 %s자리 수는 모두 몇 개 있습니까?' \
                  % (v, str_vs, n_digit_str)
        elif p < 2/3:
            que = '모두 %d개의 수 %s로 나누어 떨어질 수 있는 %s자리 수는 총 몇 개 있니까?' \
                  % (v, str_vs, n_digit_str)
        else:
            que = '%d 개의 수 %s로 나누어 떨어질 수 있는 %s자리 수는 모두 몇 개 있습니까?' \
                  % (v, str_vs, n_digit_str)
        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''




    ''' 유형6 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem06_01__(self):
        while True:
            v0, v1, v2 = self.__get_value3__(1, 100)
            calculation0 = self.__get_cal__()

            p = np.random.rand()
            if p < 0.5:
                if calculation0 == '합':
                    eq = f"{v2} - {v1} + {v0}"
                    form = "n2 - n1 + n0"
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더해야 하는데 잘못하여 {1}(을)를 더한 결과가 {2}(이)가 나왔습니다. 바르게 계산한 결과를 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더해야 하는데 잘못하여 {1}(을)를 더한 결과가 {2}(이)가 나왔다면, 바르게 계산한 결과는 몇 일까요?".format(v0, v1, v2))
                    ans = v2-v1+v0
                elif calculation0 == '차':
                    eq = f"{v2} + {v1} - {v0}"
                    form = "n2 + n1 - n0"
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 빼야 하는데 잘못하여 {1}(을)를 뺀 결과가 {2}(이)가 나왔습니다. 바르게 계산한 결과를 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 빼야 하는데 잘못하여 {1}(을)를 뺀 결과가 {2}(이)가 나왔다면, 바르게 계산한 결과는 몇 일까요?".format(v0, v1, v2))
                    ans = v2+v1-v0
            else:
                if calculation0 == '합':
                    eq = f"{v2} - {v1} + {v0}"
                    form = "n2 - n1 + n0"
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"모르는 수에서 {0}(을)를 더해야 하는데 잘못하여 {1}(을)를 더한 결과가 {2}(이)가 나왔습니다. 바르게 계산한 결과를 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"모르는 수에서 {0}(을)를 더해야 하는데 잘못하여 {1}(을)를 더한 결과가 {2}(이)가 나왔다면, 바르게 계산한 결과는 몇 일까요?".format(v0, v1, v2))
                    ans = v2-v1+v0
                elif calculation0 == '차':
                    eq = f"{v2} + {v1} - {v0}"
                    form = "n2 + n1 - n0"
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"모르는 수에서 {0}(을)를 빼야 하는데 잘못하여 {1}(을)를 뺀 결과가 {2}(이)가 나왔습니다. 바르게 계산한 결과를 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"모르는 수에서 {0}(을)를 빼야 하는데 잘못하여 {1}(을)를 뺀 결과가 {2}(이)가 나왔다면, 바르게 계산한 결과는 몇 일까요?".format(v0, v1, v2))
                    ans = v1+v2-v0
            if ans >= 1:
                break

        '''
        print('que : {}'.format(que))
        print('form : {}'.format(form))
        print('eq : {}'.format(eq))
        print('ans : {}'.format(ans))
        '''

        #return que, eq, form, ans
        return que, eq, ans

    def __get_problem06_03__(self):
        calculation0 = self.__get_cal__()
        calculation1 = self.__get_cal__()
        while True:
            v0, v1, v2 = self.__get_value3__(1, 100)

            if calculation0 == "합":
                if calculation1 == "합":
                    eq = f"{v1} - {v0} + {v2}"
                    form = "n1 - n0 + n2"
                    ans = v1-v0+v2
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더했더니 {1}(이)가 되었습니다. 어떤 수에서 {2}(을)를 더하면 얼마가 되는지 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더했더니 {1}(이)가 되었다면, 어떤 수에서 {2}(을)를 더하면 얼마가 되겠습니까?".format(v0, v1, v2))
                elif calculation1 == "차":
                    eq = f"{v1} - {v0} - {v2}"
                    form = "n1 - n0 - n2"
                    ans = v1-v0-v2
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더했더니 {1}(이)가 되었습니다. 어떤 수에서 {2}(을)를 빼면 얼마가 되는지 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 더했더니 {1}(이)가 되었다면, 어떤 수에서 {2}(을)를 빼면 얼마가 되겠습니까?".format(v0, v1, v2))
                if ans > 0:
                    break
            elif calculation0 == "차":
                if calculation1 == "합":
                    eq = f"{v1} + {v0} + {v2}"
                    form = "n1 + n0 + n2"
                    ans = v1+v0+v2
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 뺏더니 {1}(이)가 되었습니다. 어떤 수에서 {2}(을)를 더하면 얼마가 되는지 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 뺏더니 {1}(이)가 되었다면, 어떤 수에서 {2}(을)를 더하면 얼마가 되겠습니까?".format(v0, v1, v2))
                elif calculation1 == "차":
                    eq = f"{v1} + {v0} - {v2}"
                    form = "n1 + n0 - n2"
                    ans = v1+v0-v2
                    p = np.random.rand()
                    if p < 0.5:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 뺏더니 {1}(이)가 되었습니다. 어떤 수에서 {2}(을)를 빼면 얼마가 되는지 구하시오.".format(v0, v1, v2))
                    else:
                        que = pyjosa.replace_josa(u"어떤 수에서 {0}(을)를 뺏더니 {1}(이)가 되었다면, 어떤 수에서 {2}(을)를 빼면 얼마가 되겠습니까?".format(v0, v1, v2))
                if ans > 0:
                    break

        '''
        print('que : {}'.format(que))
        print('form : {}'.format(form))
        print('eq : {}'.format(eq))
        print('ans : {}'.format(ans))
        '''

        #return que, eq, form, int(ans)
        return que, eq, int(ans)

    def __get_problem06_04__(self):
        calculation0 = self.__get_cal__()
        while True:
            v0, v1, v2 = self.__get_value3__(2, 11)

            if calculation0 == "합":
                p = np.random.rand()
                if p < 0.5:
                    que = pyjosa.replace_josa(u"{0}에 어떤 수를 더해야 할 것을 잘못하여 {1}에 어떤 수를 더했더니 {2}(이)가 되었습니다. 바르게 계산하면 얼마인지 구하시오.".format(v0, v1, v2))
                else:
                    que = pyjosa.replace_josa(u"{0}에 어떤 수를 더해야 할 것을 잘못하여 {1}에 어떤 수를 더했더니 {2}(이)가 되었다면, 바르게 계산하면 얼마일까요?".format(v0, v1, v2))
                form = "n2 - n1 + n0"
                eq = f"{v2} - {v1} + {v0}"
                #eq = f"X+{v1}={v2}, X+{v0}"
                ans = v2 - v1 + v0
            elif calculation0 == "차":
                p = np.random.rand()
                if p < 0.5:
                    que = pyjosa.replace_josa(u"{0}에 어떤 수를 빼야 할 것을 잘못하여 {1}에 어떤 수를 뺏더니 {2}(이)가 되었습니다. 바르게 계산하면 얼마인지 구하시오.".format(v0, v1, v2))
                else:
                    que = pyjosa.replace_josa(u"{0}에 어떤 수를 빼야 할 것을 잘못하여 {1}에 어떤 수를 뺏더니 {2}(이)가 되었다면, 바르게 계산하면 얼마일까요?".format(v0, v1, v2))
                form = "n2 + n1 - n0"
                eq = f"{v0} + {v2} - {v1}"
                #eq = f"X-{v1}={v2}, X-{v0}"
                ans = v0 + v2 - v1
            # elif calculation0 == "곱":
            #     que = f"{v0}에 어떤 수를 곱해야 할 것을 잘못하여 {v1}에 어떤 수를 곱했더니 {v1*v2}이 되었습니다. 바르게 계산하면 얼마인지 구하시오."
            #     form = "n2 / n1 * n0"
            #     eq = f"{v1}X={v1*v2}, {v0}X"
            #     ans = v2*v0

            if ans > 0:
                break

        '''
        print('que : {}'.format(que))
        print('form : {}'.format(form))
        print('eq : {}'.format(eq))
        print('ans : {}'.format(ans))
        '''

        #return que, eq, form, ans
        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    ''' 유형7 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem07_01__(self):
        while True:
            v0 = self.__get_value__(1, 10)
            v1 = self.__get_value__(1, 11)
            v2, v3 = self.__get_sample_value__(10, 50)

            r0 = "R" + str(v2)
            r1 = "R" + str(v3)
            
            que = f"{v0}, {v0+v1}, {v0+v1*2}, {v0+v1*3}, {v0+v1*4}와 같은 규칙에서 R{v2} 번째 놓일 수와 R{v3} 번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오."
            eq = f"[{v0}, {v0+v1}, {v0+v1*2}, {v0+v1*3}, {v0+v1*4}], X{v3} - X{v2}"
            ans  = (v3-v2)*v1

            if v2 < v3:
                break
        #print('[que] : {}'.format(que))
        #print('[eq] : {}'.format(eq))
        #print('[ans] : {}'.format(ans))

        return que, eq, ans

    def __get_problem07_02__(self):
        v0 = self.__get_value__(1, 100)
        v1 = self.__get_value__(1, 11)
        v2 = self.__get_value__(0, 6)
        # p = np.random.rand()
        # if p < 1/2:

        sequence_lst = [v0, v0+v1*1, v0+v1*2, v0+v1*3, v0+v1*4, v0+v1*5]
        
        sequence_lst.pop(v2)
        sequence_lst.insert(v2, "A")
        form_lst = ["n0", "n1", "n2", "n3", "n4", "n5"]
        form_lst.pop(v2)
        form_lst.insert(v2, "A")
        form = "[ "
        for idx, f in enumerate(form_lst):
            form += f
            if idx != len(form_lst) - 1:
                form += ", "
        else:
            form += " ]"
            
        que = '자연수를 규칙에 따라 {}로 배열하였습니다. A에 알맞은 수를 구하시오.'.format(re.sub("\'","",str(sequence_lst))[1:-1])
        eq = "[ " + re.sub("\'","",str(sequence_lst))[1:-1] + " ]"
        ans = v0+v1*v2
        print('[que] : {}'.format(que))
        print('[eq] : {}'.format(eq))
        print('[ans] : {}'.format(ans))
        return que, eq, ans

    def __get_problem07_03__(self):
        v0 = self.__get_value__(1, 10)
        v1 = self.__get_value__(7, 14)
        num = ['0', '첫', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '열한', '열두', '열세']
        # p = np.random.rand()
        # if p < 1/2:
        sequence_lst = [v0, v0+2**2, v0+2**2+3**2, v0+2**2+3**2+4**2, v0+2**2+3**2+4**2+5**2, v0+2**2+3**2+4**2+5**2+6**2]
        sequence_lst = str(sequence_lst)[1:-1]
        que = '{}과 같은 규칙으로 수를 배열하고 있습니다. R{} 번째 수는 무엇입니까?'.format(sequence_lst, v1)
        form_que = '{}과 같은 규칙으로 수를 배열하고 있습니다. R{} 번째 수는 무엇입니까?'.format('n0, n1, n2, n3, n4, n5', v1)
        # print(que)
        eq = sequence_lst + f", R{v1}"
        # print(eq)
        form = f'[ n0, n1, n2, n3, n4, n5 ] , R{v1}'
        # print(form)
        ans = v0
        for i in range(2, v1):
            ans += i**2
        
        print('que : {}'.format(que))
        print('eq : {}'.format(eq))
        print('ans : {}'.format(ans))

        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


    ''' 유형8 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem08_01__(self):
        n = self.__get_value__(5, 15)
        init = self.__get_value__(1, 5)
        interval = self.__get_value__(1, 5)
        
        ans_idx = self.__get_value__(10, 150)
        
        arr = list(range(init, init + 200))[::interval]
        arr = arr[:n]
        arr = sum(itertools.repeat(arr, 2), [])
        
        str_arr = ', '.join(str(e) for e in arr)
        eq = 'PATCOM(STRPAT(%s))[%d]' %(arr, ans_idx - 1)
        ans = '%d' %(patternize(find_str_pattern(arr))[ans_idx - 1])

        p = np.random.rand()
        if p < 0.25:
            que = '%s과 같이 반복되는 수열이 있습니다. 왼쪽에서 %d번째 숫자는 무엇입니까?' \
                  % (str_arr, ans_idx)
        elif p < .5:
            que = '%s과 같이 반복되는 숫자배열이 있습니다. 왼쪽에서 %d번째 숫자는 무엇입니까?' \
                  % (str_arr, ans_idx)
        elif p < .75:
            que = '%s과 같이 반복되는 수열이 있습니다. 왼쪽에서 %d번째 수는 무엇입니까?' \
                  % (str_arr, ans_idx)
        else:
            que = '%s과 같이 반복되는 문자열이 있습니다. 왼쪽에서 %d번째 문자는 무엇입니까?' \
                  % (str_arr, ans_idx)
        return que, eq, ans

    def __get_problem08_02__(self):
        n = self.__get_value__(2, 4)
        ball_name = self.__get_ball__()
        ans_idx = self.__get_value__(10, 150)
        
        arr = []
        colors = []
        q = '왼쪽부터 '
        for idx in range(n):
            # balls 개수
            n_balls = self.__get_value__(2, 7)
            arr.append(list(itertools.repeat(idx, n_balls)))
            # color
            while True:
                color = self.__get_colorname__()
                if color not in colors:
                    colors.append(color)
                    break
            q += '%s %s %d개,' %(color, ball_name, n_balls)
        arr = sum(arr, [])
        q = q[:-1]
        
        eq = 'PATCOM(%s)[%d]' %(arr, ans_idx - 1)
        ans = '%s' %(colors[patternize(arr)[ans_idx - 1]])

        p = np.random.rand()
        if p < 0.25:
            que = '%s가 반복되어 놓여 있습니다. 왼쪽에서 %d번째 공의 색깔을 쓰시오.' \
                  % (q, ans_idx)
        elif p < .5:
            que = '%s가 반복되어 있습니다. %d번째 공의 색깔은 무엇입니까?' \
                  % (q, ans_idx)
        elif p < .75:
            que = '%s가 반복되어 놓여 있습니다. %d번째 공의 색깔은 무엇입니까?' \
                  % (q, ans_idx)
        else:
            que = '%s가 반복되어 있습니다. %d번째 공의 색깔을 쓰시오.' \
                  % (q, ans_idx)
        return que, eq, ans
        
    def __get_problem08_03__(self):
        n_people = self.__get_value__(2, 4)
        names = self.__get_name_n_2__(n_people)
        n_max = self.__get_value__(50, 150)
        ans_idx = self.__get_value__(20, n_max)
        n_foods = self.__get_value__(2, 8)
        food_name = self.__get_food__()
        
        arr = []
        names_str = ''
        for idx in range(n_people):
            arr.append(list(itertools.repeat(idx, n_foods)))
            names_str += '%s,' %(names[idx])
        arr = sum(arr, [])
        names_str = names_str[:-1]
        
        eq = 'PATCOM(%s)[%d]' %(arr, ans_idx - 1)
        ans = '%s' %(names[patternize(arr)[ans_idx - 1]])

        p = np.random.rand()
        if p < 0.25:
            que = '%d개의 %s를 %s %d명에게 순서대로 %d개씩 나누어 줍니다. %d번째 %s을 받는 사람은 누구입니까?' \
                  % (n_max, food_name, names_str, n_people, n_foods, ans_idx, food_name)
        elif p < .5:
            que = '%d개의 %s를 %s %d명에게 차례대로 %d개씩 분배합니다. %d번째 받는 사람은 누구입니까?' \
                  % (n_max, food_name, names_str, n_people, n_foods, ans_idx)
        elif p < .75:
            que = '%d개의 %s를 %s %d명에게 순서대로 %d개씩 나눠줬습니다. %d번째 %s을 받는 사람은 누구입니까?' \
                  % (n_max, food_name, names_str, n_people, n_foods, ans_idx, food_name)
        else:
            que = '%d개의 %s를 %s %d명에게 순서대로 %d개씩 나누어주면, %d번째 받는 사람은 누구입니까?' \
                  % (n_max, food_name, names_str, n_people, n_foods, ans_idx)
        return que, eq, ans
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' 유형9 '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def __get_problem09_01__(self):
        n0, n1 = self.__get_name2__()
        ns = [n0[0:2], n1[0:2]]
        v0 = self.__get_value__(2, 5)
        str_vs0, vs0 = self.__get_value_array__(1, 50, v0)
        
        v1 = self.__get_value__(2, 5)
        str_vs1, vs1 = self.__get_value_array__(1, 50, v1)

        comp_op = self.__get_value__(0, 1)
        if comp_op == 0:
            comp_op_str = '큽'
            eq = 'CMP(SUM(%s), SUM(%s))' % (str_vs0, str_vs1)
            ans = '%s' % (ns[compare(sum(vs0), sum(vs1))])
        else:
            comp_op_str = '작습'
            eq = 'CMP(SUM(%s), SUM(%s))' % (str_vs0, str_vs1)
            ans = '%s' % (ns[1 - compare(sum(vs0), sum(vs1))])

        p = np.random.rand()
        if p < 0.25:
            que = '%s는 %s를 모았습니다. %s는 %s를 모았습니다. 누가 모은 수가 더 %s니까?' \
                  % (n0, str_vs0, n1, str_vs1, comp_op_str)
        elif p < .5:
            que = '%s는 %s를 가지고있습니다. %s는 %s를 가지고있습니다. 누가 가지고있는 수가 더 %s니까?' \
                  % (n0, str_vs0, n1, str_vs1, comp_op_str)
        elif p < .75:
            que = '%s는 %s를 모았으며, %s는 %s를 모았습니다. 누가 모은 수가 더 %s니까?' \
                  % (n0, str_vs0, n1, str_vs1, comp_op_str)
        else:
            que = '%s는 %s를 가졌으며, %s는 %s를 가지고있습니다. 누가 가지고있는 수가 더 %s니까?' \
                  % (n0, str_vs0, n1, str_vs1, comp_op_str)
        return que, eq, ans

    def __get_problem09_02__(self):
        num = self.__get_value__(2, 4)
        kor_b, kor_b_str = self.__get_kor_bracket_seq__(num)
        num_q = num
        arr = range(num)
        
        if np.random.rand() >= .5: # in case box name
            obj_name = self.__get_box__()
        else: # in case ball name
            obj_name = self.__get_ball__()        
        
        if np.random.rand() >= .5: # in case ">"
            comp_op = '>'
            comp_op_str = '큰'
        else: # in case "<"
            comp_op = '<'
            comp_op_str = '작은'
        
        q_str = ''
        ans = list(arr)
        
        for idx in range(num_q):
            idxs = np.random.choice(num, 2, replace = False)
            if np.random.rand() >= .5: # in case bigger
                q_str += '%s %s는 %s %s보다 큽니다. ' %(kor_b[idxs[0]], obj_name, kor_b[idxs[1]], obj_name)
                ans = switch(ans, idxs[0], idxs[1])
                
                if idx == 0: eq = 'SWITCH(%s, %d, %d)' %(arr, idxs[0], idxs[1])
                else: eq = 'SWITCH(%s, %d, %d)' %(eq, idxs[0], idxs[1])
                
            else: # in case smaller
                q_str += '%s %s는 %s %s보다 작습니다. ' %(kor_b[idxs[0]], obj_name, kor_b[idxs[1]], obj_name)
                ans = switch(ans, idxs[1], idxs[0])
        
                if idx == 0: eq = 'SWITCH(%s, %d, %d)' %(arr, idxs[0], idxs[1])
                else: eq = 'SWITCH(%s, %d, %d)' %(eq, idxs[1], idxs[0])
        
        if comp_op == '>':
            eq = 'ARGMAX(%s)' %(eq)
            ans = kor_b[argmax(ans)]
        else:
            eq = 'ARGMIN(%s)' %(eq)
            ans = kor_b[argmin(ans)]
            

        que = '%s %d개의 %s가 있습니다. %s크기가 가장 %s %s는 무엇입니까?' \
              % (kor_b_str, num, obj_name, q_str, comp_op_str, obj_name)

        return que, eq, ans

    def __get_problem09_03__(self):
        n = self.__get_value__(2, 7)
        str_vs, vs = self.__get_value_array__(0, 15, n, dtype='float')
        th = self.__get_float_value__(0, 5)
        
        if np.random.rand() >= .5:
            comp_op_str = '큰'
            eq = 'len(CMP(%s, %.2f))' % (vs, th)
            ans = '%s' % (len(compare(vs, th)))
        else:
            comp_op_str = '작은'
            eq = 'len(%d - CMP(%s, %.2f))' % (n, vs, th)
            ans = '%s' % (n - len(compare(vs, th)))

        p = np.random.rand()
        if p < 0.25:
            que = '%d개의 수 %s이 있습니다. 이중에서 %.2f보다 %s 수는 모두 몇 개입니까?' \
                  % (n, str_vs, th, comp_op_str)
        elif p < .5:
            que = '%d개의 소수 %s가 놓여있습니다. 이중에서 %.2f보다 %s 소수는 모두 몇 개입니까?' \
                  % (n, str_vs, th, comp_op_str)
        elif p < .75:
            que = '%d개의 수 %s이 주어졌습니다. 이중에서 %.2f보다 %s 수는 모두 몇 개 있습니까?' \
                  % (n, str_vs, th, comp_op_str)
        else:
            que = '%d개의 수 %s이 있다. 이중에서 %.2f보다 %s 수는 모두 몇 개입니까?' \
                  % (n, str_vs, th, comp_op_str)
        
        return que, eq, ans
        
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

