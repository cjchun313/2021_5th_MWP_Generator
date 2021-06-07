import math
import itertools

def lcm(x):
    try:
        n = len(x0)
    except:
        n = 1
        
    if n == 1:
        return x[0]
    else:
        max_val = 1
        for idx in range(n): max_val *= x[idx]
        min_val = max(x)
        val = max_val
        for idx in range(max_val + 1, min_val, -1):
            cnt = 0
            for idx2 in range(n):
                if idx%x[idx2] == 0: cnt += 1
            if cnt == n: val = idx
        return val
				
def gcd(x):
    try:
        n = len(x0)
    except:
        n = 1
        
    if n == 1:
        return 1
    else:
        val = 1
        for idx in range(1, min(x) + 1):
            cnt = 0
            for idx2 in range(n):
                if x[idx2]%idx == 0: cnt += 1
            if cnt == n: val = idx
        return val
            
def compare(x0, x1):
    try:
        n = len(x0)
    except:
        n = 1
    if n == 1:
        if x0 > x1:
            return 0
        else:
            return 1
    else: # 비교하여 큰 숫자들만 return
        val = []
        for idx in range(n):
            if x0[idx] > x1:
                val.append(idx)
        return val

def switch(arr, i0, i1):
    if arr[i0] < arr[i1]: arr[i0], arr[i1] = arr[i1], arr[i0]
    return arr
# def switch_bigger(arr, i0, i1):
#     if arr[i0] < arr[i1]: arr[i0], arr[i1] = arr[i1], arr[i0]
#     return arr
# 
# def switch_smaller(arr, i0, i1):
#     if arr[i0] > arr[i1]: arr[i0], arr[i1] = arr[i1], arr[i0]
#     return arr

def argmax(arr):
    val = 0
    for idx in range(len(arr) - 1):
        if arr[idx + 1] > arr[idx]:
            val = idx + 1
    return val
    
def argmin(arr):
    val = 0
    for idx in range(len(arr) - 1):
        if arr[idx + 1] < arr[idx]:
            val = idx + 1
    return val


def flatten(list_arr):
    return sum(list_arr, [])

def find_str_pattern(arr, pat=5):
    count_ = [idx for idx in range(1, len(arr) * 2) if arr[idx:idx+pat] == arr[:pat]]
    return arr[:count_[0]]
    
def patternize(arr):
    return sum(itertools.repeat(arr, 100), [])    
    
