"""
# NumPy 란?
  - 파이썬 기반 데이터 분석 환경에서 행렬 연산을 위한 핵심 라이브러리.
  - "Numerical Python"의 약자로 대규모 다차원 배열과 행렬 연산에 필요한 다양한 함수를 제공.
  - 특히, 메모리 버퍼에 배열 데이터를 저장하고 처리하는 효율적인 인터페이스를 제공.
  - 파이썬 list 객체를 개선한 NumPy의 ndarray 객체를 사용하면 더많은 데이터를 더 빠르게 처리할 수 있다.
"""

# 배열 생성
import numpy as np
import matplotlib.pyplot as plt

# - 1차원 배열(Vector) 정의
arr = np.array([1, 2, 3]) # 동일한 자료형이면 list를 np패키지 안에 array함수를 이용하여 배열로 나타낼수 있다.
print(arr) # [1 2 3]      * list는 ,단위로 구분한다. numpy패키지를 통하여 배열로 변환한 것을 확인 할 수 있다.

# - 2차원 배열(Matrix) 정의
# * 행렬은 반드시 동일한 자료형, 테이블은 동일한 자료형일 필요가 없다. column 별로 각각의 자료형을 가질 수 있다. 행렬을 테이블이라 할 순 있지만 테이블은 행렬이라 할 수 없다.
arr2 = np.array([[1, 2, 3],[4, 5, 6]]) # 2행 3열
print(arr2)
#[[1 2 3]     *2차원 배열의 형태, 시작되는 대괄호의 갯수를 통하여 몇 차원의 배열인지 파악할 수 있다.
# [4 5 6]]

# - 3차원 배열(Array) 정의
arr3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]) # 2면 2행 3열
print(arr3)

# *참고 - java는 면,행,렬   /  R은 행,열,면 / Python은 [[[ ]]]

"""
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]

"""

print("arr3.shape:{0}".format(arr3.shape)) # arr3.shape:(2, 2, 3)  배열의 차원정보를 반환해준다. (면,행,열)

# 배열 생성 및 초기화
# zeros((행, 열)) : 0으로 채우는 함수
arr_zeros = np.zeros((3,4))
print(arr_zeros)
"""
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
 
 0.은 실수라는 뜻 ,의 의미가 아니다. 0으로 초기화하면서 배열을 생성할때 유용한 함수
"""

# ones((행, 열)) : 1로 채우는 함수
arr_ones = np.ones((2,2))
print(arr_ones)
"""
[[1. 1.]
 [1. 1.]]
"""

# full((행, 열),값) : 값으로 채우는 함수
arr_full = np.full((3,4),7)
print(arr_full)
'''
[[7 7 7 7]
 [7 7 7 7]
 [7 7 7 7]]
'''

# eye(N) : (N,N)의 단위 행렬 생성
arr_eye = np.eye(5)
print(arr_eye)
'''
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

# empty((행, 열)) : 초기화 없이 기존 메모리 값이 들어감
arr_empty = np.empty((3,3))
print(arr_empty)

'''
[[0.00000000e+000 0.00000000e+000 0.00000000e+000]
 [0.00000000e+000 0.00000000e+000 3.12249488e-321]
 [4.23145653e-307 1.62601486e-260 6.21813885e+175]]
'''

# _like(배열) 지정한 배열과 동일한 shape의 행렬을 만듦.
# 종류 : np.zeros_like(), np.ones_like(), np.full_like(), np.empty_like()
arr_sample = np.array([[1, 2, 3],[4, 5, 6]])
arr_like = np.ones_like(arr_sample)
print(arr_like)
'''
[[1 1 1]
 [1 1 1]]
'''

# 배열 데이터 생성 함수
# - np.linspace(시작, 종료, 개수) : 갯수에 맞게끔 시작과 종료 사이에 균등하게 분배

arr_linspace = np.linspace(1,10,5)
print(arr_linspace) # [ 1.    3.25  5.5   7.75 10.  ]

# plt.plot(arr_linspace, 'o') # 그래프를 그려주는 함수 마커를 원('o')으로 만든 그래프를 보여줌.
# plt.show()

# np.arange(시작, 종료, 스텝) : 시작과 종료 사이에 스텝 간격으로 생성.
arr_arange = np.arange(1,20,2)
print(arr_arange) # [ 1  3  5  7  9 11 13 15 17 19]
# plt.plot(arr_arange,"v")
# plt.show()

# list vs ndarray(1차원 배열(Vector))
x1 = [1, 2, 3]
y1 = [4, 5, 6]
print(x1 + y1) # [1, 2, 3, 4, 5, 6] list형식 일때는 뒤에 추가된다.

x2 = np.array([1, 2, 3])
y2 = np.array([4, 5, 6])
print(x2 + y2) # [5 7 9] 배열 형식일때는 +연산이 수행된다.

print(type(x1)) # <class 'list'>
print(type(x2)) # <class 'numpy.ndarray'> nd는 N dimension

print(x2[2]) # 요소의 참조 : 3
x2[2] = 10 # 요소의 수정
print(x2)  # [1 2 10]

# 연속된 정수 벡터의 생성
print(np.arange(10))  # [0 1 2 3 4 5 6 7 8 9]
print(np.arange(5, 10)) # [5 6 7 8 9]

x = np.array([10, 11, 12])
for i in np.arange(1,4):
    print(i)
    print(i+x)

'''
1
[11 12 13]
2
[12 13 14]
3
[13 14 15]
'''

# ndarray형의 주의점
a = np.array([1, 1])
b = a # 주소값 복사 , a 는 참조변수이다.
# a와 b의 독립적인 공간으로 주소값이 아닌 값만 저장하기 위해서는 a.copy() 메서드를 이용하여야 한다.

print('a = ' + str(a)) # [1 1] a를 String 구조로 형변환
print('b = ' + str(b)) # [1 1]

b[0] = 100
print('b = ' + str(b)) # [100 1] 주소값을 저장하는 것이기 때문에 b를 통하여 값을 변경하면 b와 같은 주소값인 a의 값도 바뀌게 된다.
print('a = ' + str(a)) # [100 1]

#########################################
a = np.array([1, 1])

b = a.copy() # 데이터 복사 , 가지고 있는 데이터의 공간을 새로 할당하여 주소값을 리턴해준다. a와 b의 주소값이 다르기때문에 a와 b는 서로 독립적이다.

print('a = ' + str(a)) # [1 1]
print('b = ' + str(b)) # [1 1]

b[0] = 100
print('b = ' + str(b)) # [100 1]
print('a = ' + str(a)) # [1 1] a와 b는 서로 독립적이 주소값이기 때문에 b값이  변경됬더라도 a값은 바뀌지 않는것을 확인 할 수 있다.

# 행렬(2차원)
x = np.array([[1, 2, 3],[4, 5, 6]]) # 2행 3열
print(x)
print(type(x)) # <class 'numpy.ndarray'>
print(x.shape)  # (2, 3) - 튜플
w, h = x.shape

print(w) # 2
print(h) # 3
print(x[1, 2]) # 6
x[1, 2] = 10 # 요소의 수정
print(x)

# 요소가 랜덤인 행렬 생성
randArray = np.random.rand(2, 3)
print(randArray) # 랜덤한 실수의 행렬 반환

'''
[[0.56846045 0.46730596 0.47543683]
 [0.39240331 0.16269284 0.64281209]]
'''

randnArray = np.random.randn(2,3) # 평균 0, 분산 1 가우스 분포로 난수를 생성.
print(randnArray)
'''
[[-0.01474835 -0.15572226  1.75875321]
 [-0.99538861  1.30377842  0.12952792]]
'''
randintArray = np.random.randint(10,20,(2,3))
print(randintArray)
'''
[[11 19 16]
 [18 19 13]]
'''

# np.random.normal(정규분포 평균, 표준편차, (행,열) or 개수)
mean = 0 # 평균
std = 1 # 표준편차
arr_normal = np.random.normal(mean, std, 10000)
# plt.hist(arr_normal, bins = 100) # bins 나누는 구간 갯수(100개정도 더 잘게 나눠보라는 의미
# plt.show()


# np.random.random((행, 열) or size) : (행, 열)의 정규 분포로 난수(0 ~ 1 사이) 생성
arr_random = np.random.random((3, 2))
print(arr_random)
'''
[[0.48378791 0.99375498]
 [0.3738114  0.62574001]
 [0.51287059 0.24650745]]
'''

data = np.random.random(10000) # size
# plt.hist(data,bins=100) # 100개의 구간으로 나누어 히스토그램을 보이라는 의미
# plt.show()

# 행렬의 크기 변경
a = np.arange(10)
print(a) # [0 1 2 3 4 5 6 7 8 9] 배열
a_arange = a.reshape(2,5)
print(a_arange) # 2행 5열의 형태로 변환
'''
[[0 1 2 3 4]
 [5 6 7 8 9]]
'''
print(type(a_arange)) # <class 'numpy.ndarray'>
print(a_arange.shape) # 2,5

# 행렬(numpy.ndarray)의 사칙 연산
# - 덧셈
x = np.array([[4,4,4],[8,8,8]])
y = np.array([[1,1,1],[2,2,2]])

print(x + y) # 2차원 배열의 합 각 행,열에 값이 더해져서 나온다
'''
[[ 5  5  5]
 [10 10 10]]
'''

# - 스칼라 x 행렬
x = np.array([[4,4,4],[8,8,8]])
scar_arr = 10 * x
print(scar_arr)
'''
[[40 40 40]
 [80 80 80]]
'''

# - 산술함수 : np.exp(x), np.sqrt(), np.log(), np.round(), np.std(), np.max(), np.min()
print(np.exp(x)) # 지수함수
'''
[[  54.59815003   54.59815003   54.59815003]
 [2980.95798704 2980.95798704 2980.95798704]]
'''

# - 행렬 * 행렬
x = np.array([[1,2,3],[4,5,6]]) # (2,3)
y = np.array([[1,1],[2,2],[3,3]]) # (3,2)
print(x.dot(y)) # 행렬의 곱샘결과
'''
[[14 14]
 [32 32]]
'''

# 원소 접근
data = np.array([[51,55],[14,19],[0,4]])
print(type(data))
print(data[0][1]) # java와 같이 원소에 접근할때 [행][열]로 접근한다. 55

for row in data:
    print(row)

y = data.flatten() # x를 1차원 배열로 변환(평탄화)
print(y) # [51 55 14 19  0  4]

# 슬라이싱
x = np.arange(10)
print(x) # [0 1 2 3 4 5 6 7 8 9]

print(x[:5]) # :앞에 아무것도 없으면 처음부터 시작하겠다는 뜻  [0 1 2 3 4], 인덱스는 0부터 카운트
print(x[5:]) # 인덱스5부터 끝까지 [5 6 7 8 9]
print(x[3:8]) # [3 4 5 6 7]

print(x[3:8:2]) # 마지막 값은 2만큼 건너 뛰겠다는 뜻

print(x[::-1]) # 파이썬에서의 음수는 뒤에서 부터 카운트 한다는 뜻, [9 8 7 6 5 4 3 2 1 0] 처음부터 끝까지 카운트하되 뒤에서부터 값을 불러온다.

y = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3행 3열
print(y[:2, 1:2]) # 0행의 1열, 1행의 1열
"""
[[2]
 [5]]
"""

# 조건을 만족하는 데이터 수정
#  - bool 배열 사용
x = np.array([1,1,2,3,5,8,15])
print(x > 3) # 배열을 스칼라와 비교하였을때 [False False False False  True  True  True] 각각의 원소들과 스칼라값을 비교하여 진리값을 배열형태로 반환해준다.

y = x[x > 3]
print(y) # [ 5  8 15] 진리값이 true 인 값만 배열형식으로 y에 담아준다.

x[x > 3] = 555
print(x) # [  1   1   2   3 555 555 555] true값인 부분에 555가 담겨 배열형태로 반환.



















