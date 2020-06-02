import matplotlib.pyplot as plt
import numpy as np

# 단순한 그래프 그리기

# 데이터 준비
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
"""
plt.plot(x, y1, label="sin") # label타이틀 혹은 범례
plt.plot(x, y2, linestyle = "--", label="cos") # cos 함수는 점선으로 그리기.
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title("sin & cos") # 제목
plt.legend() # 범례
plt.show()

"""
np.random.seed(1234) # 난수를 고정
x = np.arange(10)
y = np.random.rand(10) # 0~1사이의 random한 값을 발생시켜 0부터 9까지의 값을 꺾은선 그래프로 보여준다.

# plt.plot(x,y) # 꺾은선 그래프를 등록
# plt.show()

# 3차 함수 f(x) = (x - 2) x (x + 2)
def f(x):
    return (x-2) * x * (x+2)

print(f(0))
print(f(2))
print(f(-2))

# x값에 대해 ndarray 배열이며 각각에 대응한 f를 한꺼번에 ndarray로 돌려준다.
print(f(np.array([1,2,3]))) # [-3  0 15] 배열일 경우 각각 함수에서 계산되어 배열의 형태로 반환되어진다.
print(type(f(np.array([1,2,3]))))  # <class 'numpy.ndarray'> 선형대수의 결과 또한 numpy의 배열 자료형이다.

# 그래프를 그리는 x의 범위를 -3 ~ 3까지로 하고, 간격 0.5
x = np.arange(-3, 3.5, 0.5)
# plt.plot(x, f(x))
# plt.show()

# 그래프를 장식
def f2(x, w):
    return (x-w) * x * (x+2) # 함수 정의

# x를 정의
x = np.linspace(-3, 3, 100) # x를 100분할하기

# 차트묘사
"""
plt.plot(x, f2(x,2), color="black", label="$w=2$")
plt.plot(x, f2(x,1),color = "blue", label="$w=1$")
plt.legend(loc = "upper left")
plt.ylim(-12, 15) # y축 범위
plt.title("$f_2(x)$")
plt.xlabel("$x$") # $문자를 기울려준다.
plt.ylabel("$y$")
plt.grid(True) # grid(눈금자)
plt.show()
"""
# 그래프를 여러 개 보여주기
"""
plt.figure(figsize=(10,3)) # 전체 영역의 크기를 지정
plt.subplots_adjust(wspace=0.5, hspace=0.5) # 그래프의 간격을 지정
for i in range(6):
    plt.subplot(2,3, i+1) # 그래프 위치를 지정
    plt.title(i+1)
    plt.plot(x, f2(x,i))
    plt.ylim(-20,20)
    plt.grid(True)

plt.show()
"""
# 이미지 표시하기
from matplotlib.image import imread

img = imread('image/lena.png') # 이미지 읽어오기

plt.imshow(img)
plt.show()