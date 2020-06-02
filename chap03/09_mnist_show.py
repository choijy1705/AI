# MNIST(Modified National Institude of Standards and Technology)
# - 손으로 직접 쓴 숫자(필기체 숫자)들로 이루어진 데이터 셋
# - 0 ~ 9까지의 숫자 이미지로 구성되며, 60,000개의 트레이닝 데이터와
#   10,000개의 테스트 데이터로 이루어져 있음.
# - 28x28 size

import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져 올 수 있도록 설정.

import numpy as np

from dataset.mnist import load_mnist
from PIL import Image # pip install image

def img_show(img): # 데이터를 이미지로 볼 수 있도록 해준다.
    pil_img = Image.fromarray(np.uint8(img)) # 정수를 담을 8비트라는 의미. uint8 양수 값만 담을 수 있는 한 바이트의 자료형
    pil_img.show()

if __name__ == '__main__':
    (x_train, t_train),(x_test, t_test) = load_mnist() # 다운 받은 파일을 가져온다
# (훈련용 이미지, 레이블 ) (테스트 이미지,레이블)

    img = x_train[10]
    label = t_train[10]
    print(label) # 3
    print(img.shape) # (784, )

    img = img.reshape(28, 28) # 형상을 원래 이미지의 크기로 변형
    print(img.shape)

    img_show(img)