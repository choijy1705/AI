
from layer_arithmetic import MulLayer

mulAppleLayer = MulLayer()
mulTaxLayer = MulLayer()


# 순전파====================

applePrice = 100
appleNumber = 2
tax = 1.1

apple_price = mulAppleLayer.forward(applePrice, appleNumber)
price = mulTaxLayer.forward(apple_price, tax)

print("사과 구입 금액 : ", int(price)) # 220

# 역전파 =====================

dout = 1

backward_price, backward_tax = mulTaxLayer.backward(dout)
print("역전파 소비세 결과 : ", backward_tax)

backward_apple, backward_apple_num = mulAppleLayer.backward(backward_price)
print("역전파 사과 값 결과(미분)", backward_apple)
print("역전파 사과 값 갯수(미분)", backward_apple_num)