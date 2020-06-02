# AND 게이트를 퍼셉트론으로 표현하기
# 퍼셉트론은 머신러닝이 알려지기 전 이론, 사람이 직접 적절한 값을 찾아준다.
def AND(x1,x2):
    w1, w2, theta = 1.0, 0.5, 1.0 # 사람이 매개변수 값을 정함 , 괄호가 생략되어진 튜플자료형이다. 언패킹의 방법으로 각각 변수에 담아주고 있다.
    # x1이 x2보다 더 높은 비중을 가진다고 해석할 수 있다.
    y = w1 * x1 + w2 * x2

    if y <= theta:
        return 0
    else:
        return 1

if __name__ =='__main__':
    for xs in [(0,0),(0,1),(1,0),(1,1)]:
        result = AND(xs[0],xs[1])
        print(str(xs) + ":" +str(result))

