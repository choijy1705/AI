# 데이터 분석에 유용한 기능들...
# - split(구분자) : 구분자로 구분, 기본값은 공백

test_text = 'the-joeun-computer-with-python'
result = test_text.split('-') # - 구분자를 기준으로 나눠서 담아준다.
print(result) # ['the', 'joeun', 'computer', 'with', 'python'] 구분자를 기준으로 나눠 리스트 형태로 담아준다. 구분자는 미포함

# 구분자.join(리스트) : split 함수와 반대로 구분자로 붙인다.
test_text = ['the', 'joeun', 'computer', 'with', 'python']
print(test_text)
result = '-'.join(test_text) # 문자열 데이터를 결합시켜주는 메서드
print(result)

# split()와 join() 의 응용
result = '-'.join('345.234.6789'.split('.'))
print(result)

# enumerate(list) : 인덱스와 값을 함께 반환
for i, name in enumerate(['a','b','c','d']): # i : index, name : 값
    print(i , name)
'''
0 a
1 b
2 c
3 d
'''

# pip install ~

seq = ['mon','tue','wed','thu','fri','sat','sun']
print(dict(enumerate(seq))) # enumerate 인덱스와 값을 반환, dictionary 자료형으로 인덱스와 값 반환
# {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}

key_seq = 'abcdefg'
value_seq= ['mon','tue','wed','thu','fri','sat','sun']
print(dict(zip(key_seq, value_seq))) # zip : zipper를 의미 , 지퍼를 통하여 인덱스를 원하는 값으로 설정할 수 있다. key-value형식
# {'a': 'mon', 'b': 'tue', 'c': 'wed', 'd': 'thu', 'e': 'fri', 'f': 'sat', 'g': 'sun'}

# List comprehension - 리스트 변환하는 표현식으로 유용한 기능
day = ['mon','tue','wed','thu','fri','sat','sun']
print([x for x in day]) # day에 담겨있는 값을 하나씩 읽어오면서 x에 담아 값을 리턴해준다.
# ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
print(day)
# ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

data = [35, 56, -53, 45, 27, -28, 8, -12]
print([x for x in data if x >= 0])
# [35, 56, 45, 27, 8] data에서 0이상의 값만 반환하여 x에 담아 값을 리턴

data = [35, 56, -53, 45, 27, -28, 8, -12]
print([x**2 for x in data if x>=0]) # [1225, 3136, 2025, 729, 64] 0이상의 값을 제곱하여 리스트로 반환

# Counter를 이용한 카운팅
#  - Countsms 아이템의 갯수를 자동으로 카운팅.
from collections import Counter


message = '''
모든 국민은 종교의 자유를 가진다. 법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 효력을 발생한다. 
국회가 재적의원 과반수의 찬성으로 계엄의 해제를 요구한 때에는 대통령은 이를 해제하여야 한다.
모든 국민은 법률이 정하는 바에 의하여 선거권을 가진다. 모든 국민은 헌법과 법률이 정한 법관에 의하여 법률에 의한 재판을 받을 권리를 가진다. 
군인은 현역을 면한 후가 아니면 국무위원으로 임명될 수 없다.
'''

counter = Counter(message.split()) # 첫글자가 대문자면 참조 자료형일 가능성이 높다. instance 생성
print(counter) # 각 단어에 대한 빈도수를 체크하여 결과를 피드백 해준다.
# Counter({'모든': 3, '국민은': 3, '가진다.': 3, '법률이': 2, '의하여': 2, '종교의': 1, '자유를': 1,...

# Counter(dict) -> list 형태로 반환
print(counter.most_common()) # list형태로 결과를 반환 해준다.
# [('모든', 3), ('국민은', 3), ('가진다.', 3), ('법률이', 2), ('의하여', 2), ('종교의', 1),'...

# list -> dict 형태로 변환
dict_msg = dict(counter.most_common())
print(dict_msg) # list형식을 dict 형태로 바꿔서 값을 반환
# {'모든': 3, '국민은': 3, '가진다.': 3, '법률이': 2, '의하여': 2, '종교의': 1,....

print(dict_msg['모든']) # 2