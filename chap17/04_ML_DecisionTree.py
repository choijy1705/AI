#-*- coding:utf-8 -*-
# 1. pip install graphviz
# 2. http://graphviz.gitlab.io/_pages/Download/Download_windows.html 에서 다운로드.
# 3. path 시스템 환경 변수에 C:\Program Files (x86)\Graphviz2.38\bin 경로 추가.

import pandas as pd
import numpy as np

district_dict_list = [
    {'district': 'Gangseo-gu', 'latitude': 37.551000, 'longitude': 126.849500, 'label': 'Gangseo'},
    {'district': 'Yangcheon-gu', 'latitude': 37.52424, 'longitude': 126.855396, 'label': 'Gangseo'},
    {'district': 'Guro-gu', 'latitude': 37.4954, 'longitude': 126.8874, 'label': 'Gangseo'},
    {'district': 'Geumcheon-gu', 'latitude': 37.4519, 'longitude': 126.9020, 'label': 'Gangseo'},
    {'district': 'Mapo-gu', 'latitude': 37.560229, 'longitude': 126.908728, 'label': 'Gangseo'},

    {'district': 'Gwanak-gu', 'latitude': 37.487517, 'longitude': 126.915065, 'label': 'Gangnam'},
    {'district': 'Dongjak-gu', 'latitude': 37.5124, 'longitude': 126.9393, 'label': 'Gangnam'},
    {'district': 'Seocho-gu', 'latitude': 37.4837, 'longitude': 127.0324, 'label': 'Gangnam'},
    {'district': 'Gangnam-gu', 'latitude': 37.5172, 'longitude': 127.0473, 'label': 'Gangnam'},
    {'district': 'Songpa-gu', 'latitude': 37.503510, 'longitude': 127.117898, 'label': 'Gangnam'},

    {'district': 'Yongsan-gu', 'latitude': 37.532561, 'longitude': 127.008605, 'label': 'Gangbuk'},
    {'district': 'Jongro-gu', 'latitude': 37.5730, 'longitude': 126.9794, 'label': 'Gangbuk'},
    {'district': 'Seongbuk-gu', 'latitude': 37.603979, 'longitude': 127.056344, 'label': 'Gangbuk'},
    {'district': 'Nowon-gu', 'latitude': 37.6542, 'longitude': 127.0568, 'label': 'Gangbuk'},
    {'district': 'Dobong-gu', 'latitude': 37.6688, 'longitude': 127.0471, 'label': 'Gangbuk'},

    {'district': 'Seongdong-gu', 'latitude': 37.557340, 'longitude': 127.041667, 'label': 'Gangdong'},
    {'district': 'Dongdaemun-gu', 'latitude': 37.575759, 'longitude': 127.025288, 'label': 'Gangdong'},
    {'district': 'Gwangjin-gu', 'latitude': 37.557562, 'longitude': 127.083467, 'label': 'Gangdong'},
    {'district': 'Gangdong-gu', 'latitude': 37.554194, 'longitude': 127.151405, 'label': 'Gangdong'},
    {'district': 'Jungrang-gu', 'latitude': 37.593684, 'longitude': 127.090384, 'label': 'Gangdong'}
]

train_df = pd.DataFrame(district_dict_list)
train_df = train_df[['district', 'longitude', 'latitude', 'label']]

dong_dict_list = [
    {'dong': 'Gaebong-dong', 'latitude': 37.489853, 'longitude': 126.854547, 'label': 'Gangseo'},
    {'dong': 'Gochuk-dong', 'latitude': 37.501394, 'longitude': 126.859245, 'label': 'Gangseo'},
    {'dong': 'Hwagok-dong', 'latitude': 37.537759, 'longitude': 126.847951, 'label': 'Gangseo'},
    {'dong': 'Banghwa-dong', 'latitude': 37.575817, 'longitude': 126.815719, 'label': 'Gangseo'},
    {'dong': 'Sangam-dong', 'latitude': 37.577039, 'longitude': 126.891620, 'label': 'Gangseo'},

    {'dong': 'Nonhyun-dong', 'latitude': 37.508838, 'longitude': 127.030720, 'label': 'Gangnam'},
    {'dong': 'Daechi-dong', 'latitude': 37.501163, 'longitude': 127.057193, 'label': 'Gangnam'},
    {'dong': 'Seocho-dong', 'latitude': 37.486401, 'longitude': 127.018281, 'label': 'Gangnam'},
    {'dong': 'Bangbae-dong', 'latitude': 37.483279, 'longitude': 126.988194, 'label': 'Gangnam'},
    {'dong': 'Dogok-dong', 'latitude': 37.492896, 'longitude': 127.043159, 'label': 'Gangnam'},

    {'dong': 'Pyoungchang-dong', 'latitude': 37.612129, 'longitude': 126.975724, 'label': 'Gangbuk'},
    {'dong': 'Sungbuk-dong', 'latitude': 37.597916, 'longitude': 126.998067, 'label': 'Gangbuk'},
    {'dong': 'Ssangmoon-dong', 'latitude': 37.648094, 'longitude': 127.030421, 'label': 'Gangbuk'},
    {'dong': 'Ui-dong', 'latitude': 37.648446, 'longitude': 127.011396, 'label': 'Gangbuk'},
    {'dong': 'Samcheong-dong', 'latitude': 37.591109, 'longitude': 126.980488, 'label': 'Gangbuk'},

    {'dong': 'Hwayang-dong', 'latitude': 37.544234, 'longitude': 127.071648, 'label': 'Gangdong'},
    {'dong': 'Gui-dong', 'latitude': 37.543757, 'longitude': 127.086803, 'label': 'Gangdong'},
    {'dong': 'Neung-dong', 'latitude': 37.553102, 'longitude': 127.080248, 'label': 'Gangdong'},
    {'dong': 'Amsa-dong', 'latitude': 37.552370, 'longitude': 127.127124, 'label': 'Gangdong'},
    {'dong': 'Chunho-dong', 'latitude': 37.547436, 'longitude': 127.137382, 'label': 'Gangdong'}
]

test_df = pd.DataFrame(dong_dict_list)
test_df = test_df[['dong', 'longitude', 'latitude', 'label']]

# 현재 가지고 있는 데이터에서, 레이블의 갯수 확인
print(train_df.label.value_counts())
print(test_df.label.value_counts())
'''
Gangnam     5
Gangdong    5
Gangseo     5
Gangbuk     5
Name: label, dtype: int64
Gangnam     5
Gangdong    5
Gangseo     5
Gangbuk     5
Name: label, dtype: int64
'''

# 데이터 전처리
# - 경도와 위도의 평균과 편차
print(train_df.describe())
'''
        longitude   latitude
count   20.000000  20.000000
mean   126.999772  37.547909
std      0.089387   0.055086
min    126.849500  37.451900
25%    126.913481  37.510177
50%    127.028844  37.552597
75%    127.056458  37.573690
max    127.151405  37.668800
'''

print(train_df.head())  # R 에서의 default : 6 , python 에서의 default : 5
'''
      district   longitude   latitude    label
0    Gangseo-gu  126.849500  37.551000  Gangseo
1  Yangcheon-gu  126.855396  37.524240  Gangseo
2       Guro-gu  126.887400  37.495400  Gangseo
3  Geumcheon-gu  126.902000  37.451900  Gangseo
4       Mapo-gu  126.908728  37.560229  Gangseo
'''
print(test_df.head())
'''
 dong   longitude   latitude    label
0  Gaebong-dong  126.854547  37.489853  Gangseo
1   Gochuk-dong  126.859245  37.501394  Gangseo
2   Hwagok-dong  126.847951  37.537759  Gangseo
3  Banghwa-dong  126.815719  37.575817  Gangseo
4   Sangam-dong  126.891620  37.577039  Gangseo
'''

# 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager as fm

font_location = 'C:/Windows/Fonts/malgun.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# 경도, 위도에 따른 데이터 시각화
# scatter_kws : 좌표 상의 점의 크기
sns.lmplot('longitude', 'latitude', data=train_df, fit_reg=False, scatter_kws={'s': 150}, markers=["o", "x", "+", "*"],
           hue="label")

# title
plt.title("2차원 '구' 분포 시각화")
plt.show()

# 데이터 다듬기
# - 학습 및 테스트에 필요없는 특징(feature)를 데이터에서 제거한다.
# - 구 이름 및 동 이름은 학습 및 테스트에 필요없으므로 제거한다.

train_df.drop(['district'], axis=1, inplace=True)
test_df.drop(['dong'], axis=1, inplace=True)

X_train = train_df[['longitude', 'latitude']]
Y_train = train_df[['label']]

X_test = test_df[['longitude', 'latitude']]
Y_test = test_df[['label']]

from sklearn import tree
import numpy as np
from sklearn import preprocessing

def display_decision_surface(clf, x, y):
    # 차트의 범위가 모든 학습 데이터를 포함하도록 설정
    x_min = x.longitude.min() - 0.01
    x_max = x.longitude.max() + 0.01
    y_min = x.latitude.min() - 0.01
    y_max = x.latitude.max() + 0.01

    # parameter 설정
    n_classes = len(le.classes_)
    plot_colors = "rywb"
    plot_step = 0.001

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z, cmap=plt.cm.RdYlBu)

    # 학습 데이터를 차트에 표시
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(x.loc[idx].longitude, x.loc[idx].latitude, c=color, label=le.classes_[i], cmap=plt.cm.RdYlBu,
                    edgecolors='black', s=200)

    # 차트 제목
    plt.title("의사 결정 트리 시각화", fontsize=16)

    # 차트 기호 설명
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8)

    # x축의 이름과 폰트 크기 설정
    plt.xlabel('위도', fontsize=16)

    # y축의 이름과 폰트 크기 설정
    plt.ylabel('경도', fontsize=16)

    # 차트 크기 설정
    plt.rcParams["figure.figsize"] = [7, 5]

    # 차트 폰트 크기 설정
    plt.rcParams["font.size"] = 14

    # x축 좌표상의 폰트 크기 설정
    plt.rcParams["xtick.labelsize"] = 14

    # y축 좌표상의 폰트 크기 설정
    plt.rcParams["ytick.labelsize"] = 14

    plt.show()

# pyplot 은 숫자로 표현된 레이블을 시각화 할 수 있음.
# LabelEncoder 로 레이블을 숫자로 변경
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(Y_train)

clf = tree.DecisionTreeClassifier().fit(X_train, y_encoded)
display_decision_surface(clf, X_train, y_encoded)


# 파라미터 설정한 모델의 결정 표면 시각화

clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=2, random_state=70).fit(X_train, y_encoded)
display_decision_surface(clf, X_train, y_encoded)