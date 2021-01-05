# 딥러닝 공부

https://developers.google.com/machine-learning/crash-course

## 영현, 윤성

## 유용한 링크들

마크다운 공부:  
https://theorydb.github.io/envops/2019/05/22/envops-blog-how-to-use-md/

## 일정

**2021-01-04**  
github 블로그 생성하기.  
구글 머신러닝 코스 초기 진입하기.

**2021-01-05**  
선형 회귀 (Linear Regression)  
https://wikidocs.net/21670  
일단 위 링크의 1번 부분만읽고 구글 ㄱㄱ

**2021-01-06**  
구글 크래시 코스 - 1,2강 듣고 정리

# 1강 - ML 소개

여기에서 배우면 3가지 더 잘 하게 됨.

1. 프로그래밍 시간을 줄일 수 있는 도구를 얻게됨.
2. '맞춤 설정'으로 각 상황에 맞는 프로그래밍을 더 유동적으로 만들 수 있음.
3. '언어'관련 프로그램을 만들때에도 언어를 바꾸기만 하면 작동함.

프로그램으로써 수동으로 할 수 없는 일을 짤 수 있음.

## 머신러닝을 배워야 하는 철학적 이유

내 프로그램의 속성이 옳다는 것을 증명하기 위해 '논리'를 배움.  
보는 시야 자체를 넓히는 것에 집중할 것.

# 2강 - ML 문제로 표현하기

지도 머신러닝

- 모델 학습할때 '라벨' 제공 - 예를 들어, 이메일 스팸 필터링에서 '스팸 또는 스팸 아님' 정도. 이것이 우리가 예측하려는 타겟.
- 특성을 데이터를 표현하는 정보.
- 특성은 해당 이메일에서 추출할 수 있는 어떤 정보든 가져와서 머신러닝 시스템에 나타낼 수 있음.
- 라벨이 없는 예 - 특성 정보가 없는 것들.
- 바로 예측을 하는 모델을 만들 것.

**라벨** - '예측하는 항목' (단순 선형 회귀의 y 변수)  
 ex) 밑의 향후 가격, 사진에 표시되는 동물의 종류, 오디오 클립의 의미 등 무엇이든 라벨이 될 수 있음.  
<br />
**특성** - '입력 변수' (단순 선형 회귀의 x 변수)
ex) 간단한 머신러닝 프로젝트에서는 특성을 한 개만 사용하지만 복잡한 프로젝트의 경우 수백만 개의 특성을 사용할 수 있음. x1, x2, x3... xN 이렇게.  
 특성의 예로는 이메일 텍스트의 단어, 보내는 사람의 주소, 이메일이 전송된 시간, '특정 구문'이 포함된 이메일.  
 <br/>
그러니까 특성은 꺼내올 수 있는 대상이고 라벨은 예측하는 대상이다.

## 모델을 학습 시키려면

모델을 학습 시키려면 '라벨이 있는 예'를 사용할 것.  
스팸 감지 예시에서는 사용자가 '스팸 또는 스팸 아님'으로 명시해준 이메일들이 그 예가 됨. 라벨이 있는 것에서 없는 것을 예측하는게 머신 러닝이다.

---

**모델** - 모델이란 특성과 라벨의 관계를 정의하는 것. 예를 들어 스팸 감지 모델에서, 특정 특성과 스팸을 긴밀하게 연결하는 것.  
<br/>
**학습** - 모델을 만들거나 배우는 것. 라벨이 있는 예를 모델에 보여주고, 모델이 특성과 라벨의 관계를 점차적으로 학습하게 하는 것.  
<br/>
**추론** - 학습된 모델을 라벨이 없는 예에 적용하는 것. 학습된 모델로, 유용한 예측 (y')를 해낸다.  
<br/>
**회귀** - 연속적인 값을 예측. 예를 들어, 캘리포니아의 주택 가격이 얼마인가? 사용자가 이 광고를 클릭할 확률이 얼마인가?  
<br/>
**분류** - 불연속적인 값을 예측. 예를 들어, 주어진 이메일 메세지가 스팸인가 아닌가. 이 이미지가 강아지인가 고양이인가.
<br/>

---

사용자가 '좋아'하는게 아닌, '설명 클릭 횟수'같은게 쓸모있는 것임.

---
# 2강 - Framing the basics of maching learning
Real basics of machine learning are comprised of labels, features, and models.

In general, machine learning systems are "taught" to use various kinds of inputs in producing socially useful predictions about numerous sets of data.

There are two sets of fundamental terminologies in machine learning: 1) labels and 2) features. 

These two terms comprise of what is known as linear regression. Linear regression normally shows the relationship between x- and y- variables. 
For example, the longer the distance you run, the higher the amount of water you will want to drink. 
Mathematically, this can be thought of as the value of an independent variable (x) affecting the value of a dependent variable (y). 
A simple linear regression models this relationship between x- and y- variables. This is where the machine learning principle lies. 

__Labels__:
These are the things that we _predict_: the result values of y. It can be literally _anything_.

__Features__:
These are the inputs, or the x- variables. Usually, a complex machine learning system consists of millions of kinds of features. 

__Examples__:
These are the particular (usually numerical) types of _real_ data, x. X is a vector. There are two types of examples we consider, which are 1) labeled examples and 2) unlabeled examples. 
> 1) __Labeled examples__: 
Has BOTH feature and the label. To express this in mathematical terms: {features, label}: (x, y). 
We use these exampels to train the model, just like we use past data to predict outcomes for the future. 

> 2) __Unlabeled examples__: 
Has ONLY feature and NOT the label. To express this in mathematical terms: {features, ?}: (x, ?).
The trained model (using labeled examples) is used to predict the label on unlabeled exampeles.

__Models__:
This is the relationship between features and labels. For example, a spam detection model might associate certain features strongly with "spam".
A training process helps the model figure out the relationship or the correlation between features and labels.  An inference from the model (which was done through training) applys the model to unlabeled examples, helping us to figure out the labels for those examples. 

---

# 3강 - ML로 전환하기 

## 영상 강의

y=mx+n 이런 고등학교 에서 배운 일차방정식 그래프에서 썼던 m이 이제  
y=wx+B로 표현된다. w는 벡터의 가중치이다. b는 편향.

손실이라는 개념을 떠올려보자. 예측값 x에서 실제값을 빼면 손실을 생각할 수 있다. 손실은 항상 양수이다.  
 손실을 계산하는데에 좋은 함수들이 많지만 그 중에서도 **L2 손실**이 좋다. 이는 제곱 오차라고도 한다.  
 <br/>
예측과 라벨 간의 차이 제곱 = (관찰-예측)^2 = (y-y')^2
모델을 학습 시킬 때에는 하나의 데이터 세트가 아닌 여러 세트를 사용해 손실을 최소화 해야한다. 예측 값이 멀어질 수록 손실도 제곱의 단위로 커진다.

## 문서: 선형 회귀

![image](https://developers.google.com/machine-learning/crash-course/images/CricketLine.svg?hl=ko)
위 그래프는 선형 관계이다.  
대수학으로는 y = mx + b  
머신러닝 관습으로는 y' = b + w1x1  
<br/>
y'는 예측된 라벨(얻고자 하는 출력)이다.
b는 편향 (y절편)이다.  
w1은 특성 1의 가중치이다. 가중치는 m으로 표현된 '기울기'와 같은 개념이다.  
x1은 특성이다. 즉, 알려진 입력이다.

## 문서: 학습 및 손실

모델을 학습시킨다는 것은  
**라벨이 있는 데이터로부터 올바른 가중치와 편향값을 학습(결정)하는 것.**  
이다.

지도 학습에서 머신러닝 알고리즘은 다양한 예를 검토하고 손실을 최소화하는 모델을 찾아봄으로써 모델을 만들어낸다. 이것을 **경험적 위험 최소화** 라고 한다.  
모델 학습의 목표는, 모든 예에서 평균적으로 **작은 손실**을 갖는 **가중치**와 **편향**의 집합을 찾는 것.  
![image](https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png?hl=ko)
빨간색 화살표가 오른쪽에서 훨씬 더 짧음. 오른쪽이 훨씬 더 예측을 잘 하는 모델. 그렇다면 이것을 수학적 함수로?  
<br/>

### 제곱 손실: 잘 알려진 손실 함수

MSE - 평균 제곱 오차  
N은 D에 포함된 예의 수이다. MSE는 머신러닝에서 흔히 사용되지만, 모든 상황에서 최선인 유일한 손실 함수는 아니다.
