---
# Kyonggi University data analysis project

# </br> <주제>
</br> :pencil2: 큰 주제 : 특정 카테고리 상품 (니트, 맨투맨)에 대한 리뷰 분석을 바탕으로 소비자의 구매 결정에 도움을 줄 수 있는 인사이트 제공

- 세부 주제 : 고객의 리뷰를 바탕으로 총장, 어깨너비, 가슴단면, 소매길이가 큰지 작은지에 대한 시각화 자료를 제시
- 세부 주제 : 키, 몸무게, 원하는 핏 (예를 들면, 총장이 길지만 소매는 짧음)을 바탕으로 고객의 총장, 어깨넓이, 가슴둘레, 소매 길이를 측정

```
Q. 왜 많은 카테고리 중에서 니트, 맨투맨을 선택한 것인가?
=> 니트, 맨투맨은 많은 상품이 있고 그에 따라 많은 리뷰를 보유
=> S, M, L, XL 잘 구분되어 있음. (사이즈 추천 모델링에 활용하기 위한 목적)
```

# </br> <자료>
</br> :pushpin: 자료 출처 : 무신사 

- selenium과 beautifulsoup을 이용해 약 40,000개의 리뷰 데이터를 수집
- 무신사에서 리뷰 데이터를 수집한 이유는 남성 의류 쇼핑몰 중에서 가장 많은 상품을 보유하고 있으며 그에 따른 다량의 리뷰가 확보되어 있음

|Nickname|Product_Name|Review|Gender|Height|weight|Size|Length|Sholder|Chest|Sleeve|
|------|---|------|---|------|---|------|---|------|---|------|
|LV.3 jhcjhc|스웨트셔츠 [블랙]|생략|남성|172cm|70kg|M|71.0|48.5|54.5|63.0|
|LV.4 applebite|스웨트셔츠 [블랙]|생략|남성|167cm|63kg|M|71.0|48.5|54.5|63.0|
|LV.4 지진1|스웨트셔츠 [블랙]|생략|남성|174cm|53kg|M|71.0|48.5|54.5|63.0|
|LV.6 구니구니구니|스웨트셔츠 [블랙]|생략|남성|170cm|65kg|M|71.0|48.5|54.5|63.0|	
|LV.4 메론맛크롱|스웨트셔츠 [블랙]|생략|남성|172cm|70kg|M|71.0|48.5|54.5|63.0|

```
Nickname : Nickname은 비식별화되어 있음
Product_Name : 약 120개 제품의 리뷰가 쌓여있음
Review : /n, ㅎㅎ, ㅠㅠ, 이모티콘, 띄어쓰기에 대한 전처리 작업이 필요
ex) 옷이 너무 마음에 들어요 /n 근데 핏이 저한테는너무 크네요 ㅠㅠ😭
Gender : 남, 여
Length, Sholder, Chest, Sleeve의 경우에는 무신사에서 제공하는 제품별 사이즈를 참고하여 기입
```

① Height

- 남성
    
  </br>![image](https://github.com/TaeseongYang/Musinsa_project/assets/156265617/2b504e17-cde9-4da2-83d9-88f9f75e32fa)
```
1. 남성의 키는 정규분포와 비슷한 형태로 구성되어 있음을 확인할 수 있음
2. 키가 100cm, 220cm인 사람도 존재하지만 이들을 이상치로 판단하고 후에 전처리 할 때 제거
```
- 여성

  </br>![image](https://github.com/TaeseongYang/Musinsa_project/assets/156265617/c574414b-3498-481b-87df-45e049e86048)
```
1. 여성의 키는 정규분포와 비슷한 형태로 구성되어 있음을 확인할 수 있음
2. 키가 100cm, 200cm인 사람도 존재하지만 이들을 이상치로 판단하고 후에 전처리 할 때 제거
```

② Weight

- 남성

  </br>![image](https://github.com/TaeseongYang/Musinsa_project/assets/156265617/87f81eb1-b788-4297-a0ba-726ddcc17cdc)
```
1. 남성의 몸무게는 정규분포와 비슷한 형태로 구성되어 있음 
2. 뭄무게가 140kg이상, 30kg이하인 고객은 이상치로 판별
```

- 여성

  </br>![image](https://github.com/TaeseongYang/Musinsa_project/assets/156265617/e3a92c02-b979-488f-a078-441cf5985fd8)
```
1. 여성의 몸무게는 정규분포와 비슷한 형태로 구성되어 있음 
2. 뭄무게가 120kg이상, 300kg이하인 고객은 이상치로 판별
```

# 분석 방법 (first topic)
</br>:clipboard: 라벨링 된 data에 대해서 학습 진행 

- 약 4,000개의 리뷰를 총창, 어깨너비, 가슴단면, 소매길이에 대해서 길고, 짤은지에 대해서 라벨링 진행
- 라벨링된 데이터를 바탕으로 딥러닝 학습을 진행하고 40,000개의 데이터에 대한 예측 작업 진행

```
Q. 라벨링의 기준은 무엇인가?

1. 총장, 어깨너비, 가슴단면, 소매길이를 지칭하지 않고 '오버핏이다, 사이즈가 크다, 루즈핏이다' 이런식으로 리뷰를 쓰는 경우가 많음
=> 이와 같은 경우에는 총장, 어깨너비, 가슴단면, 소매길이를 모두 크다(=1)로 라벨링을 진행합니다. 

2. '팔 혹시나 짧을까 했는데 여유 있어용.'과 같이 구체적인 리뷰 다수 존재하고 있음을 확인할 수 있음.
=> 이와 같은 경우에는 '소매가 짧다'로 라벨링을 진행

3. 핏이 이쁘다
=> 오버핏인지, 정핏인지 구분되지 않음
=> 따라서 이와 같은 경우에는 모든 항목에 0을 부여

4. 어깨가 편하다
=> 어깨가 편하면 여유로운 핏이기 때문에 편하다고 판단
=> 이와 같은 경우에는 '어깨너비 크다'로 라벨링을 진행

5. 기장이 크롭하다.
=> '길이가 짧다'로 라벨링 진행

6. 사이즈가 넉넉해서 좋은데 기장이 길다
=> 이는 모든 카테고리를 큰 것으로 라벨링을 진행

7. '세탁 후 줄어든다'와 같은 구매 이후 변경 사항에 대해서는 고려 x

8. 기장이 짧은 오버핏
=> 기장만 0으로 라벨링 하고 나머지 카테고리는 큰 것으로 라벨링을 진행

9. 마른 체형인데 체형 보완이 됐다.
=> 체형 보온이 되는 옷은 폼에 전체적으로 큰 것이기 때문에 모든 카테고리가 큰 것으로 라벨링을 진행
```
|리뷰|총장 길다|총장 짧다|어깨너비 크다|어깨너비 좁다|가슴 단면 크다 |가슴 단면 작다|소매 길이 길다|소매 길이 짧다|
|------|---|------|---|------|---|------|---|------|
|161/53 팔 긴편인데 손등 덮는 길이입니다. 색감은 피그먼트라 맘에들어요. 가을에 흰면치마 긴거랑 입어도 귀엽고 예쁩니다.|0|0|0|0|0|0|1|0|
|천이 너무 좋은 것 같아요.입으면 누가 봐도 합리적이라고 생각할 듯 합니다.가슴 폭도 넓고 어깨 쪽도 편하고 덩치 있어도 편하게 입울 수 있을 것 같네요|0|0|1|0|1|0|0|0|
|무난하게 이쁜 옷 ㅎㅎ 오버핏이라 정말 크긴 하네여.. M사이즈 해도 됐을듯|1|0|1|0|1|0|1|0|
|총장이 72인데 생각보다 짧네여...|0|1|0|0|0|0|0|0|	
|밑단 시보리가 생각보다 짱짱해요 예뻐요 딱 원하던 루즈핏|1|0|1|0|1|0|1|0|

</br>:clipboard: train_data와 test_data를 분류 

- target 변수가 1개이면, Stratified KFold를 통해 train_data와 test_data에 target class의 비율이 동일하게끔 분류 작업이 진행
- 하지만 리뷰에 따른 target변수가 총장 길다,짧다 등등 총 8개의 target변수가 설정되어 있음.
- 각 target의 비율을 최대한 유사한 비율로 하는 층화추출 방식을 통해 train_data와 test_data에 대한 분류 작업을 진핼할 예정
- MultilabelStratifiedKFold가 이러한 역할을 수행하는 라이브러리임

# 현재 진행 상태 

- 데이터 수집 : 약 30,000개의 리뷰 데이터 확보 => 추가적으로 10,000개의 리뷰 데이터 수집 예정
- 라벨링 : 30,000개의 리뷰 데이터 중에서 무작위로 5,000개의 리뷰 데이터 추출하였고 현재 1,600개의 리뷰 데이터에 대해 라벨링 진행 
