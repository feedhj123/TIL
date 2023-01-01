> # 머신러닝
>> ## Confusion Matrix(오차행렬)
![이미지](https://www.superheuristics.com/wp-content/uploads/2021/03/Blog_image_confusion-matrix-740x414.png)
- TP: 옳은 검출
- FN: 검출되어야 할 것이 검출되지 않았음
- FP: 틀린 검출
- TN: 검출되지 말아야 할 것이 검출되지 않았음

### precision(정밀도)
- 모델이 True라고 한 것중 실제True의 비율
- 맑다고 예측했는데 실제로 맑은날인 경우
- 예측한 결과가 실제 결과와 얼마나 일치하는가
- 모델 중심 
-  $(precision) =\dfrac{TP}{TP+FP}$

### Recall(재현율)
- 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
- 실제 맑은날 중에 모델이 맑다고 예측한 비율
- 사람 중심
- $(Recall) = \dfrac{TP}{TP+FN}$

### 정확도(accuracy)
- 모델이 입력된 데이터에 대해 얼마나 정확하게 예측하는지를 나타낸다. 
- 실제 True를 모델이 True라고 예측한것 + 실제 False를 모델이 False라고 예측한 것의 비율
위 confusion matrix에서  $Acc=\dfrac{TP+TN}{TP+FN+FP+TN}$
​

### 재현율과 정밀도
재현율 vs 정밀도
재현율과 정밀도는 사용하는 경우에 따라서 중요도가 다를수 있다.
1. 재현율이 중요한 경우
 
 **실제 Postive 인 데이터를 Negative로 잘못 판단하면 안되는 경우**
 
 병 진단 : 실제 양성 인데 음성 으로 판단하면 병을 더 키울수 있다.

2.  정밀도가 더 중요한 경우 

 **실제 Negative 인 데이터를 Postive로 잘못판단하면 안되는 경우**
 
 스팸 메일 : 실제 스펨메일이 아닌데(Negative) 스펨메일(Postive)로 판단하는 경우
 메일을 받지 못할수 있다.
 ﻿

## Confidence Threshold
- Precision-Recall(PR) 곡선
  - PR 곡선은 confidence 레벨에 대한 threshold 값의 변화에 의한 물체 검출기의 성능을 평
가하는 방법이다.
  - confidence 레벨은 검출한 것에 대해 알고리즘이 얼마나 확신이 있는지를 알려주는 값
  - 만약에 어떤 물체를 검출했는데 confidence 레벨이 0.999라면 굉장히 큰 확신을 가지고 검
    출한 것이다.>>> **여기서 확신의 주체는 '알고리즘'임 사람이 아니라**

### Threshold
- confidence 레벨이 높다고 해서 무조건 적으로 검출이 정확하다고 볼 수 없음
- 알고리즘의 사용자는 보통 confidence 레벨에 대해 threshold 값을 부여해서 특정 값 이상이 되어야 검출된 것으로 인정
   - threshold값 미만의 confidence 레벨로 검출된 결과는 무시하는 방식


- Confidence Threshold 값의 변화에 따라 정밀도-재현율은 변화한다.

- Confusion Matrix 5
 
 - Confidence Threshold 값이 낮을 수록 더많은 예측 Bounding Box 를 만들게 된다.
 
 - (정밀도는 낮아지고 재현율은 높아짐)

 - Confidence Threshold 값이 높을 수록 신중하게 예측 Bounding Box 를 만들게 된다.

 - (정밀도는 높아지고 재현율은 낮아짐)

 - **Confidence Threshold 를 조정하면 정밀도 또는 재현율의 수치가 조정되고, 이는 서로 상
   보적이기 때문에 Trad-off 가 이루어 진다.**
   ![이미지](https://miro.medium.com/max/1248/1*TqzfzabXrej1FTdZuNNYIQ.png)
 - Recall 값의 변화에 따른 Precision 값을 나타낸 곡선을 정밀도 재현율 곡선이라 한다.
 Precision 값의 평균을 AP 라고 하며, 면적값으로 계산 된다.


