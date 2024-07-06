

* * *
# 주요 Initialization 소개

</br>

## Zero (Same) Initialization
- 가중치를 0으로 초기화하는 방법
- 활성함수를 ReLU, tanh 사용시 가중치가 전혀 업데이트되지 않음
- 활성함수를 Sigmoid 사용시 가중치가 미미하게 업데이트 되어 학습이 잘 되지 않으며, 모두 같은 값으로 가중치가 업데이트되는 문제가 발생
  - 단, Back Propagation (Softmax + Cross Entropy) 과정에서, **제일 마지막 레이어는 가중치 값에 영향을 받지 않음**

## Random Initialization
- 가중치를 랜덤하게 설정하는 방법
- 랜덤하게 설정하므로 서로 다른 값을 갖도록 할 수 있음 (Zero Initialization의 문제를 해결 할 수 있음)
- Activation Value가 Sigmoid의 0.5로, ReLU나 tanh는 0으로 과도하게 모여 학습/수렴 속도가 저하될 수 있음
- 극단적인 경우에 모든 가중치 값들이 1이 되거나 -1이 될 수 있음

## [Lecun Initialization](https://www.nature.com/articles/nature14539)
- ReLU 등장 전에 Input의 크기를 고려하여 분산을 조정하는 방법
- Normal(혹은 Uniform) Distribution를 이용하여 가중치를 초기화하고, 입력 뉴런의 수에 반비례하도록 분산을 조정하는 방법
- 범위를 벗어나는 특이값들을 배제 할 수 있음 (Random Initialization의 극단적인 경우의 문제를 해결 할 수 있음)

## [Xavier (Glorot) Initialization](https://proceedings.mlr.press/v9/glorot10a)
- Input과 Output의 크기를 모두 고려하여 분산을 조정하는 방법
- Normal(혹은 Uniform) Distribution를 이용하여 가중치를 초기화하고, 2를 곱한 후 입력 및 출력 뉴런의 수에 반비례하도록 분산을 조정하는 방법
- **선형 활성함수(tanh, sigmoid 등)에 적합한 방법**
- sigmoid를 사용하면 E[X]=E[Y]=0 의 가정을 위배하므로 Xavier Initialization이 성립하지 않음
  - 그러나 [후속 논문](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_26)에 따르면 **Xavier Initialization의 표준 편차에 4를 곱한 값을 이용하면 실험적으로 잘 동작**한다고 함
  - ReLU 활성함수에 다소 비효율적인 문제가 발생함

## [(Kaiming) He Initialization](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
- ReLU 등장 후에 Input의 크기를 고려하여 분산을 조정하는 방법
- **비선형 활성함수(ReLU, Leaky ReLU 등)에 적합한 방법**으로, 각 레이어를 통과하는 가중치의 크기를 유지하도록 도와줌
- Lecun Initialization에 2를 곱하거나, Xavier Initialization에서 출력 뉴런의 수에 반비례하는 부분만 제거하는 방법
  - Xavier Initialization의 가정 중 하나는 활성함수가 Linear해야 하는데, 이는 ReLU에서 성립하지 않은 문제가 발생함
- 레이어의 깊이가 얕은 모델에서 Xavier Initialization에 비해 He Initialization이 더 빨리 학습하고 성능이 우수했음
- 레이어의 깊이가 깊은 모델에서 Xavier Initialization는 학습을 전혀 하지 못했지만, He Initialization을 사용했을 때는 학습을 잘 함

## [Orthogonal Initialization](https://arxiv.org/abs/1312.6120)
- SVD(Singular Value Decomposition)으로 가중치 행렬이 모두 수직(직교)이 되도록 만드는 방법
- 각각의 벡터가 서로 직교하므로, 독립적으로 학습 될 수 있어 서로 다른 특성에 집중하게 할 수 있음
- **심층 신경망 모델에서 정보의 흐름을 안정화**(Vanishing Gradient 방지)하는데 도와줌
- ReLU 활성함수와 결합하여 보다 안정적으로 학습이 이루어질 수 있도록 도와줌
- 심층 신경망으로 구성된 RNN(LSTM)이나 CNN 모델에서 유용하게 사용됨
- 특정 값이 지나치게 커지거나 작아질 수 있어 출력 값이 들쑥날쑥하게 만들어 학습을 어렵게 만들 수 있음

* * *

</br></br>

* * *
코드 상으로 Initialization 결과 비교 내용 추가하기 ([참고 자료](https://alltommysworks.com/%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94/))
* * *


</br></br>

* * *
균일 분포 vs 정규 분포 관련 내용 추가하기
* * *

</br></br>

* * *
배치 정규화 (Batch Normalization) 와 관련된 내용 추가하기
* * *


