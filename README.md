

* * *
# 주요 Initialization 소개

</br>

## Zero (Same) Initialization
- 가중치를 0으로 초기화하는 방법
- 활성함수를 ReLU, tanh 사용시 가중치가 전혀 업데이트되지 않음
- 활성함수를 Sigmoid 사용시 가중치가 미미하게 업데이트 되어 학습이 잘 되지 않으며, 모두 같은 값으로 가중치가 업데이트되는 문제가 발생
  - 단, Back Propagation (Softmax + Cross Entropy) 과정에서, **제일 마지막 레이어는 가중치 값에 영향을 받지 않음**

</br>

## Random Initialization
- 가중치를 랜덤하게 설정하는 방법
- 랜덤하게 설정하므로 서로 다른 값을 갖도록 할 수 있음 (Zero Initialization의 문제를 해결 할 수 있음)
- Activation Value가 Sigmoid의 0.5로, ReLU나 tanh는 0으로 과도하게 모여 학습/수렴 속도가 저하될 수 있음
- 극단적인 경우에 모든 가중치 값들이 1이 되거나 -1이 될 수 있음

</br>

## [Lecun Initialization](https://www.nature.com/articles/nature14539)
- ReLU 등장 전에 Input의 크기를 고려하여 분산을 조정하는 방법
- Normal(혹은 Uniform) Distribution를 이용하여 가중치를 초기화하고, 입력 뉴런의 수에 반비례하도록 분산을 조정하는 방법
- 범위를 벗어나는 특이값들을 배제 할 수 있음 (Random Initialization의 극단적인 경우의 문제를 해결 할 수 있음)

</br>

## [Xavier (Glorot) Initialization](https://proceedings.mlr.press/v9/glorot10a)
- Input과 Output의 크기를 모두 고려하여 분산을 조정하는 방법
- Normal(혹은 Uniform) Distribution를 이용하여 가중치를 초기화하고, 2를 곱한 후 입력 및 출력 뉴런의 수에 반비례하도록 분산을 조정하는 방법
- **선형 활성함수(tanh, sigmoid 등)에 적합한 방법**
- sigmoid를 사용하면 E[X]=E[Y]=0 의 가정을 위배하므로 Xavier Initialization이 성립하지 않음
  - 그러나 [후속 논문](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_26)에 따르면 **Xavier Initialization의 표준 편차에 4를 곱한 값을 이용하면 실험적으로 잘 동작**한다고 함
  - ReLU 활성함수에 다소 비효율적인 문제가 발생함

</br>

## [(Kaiming) He Initialization](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
- ReLU 등장 후에 Input의 크기를 고려하여 분산을 조정하는 방법
- **비선형 활성함수(ReLU, Leaky ReLU 등)에 적합한 방법**으로, 각 레이어를 통과하는 가중치의 크기를 유지하도록 도와줌
- Lecun Initialization에 2를 곱하거나, Xavier Initialization에서 출력 뉴런의 수에 반비례하는 부분만 제거하는 방법
  - Xavier Initialization의 가정 중 하나는 활성함수가 Linear해야 하는데, 이는 ReLU에서 성립하지 않은 문제가 발생함
- 레이어의 깊이가 얕은 모델에서 Xavier Initialization에 비해 He Initialization이 더 빨리 학습하고 성능이 우수했음
  - Xavier 초기화보다 분산이 더 크므로, 초기 신호가 좀 더 강하게 전달될 수 있음
- 레이어의 깊이가 깊은 모델에서 Xavier Initialization는 학습을 전혀 하지 못했지만, He Initialization을 사용했을 때는 학습을 잘 함

</br>

## [Orthogonal Initialization](https://arxiv.org/abs/1312.6120)
- SVD(Singular Value Decomposition)으로 가중치 행렬이 모두 수직(직교)이 되도록 만드는 방법
- 각각의 벡터가 서로 직교하므로, 독립적으로 학습 될 수 있어 서로 다른 특성에 집중하게 할 수 있음
- **심층 신경망 모델에서 정보의 흐름을 안정화**(Vanishing Gradient 방지)하는데 도와줌
- ReLU 활성함수와 결합하여 보다 안정적으로 학습이 이루어질 수 있도록 도와줌
- 심층 신경망으로 구성된 RNN(LSTM)이나 CNN 모델에서 유용하게 사용됨
- 특정 값이 지나치게 커지거나 작아질 수 있어 출력 값이 들쑥날쑥하게 만들어 학습을 어렵게 만들 수 있음

</br>

## 이 밖에도 다양한 초기화 방법이 있으나, 이는 프레임워크에 따라 공식 문서를 참조

* * *

</br></br>

* * *
코드 상으로 Initialization 결과 비교 내용 추가하기 ([참고 자료](https://alltommysworks.com/%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94/))
* * *


</br></br>

* * *

## 균등 분포 (Uniform Distribution)
- 균등한 분포, 모든 값 동일 확률하므로 범위를 조절 할 수 있음
- 프로토타입 모델 및 간단한 모델에 유용
- 심층 신경망에 적용 할 경우, 기울기 폭발과 같은 문제가 발생 할 수 있음

</br>

## 정규 분포 (Normal Distribution)
- 대칭적이고, 가운데가 높이 솟아 있는 종 형이며, 평균이 0 이므로 분산을 조절 할 수 있음
  - weight의 분포가 너무 평균 근처에 집중되는 경우(분산이 작은 경우)
    - 학습 속도가 느려지거나, 정확도가 초기에 낮을 수 있음
  - weight의 분포가 너무 넓은 영역에 펴지는 경우(분산이 큰 경우)
    - 보다 다양한 특징을 학습 할 수는 있으나, 학습이 불안정하거나, 기울기 폭발이 발생하여 학습이 전혀 되지 않을 수 있음
    - 과적합(Over Fitting)이나 일반화 성능이 하락 할 수 있음
- 심층 신경망에 적용 할 경우, 기울기 소실 및 기울기 폭발 가능성이 낮아 안정적으로 학습이 되어 유용함
- ReLU 계열의 활성함수들은 입력 값이 0 이하인 경우 모두 0으로 출력하기 때문에, 정규 분포로 초기화시 이 영향을 받을 수 있어 Truncated Normal Distribution 을 사용 할 수 있음

</br>

## 심층 신경망 모델을 설계 할 때, 기본적으로는 정규 분포를 사용하고 신경망의 초기 입력 부분에만 균등 분포를 사용하면 어떨까?
- 초기에 대칭성을 깨거나 빠르게 모델의 성능을 탐색하기에 유리할 수 있음
- 다만, 학습의 일관성과 안정성을 유지하기 위해 실험적 검증과 적절한 튜닝이 필요
- 모델의 구조나 데이터셋의 특징에 따라 최적의 방법은 다를 수 있으므로, 다양한 초기화 방법을 테스트하여 최적의 성능을 얻는 것이 필요 할 것으로 예상됨

* * *


</br></br>

* * *

## PyTorch 
- 공식 github의 코드를 살펴보면 다음과 같음
  ```python
  def reset_parameters(self) -> None:
      # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
      # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
      # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
          fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
          if fan_in != 0:
              bound = 1 / math.sqrt(fan_in)
              init.uniform_(self.bias, -bound, bound)
  ```
  - initializer가 한 번 바뀐적이 있는데, 이 중 **a=math.sqrt(5)**에 대해서는 이슈가 있음
    - 코드에서도 나와있듯이 [관련 이슈 링크](https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573)를 살펴보면,
      "So the sqrt(5) is nothing more than giving the code the same end-result as before." 라고 언급하고 있음
  - 따라서, 활성함수를 어떤 것을 사용하는가, 모델의 크기나 용도, 도메인(데이터셋)의 특성은 어떠한지에 따라 사용자가 별도로 초기화를 해야 할 필요가 있음

</br>

## TensorFlow / Keras
- 공식 github의 코드를 살펴보면 다음과 같음
  ```python
  if initializer is None:
      if "float" in dtype:
          initializer = "glorot_uniform"
      else:
          initializer = "zeros"
  ```
- Tensorflow는 현재 glorot_uniform을 default로 지정하여 사용하고 있음
- 마찬가지로 활성함수를 어떤 것을 사용하는가, 모델의 크기나 용도, 도메인(데이터셋)의 특성은 어떠한지에 따라 사용자가 별도로 초기화를 해야 할 필요가 있음

* * *
