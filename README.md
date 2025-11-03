# AGI 발현을 위한 메타인지 프레임워크 핵심기술 개발 및 실증
## AI 모델의 내재된 지식을 기반으로 한 Certain / Uncertain (Seen / Unseen) 여부 판단 기법
### 💡 예시
- AI 모델이 **certain**이라고 판단한 경우

![image](./image/ex_certain.png)

- AI 모델이 **uncertain**이라고 판단한 경우 (= 추가 정보가 필요하다고 판단한 경우)

![image](./image/ex_uncertain.png)

## ⚙️ Requirements
To install requirements:
```
pip install -r requirements.txt
```

## 💻 사용 방법
### Step 1. Dataset 준비
- 자세한 내용은 [README.md](data/README.md)를 참고해주세요.

### Step 2. Run the Classifier
```
python src/classifier.py --dataset_name "dataset_name" --model_name "model_name"
```
