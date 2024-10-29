# Humanoid Project: 복합 환경에서의 로봇 강화학습

### 📌 프로젝트 개요
이 프로젝트는 TD-MPC2 알고리즘과 YOLO 기반 시각 인식 모델을 활용하여 복합 환경에서 로봇이 여러 장애물을 스스로 학습해 해결하도록 하는 강화학습 연구입니다.
[휴머노이드 벤치(HumanoidBench)](https://github.com/carlosferrazza/humanoid-bench)를 활용해 걷기, 폴대 통과, 허들 넘기기, 슬라이드 오르기 등의 태스크를 다루었습니다.

### 🚀 주요 기능 및 특징
* TD-MPC2 알고리즘: 복합 환경에서의 강화학습을 최적화합니다.
* YOLO 기반 시각 인식: 장애물을 실시간으로 인식하고 학습에 반영합니다.
* 다양한 보상 함수 실험: 태스크에 맞는 최적의 보상 구조를 설계합니다.
* 복합 장애물 환경: 여러 장애물이 혼합된 환경에서의 적응 학습.

### 📂 프로젝트 구조
```
humanoid-bench/
│
├── classifier/           # 장애물 분류 및 Task 인식 모델
├── custom_env/           # 사용자 정의 환경 설정
├── data/                 # 학습 및 평가 데이터
├── dreamerv3/            # Dreamer V3 알고리즘 구현
├── humanoid_bench/       # 환경 및 실험 관련 설정 코드
├── jaxrl_m/              # JAXRL-M 구현 모듈
├── ppo/                  # PPO 알고리즘 구현
├── tdmpc2/               # TD-MPC2 알고리즘 구현
│
├── .gitignore            # Git 관리 제외 파일 목록
├── basic_graph_hurdle.sh # 허들 실험 시각화 스크립트
├── basic_graph_pole.sh   # 폴대 통과 실험 시각화 스크립트
├── basic_graph_stair.sh  # 계단 오르기 실험 시각화 스크립트
├── basic_graph_walk.sh   # 걷기 실험 시각화 스크립트
├── LICENSE               # 라이선스 정보
├── README.md             # 프로젝트 설명 파일
├── requirements_dreamerv3.txt # Dreamer V3 종속성 목록
├── requirements_jaxrl.txt     # JaxRL 종속성 목록
├── requirements_tdmpc2.txt    # TD-MPC2 종속성 목록
├── setup.py              # 패키지 설정 스크립트
└── test_env_img.png      # 테스트 환경 이미지
```

### 🛠️ 설치 방법
1. 필수 의존성 설치
먼저 리포지토리를 클론합니다:
```
git clone https://github.com/your-repository/humanoid-bench.git
cd humanoid-bench
```

가상 환경을 생성한 후 활성화합니다:
```
conda create -n humanoidbench python=3.11
conda activate humanoidbench
```

필요한 종속성을 설치합니다:

```
pip install -r requirements_tdmpc2.txt
pip install -r requirements_dreamerv3.txt
pip install -r requirements_jaxrl.txt
```

### 🚀 사용 방법
1. 학습 시작
아래 명령어를 통해 TD-MPC2 알고리즘으로 학습을 시작합니다:

```
python -m tdmpc2.train disable_wandb=False wandb_entity=[WANDB_ENTITY] exp_name=tdmpc task=humanoid_${TASK} seed=0
```

### 📊 실험 결과

### ⚙️ 주요 기능 및 알고리즘

TD-MPC2: 복합 장애물 환경에서 로봇의 적응적 학습 지원
Dreamer V3: 장기적 예측을 통해 모델 성능 개선
PPO: 정책 최적화 강화
YOLO 통합: 실시간 장애물 인식

### 💡 어려움과 해결 방법
**폴대 통과에서의 문제 해결**
* 문제: 로봇이 폴대에 기대면서 통과하려는 경향이 있었습니다.
* 해결: 폴대 접촉에 대한 패널티를 강화하여 문제를 해결했습니다.

**슬라이드 환경에서의 문제**
* 문제: 로봇이 벽에 기대면서 경사를 올라가려는 문제가 발생했습니다.
* 해결: 벽 접촉 패널티를 추가하고 보상 구조를 조정했습니다.


### 📅 향후 계획

상부 메타컨트롤러 개발을 통한 더 정교한 제어 기능 추가
YOLO 모델의 성능 개선: 데이터셋 확대와 추가 훈련
실시간 로봇 제어를 위한 실제 환경 확장

### 👥 팀 구성

* 윤소정: 커스텀 환경 설계 및 장애물 분류
* 김주현: PPO 알고리즘 개발 및 실험 지원
* 신재현: TD-MPC2 연구 및 학습 구조 개선
* 현동철: 장애물 테스트 및 시각화