# 뇌파 데이터 기반 알코올 중독자 분류 및 3D 시각화

> 뇌파(EEG) 데이터를 FFT로 분석하여 알코올 중독자를 분류하고, 그 결과를 별자리·은하 형상의 인터랙티브 3D 아트로 재구성한 토이 프로젝트입니다.

---

## 프로젝트 배경

서울과학기술대학교 머신러닝 과목 팀 프로젝트에서 출발했습니다.

- **FFT**(고속 푸리에 변환)로 뇌파 데이터의 주파수 영역 특성을 추출
- 델타 · 세타 · 알파 · 베타 · 감마 대역별 중앙값을 피처로 사용
- SVM으로 알코올 중독자를 분류 → **테스트 정확도 약 97%**

데이터 분석 과정에서 뇌파를 단순 2D 그래프가 아닌 **3D 공간**에 시각화하면 흥미롭겠다는 아이디어가 떠올라, 사용자가 파라미터를 조절하며 별자리와 은하 형상을 만들 수 있는 **인터랙티브 시각화 대시보드**로 확장했습니다.

---

## 주요 기능

### 3D 뇌파 시각화

뇌파 밴드(δ, θ, α, β, γ)의 파워 데이터를 3D 산점도로 렌더링합니다. 각 측정치는 별처럼 표현되며, 파워 값의 로그에 비례하여 크기가 달라집니다.

### 사용자 정의 옵션

| 옵션 | 설명 | 기본값 | 범위 |
|------|------|--------|------|
| 포인트 크기 | 전체 포인트 크기 조절 | 15 | 1 – 30 |
| Y축 변동성 | 뇌파 밴드 간 Y축 랜덤 오프셋 추가 | 0 | 0 – 100 |
| Z축 변동성 | Z축 랜덤 오프셋 추가 | 0 | 0 – 100 |
| 색상 스케일 | 알코올 / 비중독자 그룹별 색상 선택 | Reds / Blues | — |
| 그룹 표시 | 각 그룹 개별 ON/OFF | — | — |
| 채널 범위 | 시각화할 EEG 채널 인덱스 범위 | — | — |
| 밴드 선택 | 특정 뇌파 밴드만 필터링 | 전체 | δ θ α β γ |

---

## 설치 및 실행

### 1. 환경 설정

> Python 3.8+ 필요. 가상 환경 사용을 권장합니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install streamlit numpy scipy pandas plotly
```

> ML 노트북(`ML2024_project_EEG_classification.ipynb`) 실행 시 `scikit-learn` 등 추가 라이브러리가 필요할 수 있습니다.

### 3. 데이터 준비

`seoultech-applied-ai-machine-learning1/train.npy` 파일이 프로젝트 루트에 위치해 있는지 확인하세요.

### 4. 실행

```bash
streamlit run eeg_galaxy.py
```

브라우저에서 자동으로 대시보드가 열립니다.

---

## 프로젝트 구조

```
├── eeg_galaxy.py                          # Streamlit 3D 시각화 대시보드
├── split_train_data.py                    # train.npy 데이터 분할 스크립트
├── ML project/
│   ├── ML2024_project_EEG_classification.ipynb   # EEG 분류 모델 (SVM)
│   ├── SVM classifier.csv                        # 분류 결과
│   └── seoultech-applied-ai-machine-learning1/
│       └── chunks/                               # 분할된 데이터 청크
│           ├── train_X_part_0.npy
│           ├── train_X_part_1.npy
│           └── ...
├── README.md
└── .gitignore
```

---

## 실행 화면
![image](https://github.com/user-attachments/assets/9f9657d0-0e44-438b-b5c6-71f4295a0fa1)
![image](https://github.com/user-attachments/assets/235e2561-de2a-4d8e-8708-18ea4961f3f0)
![image](https://github.com/user-attachments/assets/5a1d1de3-aa16-4c9c-9bd3-7772a3f75728)


