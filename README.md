# 뇌파 추상화: 상호작용형 시각 예술

## 💡 프로젝트 개요
이 프로젝트는 대학교 2학년 1학기에 진행된 머신러닝 과제인 ‘EEG 데이터를 활용한 알코올 중독자 분류’에서 출발한 토이 프로젝트입니다. 원래 프로젝트에서는 뇌파 데이터를 FFT(고속 푸리에 변환)로 주파수 영역 특성을 추출하고, 델타·세타·알파·베타·감마 대역별 중앙값을 주요 피처로 삼아 선형 커널 SVM으로 알코올 중독자를 분류했습니다.

분석을 진행하며 뇌파 데이터를 단순한 2D 그래프가 아니라 3D 공간에서 시각화하면 좋겠다는 생각이 들었습니다. 별자리와 은하의 형상을 차용해 인터랙티브 예술 작품처럼 재구성했으며, 사용자는 포인트 크기를 조절하고 랜덤성을 설정하면서 실시간으로 다양한 시각 패턴을 탐색할 수 있습니다.

## ✨ 주요 기능
*   **인터랙티브 뇌파 시각화**: 뇌파 밴드(델타, 세타, 알파, 베타, 감마)의 파워 데이터를 3D 산점도 형태로 시각화합니다.
*   **별자리 컨셉**: 각 뇌파 측정치를 '별'처럼 표현하며, 파워 값의 로그에 비례하여 별의 크기가 달라집니다.
*   **2D에서 3D로의 전환**: Z축 변동성 슬라이더를 통해 초기 2D 평면 시각화에서 3D 별자리 형태로 동적으로 전환됩니다. 변동성은 뇌파 파워의 로그 값에 비례하여 적용되어 파워가 높은 별들이 더 큰 Z축 변동성을 가집니다.
*   **사용자 정의 옵션**:
    *   **포인트 크기**: 별의 전체적인 크기를 조절합니다. (기본값: 15, 범위: 1-30)
    *   **Y축 변동성**: 뇌파 밴드 간의 Y축 위치에 무작위 변동성을 추가하여 연속적인 스펙트럼처럼 보이게 합니다. (기본값: 0, 범위: 0-100)
    *   **Z축 변동성**: Z축에 무작위 변동성을 추가하여 별자리 효과를 강화합니다. (기본값: 0, 범위: 0-100)
    *   **색상 스케일**: 알코올 그룹과 비중독자 그룹의 색상 스케일을 선택할 수 있습니다. (기본값: Reds, Blues)
    *   **그룹 표시/숨기기**: 알코올 그룹과 비중독자 그룹의 시각화를 개별적으로 켜고 끌 수 있습니다.
    *   **채널 인덱스 범위**: 시각화할 뇌파 채널의 범위를 선택합니다.
    *   **뇌파 밴드 선택**: 특정 뇌파 밴드만 선택하여 시각화할 수 있습니다.
*   **직관적인 UI**: Streamlit을 기반으로 한 사용자 친화적인 인터페이스를 제공합니다.

## 🚀 설치 및 실행 방법

### 1. 환경 설정
Python 3.8 이상 버전이 설치되어 있어야 합니다. 가상 환경을 사용하는 것을 권장합니다.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. 의존성 설치
프로젝트에 필요한 라이브러리를 설치합니다. 

```bash
pip install streamlit numpy scipy pandas plotly
```
(참고: `ML2024_project_EEG_classification.ipynb`에 추가적인 머신러닝 관련 라이브러리(예: scikit-learn)가 필요할 수 있습니다.)

### 3. 데이터 준비
`seoultech-applied-ai-machine-learning1/train.npy` 파일이 프로젝트 루트 디렉토리에 올바르게 위치해 있는지 확인하십시오. 이 파일은 뇌파 데이터와 레이블을 포함하고 있습니다.

### 4. 애플리케이션 실행
Streamlit 애플리케이션을 실행합니다.

```bash
streamlit run eeg_galaxy.py
```
명령어를 실행하면 웹 브라우저에서 애플리케이션이 열립니다.

## 📁 프로젝트 구조
```
├── .gitignore
├── .venv/                       # Python 가상 환경 폴더
├── eeg_galaxy.py              # 뇌파 시각화 Streamlit 애플리케이션
├── split_train_data.py        # train.npy 파일을 분할하는 스크립트
├── ML project/ # 진행했던 머신러닝 프로젝트
│   ├── ML2024_project_EEG_classification.ipynb # 뇌파 분류 머신러닝 모델 개발
│   ├── SVM classifier.csv         # SVM 분류기 결과
│   └── seoultech-applied-ai-machine-learning1/
│       └── chunks/                # 분할된 train.npy 데이터 청크
│           ├── train_X_part_0.npy
│           ├── train_X_part_1.npy
│           └── ... (나머지 청크 파일들)
├── README.md                  
```

## 🖼️ 데모 화면
![image](https://github.com/user-attachments/assets/9f9657d0-0e44-438b-b5c6-71f4295a0fa1)
![image](https://github.com/user-attachments/assets/235e2561-de2a-4d8e-8708-18ea4961f3f0)
![image](https://github.com/user-attachments/assets/5a1d1de3-aa16-4c9c-9bd3-7772a3f75728)


