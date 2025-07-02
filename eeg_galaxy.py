import streamlit as st
import numpy as np
import scipy.signal
import pandas as pd
import plotly.graph_objects as go
import os

# EEG feature extraction 함수
@st.cache_data

def extract_eeg_features(input_data, labels, fs=256, nperseg=256):
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 31), 'Gamma': (31, 46)}
    rows = []
    for subj_idx in range(input_data.shape[0]):
        label = labels[subj_idx]
        for ch_idx in range(input_data.shape[2]):
            signal = input_data[subj_idx, :, ch_idx]
            freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)
            for name, (low, high) in bands.items():
                power = np.median(psd[(freqs >= low) & (freqs < high)])
                rows.append({'Subject': subj_idx, 'Label': 'Alcoholic' if label == 1 else 'Non-Alcoholic',
                             'Channel': ch_idx, 'Band': name, 'Power': power})
    return pd.DataFrame(rows)

# 데이터 로드 함수
@st.cache_data
def load_data():
    chunks_dir = 'ML project/seoultech-applied-ai-machine-learning1/chunks'
    num_chunks = 10 # Assuming 10 chunks were created

    X_chunks = []
    y_chunks = []

    for i in range(num_chunks):
        X_chunk_path = os.path.join(chunks_dir, f'train_X_part_{i}.npy')
        y_chunk_path = os.path.join(chunks_dir, f'train_y_part_{i}.npy')
        X_chunks.append(np.load(X_chunk_path, allow_pickle=True))
        y_chunks.append(np.load(y_chunk_path, allow_pickle=True))

    x = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    x = x.reshape(-1, 256, 64) # Ensure correct reshape after concatenation
    return x, y

# 집계된 평균 파워 계산
@st.cache_data
def get_aggregated():
    x, y = load_data()
    df = extract_eeg_features(x, y)
    agg_alc = df[df['Label']=='Alcoholic'].groupby(['Channel','Band'])['Power'].mean().reset_index()
    agg_non = df[df['Label']=='Non-Alcoholic'].groupby(['Channel','Band'])['Power'].mean().reset_index()
    return agg_alc, agg_non

# 앱 레이아웃 설정
st.set_page_config(layout='wide')
st.title('뇌파 추상화: 상호작용형 시각 예술')
st.markdown('뇌파 데이터를 기반으로 사용자가 실시간으로 조작하며 별자리처럼 만들어볼 수 있는 인터랙티브 아트워크')


# 사이드바 위젯
st.sidebar.header('시각화 옵션')
st.sidebar.markdown('**뇌파 밴드 선택**')
band_list = ['Delta','Theta','Alpha','Beta','Gamma']
selected_bands = []
for band in band_list:
    if st.sidebar.checkbox(band, value=True):
        selected_bands.append(band)
channel_range = st.sidebar.slider('채널 인덱스 범위', 1, 64, (1,64))
point_size = st.sidebar.slider('포인트 크기', 1.0, 30.0, 15.0, step=0.1)
y_spread = st.sidebar.slider('Y축 변동성', 0.0, 100.0, 0.0, step=0.1)
z_spread = st.sidebar.slider('Z축 변동성', 0.0, 100.0, 0.0, step=1.0)
red_scale = st.sidebar.selectbox('알코올 그룹 색상', ['Reds', 'Plasma', 'Inferno', 'Viridis'], index=0)
blue_scale = st.sidebar.selectbox('비중독자 그룹 색상', ['Blues', 'Plasma', 'Inferno', 'Viridis'], index=0)
show_alc = st.sidebar.checkbox('알코올 그룹 표시', value=True)
show_non = st.sidebar.checkbox('비중독자 그룹 표시', value=True)

# 데이터 준비
with st.spinner('데이터 준비 중…'):
    agg_alc, agg_non = get_aggregated()

# 필터링: 밴드, 채널
start, end = channel_range
mask_alc = agg_alc['Band'].isin(selected_bands) & agg_alc['Channel'].between(start-1, end-1)
mask_non = agg_non['Band'].isin(selected_bands) & agg_non['Channel'].between(start-1, end-1)
alc = agg_alc[mask_alc].copy()
non = agg_non[mask_non].copy()

# Y축 매핑
band_to_y = {'Delta':1, 'Theta':2, 'Alpha':3, 'Beta':4, 'Gamma':5}
band_to_y = {'Delta': 2.5, 'Theta': 6, 'Alpha': 11, 'Beta': 22.5, 'Gamma': 38.5}
alc['Y'] = alc['Band'].map(band_to_y) + np.random.uniform(-y_spread/2, y_spread/2, len(alc))
non['Y'] = non['Band'].map(band_to_y) + np.random.uniform(-y_spread/2, y_spread/2, len(non))

# Z축 (랜덤성) 추가 및 포인트 크기 조절
alc['Z'] = np.random.uniform(-z_spread/2, z_spread/2, len(alc)) * np.log1p(alc['Power'])
non['Z'] = np.random.uniform(-z_spread/2, z_spread/2, len(non)) * np.log1p(non['Power'])

alc['Size'] = np.log1p(alc['Power']) * point_size / np.log1p(alc['Power'].max())
non['Size'] = np.log1p(non['Power']) * point_size / np.log1p(non['Power'].max())

# 3D Scatter 그리기 (별자리 컨셉)
fig = go.Figure()
marker_opacity = 0.8 # 별 투명도

if show_alc:
    fig.add_trace(go.Scatter3d(
        x=alc['Channel'],
        y=alc['Y'],
        z=alc['Z'],
        mode='markers',
        marker=dict(
            size=alc['Size'],
            color=alc['Power'], # 파워 값에 따라 색상 변화
            colorscale=red_scale,
            opacity=marker_opacity,
            colorbar=dict(title='Power', x=0.9, y=0.75, len=0.4), # 색상바 추가
            showscale=True # 색상 스케일 표시
        ),
        hovertemplate='그룹: Alcoholic Group<br>채널 %{x}<br>밴드: %{customdata[0]}<br>파워: %{customdata[1]:.2f}<extra></extra>',
        customdata=np.stack([alc['Band'], alc['Power']], axis=-1), # 밴드 이름과 파워를 customdata로 전달
        name='Alcoholic Group'
    ))
if show_non:
    fig.add_trace(go.Scatter3d(
        x=non['Channel'],
        y=non['Y'],
        z=non['Z'],
        mode='markers',
        marker=dict(
            size=non['Size'],
            color=non['Power'], # 파워 값에 따라 색상 변화
            colorscale=blue_scale,
            opacity=marker_opacity,
            colorbar=dict(title='Power', x=0.9, y=0.25, len=0.4), # 색상바 추가
            showscale=True # 색상 스케일 표시
        ),
        hovertemplate='그룹: Non-Alcoholic Group<br>채널 %{x}<br>밴드: %{customdata[0]}<br>파워: %{customdata[1]:.2f}<extra></extra>',
        customdata=np.stack([non['Band'], non['Power']], axis=-1), # 밴드 이름과 파워를 customdata로 전달
        name='Non-Alcoholic Group'
    ))

# 레이아웃 최적화
fig.update_layout(
    scene=dict(
        yaxis=dict(tickmode='array', tickvals=list(band_to_y.values()),
                   ticktext=list(band_to_y.keys()), showgrid=False, zeroline=False, title='', showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='')
    ),
    margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='black', plot_bgcolor='black', font_color='white',
    showlegend=False # 범례 숨기기
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})