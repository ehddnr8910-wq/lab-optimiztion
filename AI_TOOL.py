import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 함수
def setup_korean_font():
    # Streamlit Cloud (Linux) 환경의 나눔고딕 폰트 경로
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    # 폰트 파일이 존재하는지 확인 (서버 환경)
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        # 윈도우(로컬) 환경일 경우 'Malgun Gothic' 사용
        plt.rc('font', family='Malgun Gothic')
    
    # 마이너스(-) 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 함수 실행
setup_korean_font()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import platform
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# [한글 폰트 설정] Matplotlib 한글 깨짐 해결 (OS 자동 감지)
# -----------------------------------------------------------------------------
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우: 맑은 고딕
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')    # 맥: 애플 고딕
else:
    plt.rc('font', family='NanumGothic')    # 리눅스: 나눔 고딕 (설치 필요)

# 마이너스(-) 기호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. 시스템 설정 & 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="소재 공정 최적화 시스템", layout="wide")

if 'opt_result' not in st.session_state:
    st.session_state['opt_result'] = None
if 'opt_model' not in st.session_state:
    st.session_state['opt_model'] = None

# [타이틀] 국문으로 전문성 있게 변경
st.title("🔬 소재 공정 최적화 & 인공지능 분석 시스템")
st.markdown("""
> **System Overview**
> 본 시스템은 **RSM (반응 표면 분석법)** 및 **GPR (가우시안 프로세스)** 알고리즘을 활용하여 
> 실험 데이터를 분석하고 최적의 공정 변수(Parameter)를 도출하는 **연구 지원 프로그램** 입니다.
""")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 사이드바: 데이터 및 모델 설정
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 시스템 설정 (Configuration)")

uploaded_file = st.sidebar.file_uploader("📂 데이터 파일 업로드 (CSV)", type=["csv"])

if uploaded_file is not None:
    # 데이터 로딩
    df = pd.read_csv(uploaded_file)
    all_columns = df.columns.tolist()

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. 변수 설정 (Variables)")

    # [용어] 목표 변수 / 설계 인자
    y_col_name = st.sidebar.selectbox("🎯 목표 변수 (Y, 종속)", all_columns, index=len(all_columns)-1)
    remaining_cols = [c for c in all_columns if c != y_col_name]
    X_col_names = st.sidebar.multiselect("🧪 설계 인자 (X, 독립)", remaining_cols, default=remaining_cols)

    if not X_col_names:
        st.error("⛔ 분석할 독립 변수(X)를 1개 이상 선택하십시오.")
        st.stop()

    # (주의) std, run 등의 불필요한 컬럼이 X인자에 포함되지 않도록 주의하라는 안내
    st.sidebar.caption("※ 실험번호(Run)나 분산(Std) 같은 단순 정보는 X인자에서 제외해주세요.")

    X = df[X_col_names].values
    y = df[y_col_name].values

    st.sidebar.success(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
    st.sidebar.markdown("---")
    
    # 모델 선택
    st.sidebar.subheader("2. 알고리즘 선택")
    model_option = st.sidebar.selectbox("분석 모델", ["RSM (다항 회귀)", "GPR (가우시안 프로세스)"])

    if st.session_state['opt_model'] != model_option:
        st.session_state['opt_result'] = None
        st.session_state['opt_model'] = model_option

    # 하이퍼파라미터 설정
    st.sidebar.subheader("3. 민감도 설정 (Hyperparameter)")
    noise_val = 0.1
    
    if model_option == "GPR (가우시안 프로세스)":
        noise_val = st.sidebar.slider("오차 허용 범위 (Alpha)", 0.00, 0.50, 0.10, 0.01, help="값이 클수록 실험 오차를 관대하게 허용하며(부드러운 곡선), 작을수록 데이터를 엄격하게 따릅니다.")
    else: 
        noise_val = st.sidebar.slider("규제 강도 (Alpha)", 0.00, 2.00, 0.00, 0.10, help="모델의 과적합을 막기 위한 L2 규제 강도입니다.")

    # -------------------------------------------------------------------------
    # 3. 모델 학습 및 성능 평가
    # -------------------------------------------------------------------------
    st.subheader(f"📊 모델 성능 평가 리포트 ({model_option.split(' ')[0]})")
    
    model = None
    poly = None
    scaler_X = None
    scaler_y = None
    r2_score = 0
    q2_score = 0
    rmse_score = 0
    mae_score = 0
    
    X_train = None
    y_train = None
    y_pred_train = None

    # 모델링 로직
    if model_option == "RSM (다항 회귀)":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        if noise_val == 0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=noise_val)
            
        model.fit(X_poly, y)
        r2_score = model.score(X_poly, y)
        
        X_train = X_poly
        y_train = y
        y_pred_train = model.predict(X_poly)
        
    elif model_option == "GPR (가우시안 프로세스)":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        dims = X.shape[1] 
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * dims, (1e-2, 1e2))
        safe_alpha = noise_val if noise_val > 0 else 1e-10
        
        model = GaussianProcessRegressor(kernel=kernel, alpha=safe_alpha, n_restarts_optimizer=10, random_state=42)
        model.fit(X_scaled, y_scaled)
        r2_score = model.score(X_scaled, y_scaled)
        
        X_train = X_scaled
        y_train = y_scaled.flatten()
        
        pred_scaled = model.predict(X_scaled)
        y_pred_train = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # 기본 지표 계산
    rmse_score = np.sqrt(mean_squared_error(y, y_pred_train))
    mae_score = mean_absolute_error(y, y_pred_train)

    n_splits = 5 if len(X) >= 5 else len(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)
        q2_score = cv_scores.mean()
    except:
        q2_score = 0.0

    # [지표 출력] 국문 라벨 적용
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("학습 정확도 ($R^2$)", f"{r2_score:.4f}")
    c2.metric("예측 정확도 ($Q^2_{CV}$)", f"{q2_score:.4f}")
    c3.metric("오차 (RMSE)", f"{rmse_score:.2f}")
    c4.metric("오차 (MAE)", f"{mae_score:.2f}")

    # -------------------------------------------------------------------------
    # [NEW] 고급 분석 지표 (TIC) - 국문 적용
    # -------------------------------------------------------------------------
    with st.expander("🔎 상세 오차 분석 (Theil's Inequality Coefficient) 보기"):
        st.markdown("###  오차 원인 정밀 분석 (TIC Decomposition)")
        
        actual = np.array(y)
        predicted = np.array(y_pred_train)
        
        # TIC 계산
        num = np.sqrt(np.mean((actual - predicted) ** 2))
        den = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
        tic_score = num / den
        
        mse_val = mean_squared_error(actual, predicted)
        
        # Um (Bias)
        um_num = (np.mean(actual) - np.mean(predicted)) ** 2
        um = um_num / mse_val
        
        # Us (Variance)
        std_act = np.std(actual)
        std_pred = np.std(predicted)
        us_num = (std_act - std_pred) ** 2
        us = us_num / mse_val
        
        # Uc (Covariance)
        uc = 1 - (um + us)
        
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("TIC (총 불일치도)", f"{tic_score:.4f}", help="0에 가까울수록 완벽한 모델 (0.1 미만 권장)")
        t2.metric("Um (편향 비율)", f"{um:.4f}", help="오차가 '평균' 차이에서 온 비율 (0에 가까워야 함)")
        t3.metric("Us (변동 비율)", f"{us:.4f}", help="오차가 '변동폭' 차이에서 온 비율 (0에 가까워야 함)")
        t4.metric("Uc (랜덤 비율)", f"{uc:.4f}", help="오차가 '랜덤 노이즈'인 비율 (1에 가까울수록 좋음)")

        st.caption("---")
        if tic_score < 0.1:
            st.success(f"✅ **매우 우수함:** TIC({tic_score:.4f})가 0.1 미만으로, 예측값이 실제값과 거의 일치합니다.")
        elif tic_score < 0.3:
            st.info(f"ℹ️ **양호함:** TIC({tic_score:.4f})가 허용 범위 내에 있습니다.")
        else:
            st.warning(f"⚠️ **주의:** 예측 오차가 다소 큽니다.")
            
        if um > 0.2:
            st.error("🚨 **편향(Bias) 경고:** 모델이 값을 전체적으로 너무 높게(혹은 낮게) 예측하고 있습니다.")
        if us > 0.2:
            st.warning("⚠️ **변동성(Variance) 경고:** 모델이 데이터의 출렁임을 제대로 따라가지 못하고 있습니다.")
        if uc > 0.8:
            st.success("🌟 **이상적인 오차 분포:** 발생한 오차의 대부분이 통제 불가능한 랜덤 노이즈입니다. 모델 구조는 훌륭합니다.")

    # -------------------------------------------------------------------------
    # 진단 메시지 (Diagnostic Logic) - 국문 적용
    # -------------------------------------------------------------------------
    st.markdown("---")
    gap = r2_score - q2_score

    if r2_score > 0.85 and q2_score < 0.3:
        st.error(f"⚠️ **과적합 의심 (Overfitting):** 학습은 잘 됐으나 예측력이 떨어집니다. 사이드바의 '오차 허용 범위'를 높여주세요.")
    
    elif q2_score >= 0.5:
        if gap < 0.2:
             st.success("✅ **고신뢰도 모델 확보 (High Reliability):** 학습 및 예측 성능이 모두 우수합니다.")
        elif gap < 0.4:
             st.success("🆗 **유효 모델 (Valid Model):** 예측 성능($Q^2$)이 기준치(0.5)를 상회하여 실전 적용 가능합니다.")
        else:
             st.warning(f"⚠️ **격차 주의:** 예측력은 좋으나($Q^2$={q2_score:.2f}), 학습 데이터와의 격차가 큽니다. 추가 검증이 권장됩니다.")

    elif q2_score >= 0.3:
        st.warning(f"⚠️ **경향성 파악 수준:** $Q^2$ ({q2_score:.2f})가 다소 낮습니다. 정밀한 예측보다는 경향성 확인용으로 사용하세요.")

    else:
        st.info("ℹ️ **데이터 부족:** 아직 모델이 상관관계를 명확히 찾지 못했습니다. 샘플 수를 늘려주세요.")

    # -------------------------------------------------------------------------
    # 4. 분석 인사이트 (그래프)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 변수 영향력 분석")
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        if model_option == "RSM (다항 회귀)":
            temp_scaler = StandardScaler()
            X_sc = temp_scaler.fit_transform(X)
            simple_model = LinearRegression()
            simple_model.fit(X_sc, y)
            importance = np.abs(simple_model.coef_)
            
            fig_imp = go.Figure(go.Bar(
                x=importance, y=X_col_names, orientation='h', marker=dict(color='teal')
            ))
            fig_imp.update_layout(title="표준화 회귀 계수 (영향력 크기)", xaxis_title="계수 절댓값", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

        elif model_option == "GPR (가우시안 프로세스)":
            if hasattr(model.kernel_, 'k2'):
                length_scales = model.kernel_.k2.length_scale
                if np.isscalar(length_scales):
                    st.warning("⚠️ 등방성 커널이 감지되어 개별 변수 중요도를 산출할 수 없습니다.")
                else:
                    sensitivity = 1 / length_scales
                    fig_imp = go.Figure(go.Bar(
                        x=sensitivity, y=X_col_names, orientation='h', marker=dict(color='purple')
                    ))
                    fig_imp.update_layout(title="변수 민감도 (Sensitivity)", xaxis_title="민감도 (1/LengthScale)", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_imp2:
        st.write("**변수 간 상관관계 (Pearson Correlation)**")
        corr_matrix = df[X_col_names + [y_col_name]].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}", showscale=True
        ))
        fig_corr.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

   # -------------------------------------------------------------------------
    # 5. 가상 실험 및 최적화
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎛️ 가상 실험실 & 공정 최적화 (Virtual Lab)")

    col_sim, col_graph = st.columns([1, 2])

    with col_sim:
        st.markdown("**🧪 조건 시뮬레이션**")
        
        for col in X_col_names:
            if col not in st.session_state:
                st.session_state[col] = float(df[col].mean())

        # 동기화 함수
        def update_slider_from_input(key):
            st.session_state[key] = st.session_state[f"{key}_input"]

        def update_input_from_slider(key):
            st.session_state[f"{key}_input"] = st.session_state[key]

        input_values = []
        bounds = []

        for i, col_name in enumerate(X_col_names):
            data_min = float(df[col_name].min())
            data_max = float(df[col_name].max())
            
            extended_min = data_min * 0.5 
            extended_max = data_max * 1.5
            if data_min >= 0: extended_min = max(0.0, extended_min)

            if f"{col_name}_input" not in st.session_state:
                st.session_state[f"{col_name}_input"] = st.session_state[col_name]

            c1, c2 = st.columns([3, 1])
            with c1:
                val = st.slider(
                    f"{col_name}", 
                    min_value=extended_min, 
                    max_value=extended_max, 
                    key=col_name,
                    step=0.01,
                    on_change=update_input_from_slider,
                    args=(col_name,)
                )
            with c2:
                st.number_input(
                    "입력",
                    min_value=extended_min,
                    max_value=extended_max,
                    key=f"{col_name}_input",
                    step=0.01,
                    label_visibility="collapsed",
                    on_change=update_slider_from_input,
                    args=(col_name,)
                )
            
            input_values.append(val)
            bounds.append((data_min, data_max))
            
        st.markdown("---")

        # 실시간 예측
        current_pred_val = 0
        if model_option == "RSM (다항 회귀)":
            current_pred_val = model.predict(poly.transform([input_values]))[0]
        elif model_option == "GPR (가우시안 프로세스)":
            x_scaled_in = scaler_X.transform([input_values])
            pred_scaled = model.predict(x_scaled_in)
            current_pred_val = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        st.metric(
            label=f"AI 예측 결과 ({y_col_name})", 
            value=f"{current_pred_val:.2f}", 
            delta="실시간 예측값"
        )
        
        st.write("")
        
        # 최적화 실행
        if st.button("🚀 최적 조건 자동 탐색 (Run Optimization)"):
            def objective_func(x_input):
                if model_option == "RSM (다항 회귀)":
                    return -model.predict(poly.transform([x_input]))[0]
                else: 
                    x_scaled_in = scaler_X.transform([x_input])
                    pred_scaled = model.predict(x_scaled_in)
                    return -scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]

            res = minimize(objective_func, input_values, bounds=bounds, method='L-BFGS-B')
            st.session_state['opt_result'] = res
            st.session_state['opt_model'] = model_option
            
            st.success(f"탐색 완료! 예상 최대값: {-res.fun:.2f}")
        
        if st.session_state['opt_result'] is not None and st.session_state['opt_model'] == model_option:
            res = st.session_state['opt_result']
            
            st.write("---")
            st.write("**📝 도출된 최적 조건**")
            for i, name in enumerate(X_col_names):
                st.write(f"- **{name}:** {res.x[i]:.2f}") 
            
            def set_sliders_to_optimal():
                for i, name in enumerate(X_col_names):
                    opt_val = float(res.x[i])
                    st.session_state[name] = opt_val
                    st.session_state[f"{name}_input"] = opt_val

            st.button("🔄 이 조건을 시뮬레이터에 적용", on_click=set_sliders_to_optimal)
            
            # CSV 다운로드
            result_dict = {"변수명": X_col_names, "최적값": res.x}
            res_df = pd.DataFrame(result_dict)
            new_row = pd.DataFrame([{"변수명": f"예측 {y_col_name}", "최적값": -res.fun}])
            res_df = pd.concat([res_df, new_row], ignore_index=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 최적화 결과 저장 (CSV)",
                data=csv,
                file_name='Optimization_Result.csv',
                mime='text/csv',
            )

    # -------------------------------------------------------------------------
    # 6. 그래프 시각화 (Visualization)
    # -------------------------------------------------------------------------
    with col_graph:
        st.write(f"**📉 반응 표면 그래프 (3D/2D)**")
         
        tab1, tab2 = st.tabs(["2D 단면 분석", "3D 표면 분석"])
        
        with tab1:
            graph_x_col = st.selectbox("X축 변수 선택", X_col_names, key="2d_x_select")
            x_idx = X_col_names.index(graph_x_col)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            x_grid = np.linspace(df[graph_x_col].min(), df[graph_x_col].max(), 100)
            input_grid = np.array([input_values] * 100)
            input_grid[:, x_idx] = x_grid
            
            y_pred = []
            if model_option == "RSM (다항 회귀)":
                y_pred = model.predict(poly.transform(input_grid))
                ax.plot(x_grid, y_pred, 'b-', label='AI 예측 모델', linewidth=2)
            else: 
                p_sc, s_sc = model.predict(scaler_X.transform(input_grid), return_std=True)
                y_pred = scaler_y.inverse_transform(p_sc.reshape(-1, 1)).flatten()
                y_std = s_sc * scaler_y.scale_[0]
                ax.plot(x_grid, y_pred, 'g-', label='AI 예측 평균', linewidth=2)
                ax.fill_between(x_grid, y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='green', alpha=0.1, label='95% 신뢰구간')

            ax.scatter(df[graph_x_col], df[y_col_name], color='red', s=40, alpha=0.5, label='실제 실험값')
            
            curr_y = 0
            if model_option == "RSM (다항 회귀)": curr_y = model.predict(poly.transform([input_values]))[0]
            else: curr_y = scaler_y.inverse_transform(model.predict(scaler_X.transform([input_values])).reshape(-1,1))[0][0]
            ax.scatter(input_values[x_idx], curr_y, color='blue', s=100, edgecolors='white', label='현재 설정값', zorder=10)

            if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                opt = st.session_state['opt_result']
                opt_x, opt_y = opt.x[x_idx], -opt.fun
                ax.scatter(opt_x, opt_y, color='gold', marker='*', s=300, edgecolors='k', label='최적점 (AI)', zorder=10)
                ax.vlines(x=opt_x, ymin=ax.get_ylim()[0], ymax=opt_y, colors='gold', linestyles='--')

            ax.set_xlabel(graph_x_col)
            ax.set_ylabel(y_col_name)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        with tab2:
            if len(X_col_names) < 2:
                st.warning("⚠️ 3D 그래프를 그리려면 최소 2개의 변수가 필요합니다.")
            else:
                c1, c2 = st.columns(2)
                x_axis = c1.selectbox("X축", X_col_names, index=0, key="3d_x")
                y_axis = c2.selectbox("Y축", X_col_names, index=1, key="3d_y")

                if x_axis == y_axis:
                    st.error("X축과 Y축은 서로 다른 변수여야 합니다.")
                else:
                    fixed_vars = [col for col in X_col_names if col not in [x_axis, y_axis]]
                    if fixed_vars:
                        fixed_str = ", ".join([f"{col}={input_values[X_col_names.index(col)]:.2f}" for col in fixed_vars])
                        st.caption(f"ℹ️ **고정된 변수 (현재 슬라이더 값):** {fixed_str}")

                    x_min, x_max = df[x_axis].min(), df[x_axis].max()
                    y_min, y_max = df[y_axis].min(), df[y_axis].max()
                    padding_x = (x_max - x_min) * 0.1
                    padding_y = (y_max - y_min) * 0.1
                    
                    resolution = 60 
                    x_range = np.linspace(x_min - padding_x, x_max + padding_x, resolution)
                    y_range = np.linspace(y_min - padding_y, y_max + padding_y, resolution)
                    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
                    
                    idx_x, idx_y = X_col_names.index(x_axis), X_col_names.index(y_axis)
                    Z_mesh = np.zeros_like(X_mesh)
                    
                    for i in range(X_mesh.shape[0]):
                        for j in range(X_mesh.shape[1]):
                            temp_in = input_values.copy()
                            temp_in[idx_x] = X_mesh[i, j]
                            temp_in[idx_y] = Y_mesh[i, j]
                            if model_option == "RSM (다항 회귀)":
                                Z_mesh[i, j] = model.predict(poly.transform([temp_in]))[0]
                            else:
                                p = model.predict(scaler_X.transform([temp_in]))
                                Z_mesh[i, j] = scaler_y.inverse_transform(p.reshape(-1,1))[0][0]

                    fig_3d = go.Figure(data=[go.Surface(
                        z=Z_mesh, x=X_mesh, y=Y_mesh, 
                        colorscale='Viridis', opacity=0.8, name='AI 예측 표면',
                        contours = {"z": {"show": True, "start": 0, "end": 200, "size": 2, "color":"white"}},
                        colorbar=dict(title=dict(text=y_col_name, side="right"))
                    )])
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=df[x_axis], y=df[y_axis], z=df[y_col_name],
                        mode='markers', marker=dict(size=5, color='red', line=dict(color='white', width=1)), name='실제 실험값'
                    ))

                    if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                        opt = st.session_state['opt_result']
                        opt_x, opt_y, opt_z = opt.x[idx_x], opt.x[idx_y], -opt.fun
                        
                        fig_3d.add_trace(go.Scatter3d(
                            x=[opt_x], y=[opt_y], z=[opt_z],
                            mode='markers+text',
                            marker=dict(
                                size=8, color='#FF00FF', symbol='square', 
                                line=dict(color='white', width=2)
                            ),
                            text=[f"★ 최적값\n{opt_z:.2f}"], 
                            textposition="top center",
                            textfont=dict(color='black', size=12, family="Arial Black"),
                            name='AI 도출 최적점'
                        ))

                    fig_3d.update_layout(
                        title=f"3D 반응 표면 그래프 ({x_axis} vs {y_axis})",
                        scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=y_col_name, aspectmode='cube'),
                        width=800, height=600,
                        margin=dict(l=0, r=0, b=50, t=40),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_3d)

else:
    st.info("👈 왼쪽 사이드바에서 실험 데이터(CSV)를 업로드해주세요.")
