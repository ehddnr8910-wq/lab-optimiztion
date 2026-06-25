import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import streamlit as st
import pandas as pd
import numpy as np
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
# [한글 폰트 설정]
# -----------------------------------------------------------------------------
def setup_korean_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. 시스템 설정 & 초기화
# -----------------------------------------------------------------------------
st.set_page_config(page_title="소재 공정 최적화 시스템", layout="wide")

if 'opt_result' not in st.session_state:
    st.session_state['opt_result'] = None
if 'opt_model' not in st.session_state:
    st.session_state['opt_model'] = None

# [타이틀]
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

# 데이터가 로드되면 즉시 분석 시작
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
        noise_val = st.sidebar.slider("오차 허용 범위 (Alpha)", 0.00, 0.50, 0.10, 0.01, help="값이 클수록 실험 오차를 관대하게 허용하며, 작을수록 데이터를 엄격하게 따릅니다.")
    else: 
        noise_val = st.sidebar.slider("규제 강도 (Alpha)", 0.00, 2.00, 0.00, 0.10, help="모델의 과적합을 막기 위한 L2 규제 강도입니다.")

    # -------------------------------------------------------------------------
    # 3. 모델 학습 및 성능 평가 (버튼 없이 즉시 실행)
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
        
        # 세션 저장 (RSM 계수 표 출력용)
        st.session_state['real_rsm_reg'] = model      
        st.session_state['real_rsm_poly'] = poly      
        st.session_state['real_rsm_names'] = X_col_names 
        
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

    # 모델 및 스케일러 세션 저장 (최적화용)
    st.session_state['trained_model'] = model
    st.session_state['trained_poly'] = poly
    st.session_state['scaler_X'] = scaler_X
    st.session_state['scaler_y'] = scaler_y
    
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

    # [지표 출력]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("학습 정확도 ($R^2$)", f"{r2_score:.4f}")
    c2.metric("예측 정확도 ($Q^2_{CV}$)", f"{q2_score:.4f}")
    c3.metric("오차 (RMSE)", f"{rmse_score:.2f}")
    c4.metric("오차 (MAE)", f"{mae_score:.2f}")

    # 상세 오차 분석 (TIC)
    with st.expander("🔎 상세 오차 분석 (Theil's Inequality Coefficient) 보기"):
        st.markdown("###  오차 원인 정밀 분석 (TIC Decomposition)")
        actual = np.array(y)
        predicted = np.array(y_pred_train)
        
        num = np.sqrt(np.mean((actual - predicted) ** 2))
        den = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
        tic_score = num / den if den != 0 else 0
        
        mse_val = mean_squared_error(actual, predicted)
        um = ((np.mean(actual) - np.mean(predicted)) ** 2) / mse_val if mse_val !=0 else 0
        std_act = np.std(actual)
        std_pred = np.std(predicted)
        us = ((std_act - std_pred) ** 2) / mse_val if mse_val !=0 else 0
        uc = 1 - (um + us)
        
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("TIC (총 불일치도)", f"{tic_score:.4f}")
        t2.metric("Um (편향 비율)", f"{um:.4f}")
        t3.metric("Us (변동 비율)", f"{us:.4f}")
        t4.metric("Uc (랜덤 비율)", f"{uc:.4f}")

    # -------------------------------------------------------------------------
    # 4. 분석 인사이트 (그래프)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 변수 영향력 분석")
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        if model_option == "RSM (다항 회귀)":
            # RSM 표준화 계수 (단순화)
            temp_scaler = StandardScaler()
            X_sc = temp_scaler.fit_transform(X)
            simple_model = LinearRegression()
            simple_model.fit(X_sc, y)
            importance = np.abs(simple_model.coef_)
            
            fig_imp = go.Figure(go.Bar(
                x=importance, y=X_col_names, orientation='h', marker=dict(color='teal')
            ))
            fig_imp.update_layout(title="표준화 회귀 계수 (영향력 크기)", xaxis_title="계수 절댓값")
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
                    fig_imp.update_layout(title="변수 민감도 (Sensitivity)", xaxis_title="민감도 (1/LengthScale)")
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
        fig_corr.update_layout(height=300)
        st.plotly_chart(fig_corr, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. 가상 실험 및 최적화
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎛️ 가상 실험실 & 공정 최적화")

    col_sim, col_graph = st.columns([1, 2])

    # 슬라이더 초기화
    if 'slider_initialized' not in st.session_state:
        for col in X_col_names:
            st.session_state[col] = float(df[col].mean())
            st.session_state[f"{col}_input"] = float(df[col].mean())
        st.session_state['slider_initialized'] = True

    def update_slider_from_input(key):
        st.session_state[key] = st.session_state[f"{key}_input"]
    def update_input_from_slider(key):
        st.session_state[f"{key}_input"] = st.session_state[key]

    input_values = []
    bounds = []

    with col_sim:
        st.markdown("**🧪 조건 시뮬레이션**")
        
        for col_name in X_col_names:
            data_min = float(df[col_name].min())
            data_max = float(df[col_name].max())
            extended_min = max(0.0, data_min * 0.5)
            extended_max = data_max * 1.5

            # 세션 상태가 없으면 초기화 (오류 방지)
            if f"{col_name}_input" not in st.session_state:
                st.session_state[f"{col_name}_input"] = float(df[col_name].mean())
            if col_name not in st.session_state:
                st.session_state[col_name] = float(df[col_name].mean())

            c1_sl, c2_sl = st.columns([3, 1])
            with c1_sl:
                val = st.slider(f"{col_name}", extended_min, extended_max, key=col_name, step=0.01, on_change=update_input_from_slider, args=(col_name,))
            with c2_sl:
                st.number_input("입력", extended_min, extended_max, key=f"{col_name}_input", step=0.01, label_visibility="collapsed", on_change=update_slider_from_input, args=(col_name,))
            
            input_values.append(val)
            bounds.append((data_min, data_max))

        # 실시간 예측
        current_pred_val = 0
        if model_option == "RSM (다항 회귀)":
            current_pred_val = model.predict(poly.transform([input_values]))[0]
        elif model_option == "GPR (가우시안 프로세스)":
            x_scaled_in = scaler_X.transform([input_values])
            pred_scaled = model.predict(x_scaled_in)
            current_pred_val = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        st.metric(f"AI 예측 결과 ({y_col_name})", f"{current_pred_val:.2f}")

        # 최적화
        if st.button("🚀 최적 조건 자동 탐색"):
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

        # 결과 적용 버튼
        if st.session_state['opt_result'] is not None and st.session_state['opt_model'] == model_option:
            res = st.session_state['opt_result']
            st.write("**📝 도출된 최적 조건**")
            for i, name in enumerate(X_col_names):
                st.write(f"- **{name}:** {res.x[i]:.2f}") 
            
            # 콜백 함수 정의 (위젯 생성 전/후 충돌 방지)
            def apply_opt_to_sim(opt_res_x, col_names):
                for i, name in enumerate(col_names):
                    st.session_state[name] = float(opt_res_x[i])
                    st.session_state[f"{name}_input"] = float(opt_res_x[i])

            # on_click을 사용하여 버튼 클릭 시 즉시 상태 업데이트
            st.button("🔄 이 조건을 시뮬레이터에 적용", on_click=apply_opt_to_sim, args=(res.x, X_col_names))

    # -------------------------------------------------------------------------
    # 6. 그래프 시각화 (논문용 포맷 + Plotly 최신 문법 + Jet Colormap 적용)
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
            
            if model_option == "RSM (다항 회귀)":
                y_pred = model.predict(poly.transform(input_grid))
                ax.plot(x_grid, y_pred, 'b-', label='AI 예측', linewidth=2)
            else: 
                p_sc, s_sc = model.predict(scaler_X.transform(input_grid), return_std=True)
                y_pred = scaler_y.inverse_transform(p_sc.reshape(-1, 1)).flatten()
                y_std = s_sc * scaler_y.scale_[0]
                ax.plot(x_grid, y_pred, 'g-', label='AI 예측 평균', linewidth=2)
                ax.fill_between(x_grid, y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='green', alpha=0.1)

            ax.scatter(df[graph_x_col], df[y_col_name], color='red', s=40, alpha=0.5, label='실제값')
            curr_y = current_pred_val
            ax.scatter(input_values[x_idx], curr_y, color='blue', s=100, edgecolors='white', label='현재값', zorder=10)
            
            if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                opt = st.session_state['opt_result']
                ax.scatter(opt.x[x_idx], -opt.fun, color='gold', marker='*', s=300, edgecolors='k', label='최적점')

            ax.set_xlabel(graph_x_col)
            ax.set_ylabel(y_col_name)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            # ── SigmaPlot 데이터 추출 (2D) ──────────────────────────────
            with st.expander("📥 SigmaPlot용 데이터 추출 (2D 단면 그래프)"):
                st.caption("아래 CSV를 SigmaPlot에서 File → Import 하거나, 직접 복사하여 붙여넣기 하세요.")

                # (1) AI 예측 곡선
                df_2d_curve = pd.DataFrame({
                    f"X_{graph_x_col}": x_grid,
                    f"Y_pred_{y_col_name}": y_pred
                })
                if model_option == "GPR (가우시안 프로세스)":
                    df_2d_curve[f"Y_pred_upper_95CI"] = y_pred + 1.96 * y_std
                    df_2d_curve[f"Y_pred_lower_95CI"] = y_pred - 1.96 * y_std

                # (2) 실제 실험 데이터 (산점)
                df_2d_scatter = pd.DataFrame({
                    f"X_{graph_x_col}_exp": df[graph_x_col].values,
                    f"Y_{y_col_name}_exp": df[y_col_name].values
                })

                # (3) 현재 입력점
                df_2d_current = pd.DataFrame({
                    f"X_current": [input_values[x_idx]],
                    f"Y_current": [curr_y]
                })

                # 세 데이터셋을 열 방향으로 합치기 (길이 다를 수 있으므로 concat axis=1)
                df_2d_export = pd.concat([df_2d_curve, df_2d_scatter, df_2d_current], axis=1)

                # 최적점이 있으면 추가
                if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                    opt = st.session_state['opt_result']
                    df_2d_export[f"X_optimum"] = pd.Series([opt.x[x_idx]])
                    df_2d_export[f"Y_optimum"] = pd.Series([-opt.fun])

                csv_2d = df_2d_export.to_csv(index=False).encode('utf-8-sig')
                st.dataframe(df_2d_export.head(10), use_container_width=True)
                st.download_button(
                    label="⬇️ CSV 다운로드 (2D 단면 그래프)",
                    data=csv_2d,
                    file_name=f"sigmaplot_2D_{graph_x_col}_vs_{y_col_name}.csv",
                    mime="text/csv",
                    key="dl_2d"
                )

        with tab2:
            if len(X_col_names) < 2:
                st.warning("⚠️ 3D plots require at least 2 variables.")
            else:
                st.markdown("### 🖼️ Publication-Ready 3D Plot")
                
                c1, c2 = st.columns(2)
                x_axis = c1.selectbox("X-axis Variable", X_col_names, index=0, key="3d_x")
                y_axis = c2.selectbox("Y-axis Variable", X_col_names, index=1, key="3d_y")

                if x_axis == y_axis:
                    st.error("Please select different variables for X and Y axes.")
                else:
                    # 1. 데이터 범위 설정
                    x_min, x_max = df[x_axis].min(), df[x_axis].max()
                    y_min, y_max = df[y_axis].min(), df[y_axis].max()
                    
                    pad_x, pad_y = (x_max - x_min)*0.1, (y_max - y_min)*0.1
                    x_range = np.linspace(x_min - pad_x, x_max + pad_x, 60) 
                    y_range = np.linspace(y_min - pad_y, y_max + pad_y, 60)
                    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
                    
                    idx_x, idx_y = X_col_names.index(x_axis), X_col_names.index(y_axis)
                    Z_mesh = np.zeros_like(X_mesh)
                    
                    # 2. 예측값 계산
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

                    # 3. 그래프 그리기
                    fig_3d = go.Figure()

                    # (1) 반응 표면 (Surface) - Jet Colormap
                    fig_3d.add_trace(go.Surface(
                        z=Z_mesh, x=X_mesh, y=Y_mesh,
                        colorscale='Jet', 
                        opacity=0.8,
                        colorbar=dict(
                            title=dict(text='Adsorption (mg/g)', font=dict(size=14)),
                            tickfont=dict(size=12),
                            len=0.8
                        )
                    ))

                    # (2) 실제 실험 데이터 (Experimental Data)
                    fig_3d.add_trace(go.Scatter3d(
                        x=df[x_axis], y=df[y_axis], z=df[y_col_name],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color='black', 
                            symbol='circle', 
                            line=dict(color='white', width=1)
                        ),
                        name='Experimental Data'
                    ))
                    
                    # (3) 최적점 표시 (Optimum Point)
                    if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                        opt = st.session_state['opt_result']
                        opt_val = -opt.fun
                        
                        fig_3d.add_trace(go.Scatter3d(
                            x=[opt.x[idx_x]], y=[opt.x[idx_y]], z=[opt_val],
                            mode='markers+text',
                            text=[f"Max: {opt_val:.2f}"], 
                            textposition="top center", 
                            textfont=dict(size=14, color="black", family="Arial Black"),
                            marker=dict(
                                size=12, 
                                color='red', 
                                symbol='diamond',
                                line=dict(color='white', width=2)
                            ),
                            name='AI Predicted Optimum'
                        ))

                    # 4. 레이아웃 설정
                    fig_3d.update_layout(
                        title={
                            'text': f"Response Surface Plot: {x_axis} vs {y_axis}",
                            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                            'font': dict(size=20, family="Arial")
                        },
                        scene=dict(
                            xaxis=dict(
                                title=dict(text=f"{x_axis} (%)", font=dict(size=14)),
                                tickfont=dict(size=12), 
                                backgroundcolor="white"
                            ),
                            yaxis=dict(
                                title=dict(text=f"{y_axis} (mL)" if "ECH" in y_axis else f"{y_axis} (%)", font=dict(size=14)),
                                tickfont=dict(size=12), 
                                backgroundcolor="white"
                            ),
                            zaxis=dict(
                                title=dict(text="q (mg/g)", font=dict(size=14)),
                                tickfont=dict(size=12), 
                                backgroundcolor="white"
                            ),
                            aspectratio=dict(x=1, y=1, z=0.8)
                        ),
                        width=900, height=700,
                        margin=dict(l=0, r=0, b=0, t=50),
                        legend=dict(x=0.7, y=0.9, font=dict(size=14)),
                        template='plotly_white'
                    )

                    st.plotly_chart(fig_3d, use_container_width=True)

                    # ── SigmaPlot 데이터 추출 (3D) ───────────────────────────
                    with st.expander("📥 SigmaPlot용 데이터 추출 (3D 반응 표면 그래프)"):
                        st.caption("SigmaPlot 3D Mesh/Surface 또는 Scatter 3D 그래프에 사용할 수 있는 CSV입니다.")

                        # ── (1) SigmaPlot Matrix 형식 (행=X, 열=Y, 셀=Z) ──
                        # SigmaPlot 3D Surface는 피벗 테이블(Grid) 구조를 요구함
                        # 첫 번째 행: Y값 헤더 (빈 셀 + Y값들)
                        # 이후 행: X값 | Z값들
                        y_labels = np.round(y_range, 4)   # 열 헤더 = Y축 값
                        x_labels = np.round(x_range, 4)   # 행 인덱스 = X축 값

                        # Z_mesh shape: (len(y_range), len(x_range)) — meshgrid 구조
                        # SigmaPlot: 행이 X, 열이 Y → Z_mesh.T 사용
                        df_pivot = pd.DataFrame(
                            Z_mesh.T,
                            index=x_labels,
                            columns=y_labels
                        )
                        df_pivot.index.name = f"{x_axis} \\ {y_axis}"

                        # ── (2) 실험 데이터 산점 (XYZ 3열) ──
                        df_3d_scatter = pd.DataFrame({
                            f"X_{x_axis}_exp": df[x_axis].values,
                            f"Y_{y_axis}_exp": df[y_axis].values,
                            f"Z_{y_col_name}_exp": df[y_col_name].values
                        })

                        # 두 탭으로 분리 출력
                        tab_surf, tab_scat = st.tabs(["Surface Matrix (SigmaPlot용)", "Experimental Scatter 데이터"])
                        with tab_surf:
                            st.caption(f"📌 행(Row) = {x_axis} 값 / 열(Column) = {y_axis} 값 / 셀 = 예측 {y_col_name}")
                            st.caption("SigmaPlot: Graph → 3D → Surface Plot → XYZ Matrix 선택 후 이 데이터를 붙여넣기")
                            st.dataframe(df_pivot.iloc[:10, :10], use_container_width=True)
                            csv_surf = df_pivot.to_csv().encode('utf-8-sig')
                            st.download_button(
                                label="⬇️ CSV 다운로드 (3D Surface Matrix)",
                                data=csv_surf,
                                file_name=f"sigmaplot_3D_surface_{x_axis}_vs_{y_axis}.csv",
                                mime="text/csv",
                                key="dl_3d_surf"
                            )
                        with tab_scat:
                            st.caption("📌 SigmaPlot: Graph → 3D → Scatter Plot → XYZ 3열 선택")
                            st.dataframe(df_3d_scatter, use_container_width=True)
                            csv_scat = df_3d_scatter.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="⬇️ CSV 다운로드 (3D Experimental Scatter)",
                                data=csv_scat,
                                file_name=f"sigmaplot_3D_scatter_{x_axis}_vs_{y_axis}.csv",
                                mime="text/csv",
                                key="dl_3d_scat"
                            )

                        # (3) 최적점 있으면 함께 출력
                        if st.session_state['opt_result'] and st.session_state['opt_model'] == model_option:
                            opt = st.session_state['opt_result']
                            df_3d_opt = pd.DataFrame({
                                f"X_{x_axis}_opt": [opt.x[idx_x]],
                                f"Y_{y_axis}_opt": [opt.x[idx_y]],
                                f"Z_{y_col_name}_opt": [-opt.fun]
                            })
                            st.markdown("**최적점 좌표**")
                            st.dataframe(df_3d_opt, use_container_width=True)
                            csv_opt = df_3d_opt.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="⬇️ CSV 다운로드 (최적점)",
                                data=csv_opt,
                                file_name=f"sigmaplot_3D_optimum_{x_axis}_vs_{y_axis}.csv",
                                mime="text/csv",
                                key="dl_3d_opt"
                            )

    # -------------------------------------------------------------------------
    # [추가된 기능] 7. 논문 작성용 RSM 수식 계수 확인
    # -------------------------------------------------------------------------
    if model_option == "RSM (다항 회귀)" and 'real_rsm_reg' in st.session_state:
        st.markdown("---")
        st.subheader("📊 논문 작성용 RSM 수식 계수 확인 (Coefficients)")
        st.info("💡 Design Expert의 'Actual Equation'과 비교하기 위한 실제 계수표입니다. (Alpha=0 설정 필수)")

        try:
            reg_model = st.session_state['real_rsm_reg']
            poly_features = st.session_state['real_rsm_poly']
            input_names = st.session_state['real_rsm_names']

            # 1. 변수명 생성
            feature_names = poly_features.get_feature_names_out(input_names)

            # 2. 계수 및 절편 추출
            coefs = reg_model.coef_
            intercept = reg_model.intercept_
            
            if coefs.ndim > 1:
                coefs = coefs.flatten()

            # 3. 데이터프레임 생성
            data_rows = [{"항 (Term)": "Intercept", "계수 (Coefficient)": intercept}]
            for name, val in zip(feature_names, coefs):
                data_rows.append({"항 (Term)": name, "계수 (Coefficient)": val})

            df_final = pd.DataFrame(data_rows)

            # 4. 소수점 포맷팅 및 출력
            st.table(df_final.style.format({"계수 (Coefficient)": "{:.4f}"}))

        except Exception as e:
            st.error(f"계수 추출 중 오류가 발생했습니다: {e}")
