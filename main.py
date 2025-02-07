import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="슈퍼마켓 판매 대시보드", layout="wide")

@st.cache_data
def load_data():
    """슈퍼마켓 판매 데이터를 불러오고 전처리하는 함수"""
    df = pd.read_csv("data/supermarket_sales.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


# 데이터 로드
df = load_data()

# --------------------------------------------
# 탭 설정
# --------------------------------------------
tabs = st.tabs(["Home", "EDA", "KPI", "예측 모델"])

with tabs[0]:
    st.title("슈퍼마켓 판매 데이터 대시보드")
    st.write(
        """
        
    슈퍼마켓 판매 데이터를 활용한 대시보드입니다.
    상단 탭을 클릭하여 탐색적 데이터 분석(EDA), 주요 지표(KPI), 예측 모델 섹션을 확인할 수 있습니다.""")
    st.markdown("---")

    # 대시보드 개요
    st.subheader("데이터 개요")
    st.write(f"총 데이터 개수: {len(df)} 행")
    st.write(f"컬럼: {list(df.columns)}")

with tabs[1]:
    st.title("탐색적 데이터 분석 (EDA)")

    st.markdown("#### 1. 데이터 미리보기")
    st.dataframe(df.head())

    st.markdown("#### 2. 기본 통계량")
    st.write(df.describe())

    st.markdown("#### 3. 상관관계 히트맵")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("#### 4. 히스토그램 (분포 확인)")
    col_selection = st.selectbox("컬럼을 선택하세요", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[col_selection], kde=True, ax=ax2, color="skyblue")
    st.pyplot(fig2)

with tabs[2]:
    st.title("주요 KPI")

    # 기본 KPI 계산
    total_sales = df["Total"].sum()
    avg_sales = df["Total"].mean()
    total_transactions = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("총 매출 (USD)", f"{total_sales:,.2f}")
    col2.metric("평균 거래금액 (USD)", f"{avg_sales:,.2f}")
    col3.metric("총 거래 건수", f"{total_transactions}")

    st.markdown("---")

    # (1) 매출 목표 설정 & 달성도 Gauge
    st.subheader("KPI: 매출 목표 달성도")
    target = st.number_input(
        "목표 매출을 설정하세요 (USD)", min_value=0, value=60000, step=5000
    )

    # Gauge 차트 그리기
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=total_sales,
            delta={
                "reference": target,
                "increasing": {"color": "green"},
                "decreasing": {"color": "red"},
            },
            gauge={
                "axis": {"range": [None, max(target, total_sales) * 1.1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, target * 0.5], "color": "lightgray"},
                    {"range": [target * 0.5, target], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": target,
                },
            },
            title={"text": "총 매출"},
        )
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if total_sales >= target:
        st.success("목표 매출을 달성했습니다!")
    else:
        st.warning(f"아직 목표까지 ${(target - total_sales):,.2f} 남았습니다.")

    st.markdown("---")

    # (2) 결제 방법별 매출
    st.subheader("결제 방법별 매출")
    payment_sales = df.groupby("Payment")["Total"].sum().reset_index()
    fig_pay = px.bar(
        payment_sales,
        x="Payment",
        y="Total",
        color="Payment",
        title="결제 방법별 매출 비교",
    )
    st.plotly_chart(fig_pay, use_container_width=True)

    # (3) 도시별 매출 추이
    st.subheader("도시별 매출 추이")
    city_selection = st.selectbox("도시를 선택하세요", df["City"].unique())
    city_df = df[df["City"] == city_selection]
    daily_sales = city_df.groupby("Date")["Total"].sum().reset_index()
    fig_city = px.line(
        daily_sales,
        x="Date",
        y="Total",
        title=f"{city_selection} 도시의 일자별 매출 추이",
    )
    st.plotly_chart(fig_city, use_container_width=True)

    # (4) 요일별 매출
    st.subheader("요일별 매출")
    if "Weekday" not in df.columns:
        df["Weekday"] = df["Date"].dt.day_name()
    weekday_sales = df.groupby("Weekday")["Total"].sum().reset_index()
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_sales["Weekday"] = pd.Categorical(
        weekday_sales["Weekday"], categories=weekday_order, ordered=True
    )
    weekday_sales.sort_values("Weekday", inplace=True)

    fig_week = px.bar(
        weekday_sales, x="Weekday", y="Total", color="Weekday", title="요일별 매출"
    )
    st.plotly_chart(fig_week, use_container_width=True)

with tabs[3]:
    st.title("예측 모델")

    st.write(
            """
            간단한 선형회귀 모델로,
            Unit price 와 Quantity 을 이용하여 Total 을 예측합니다.
            """
        )

    # -- 모델 학습 --
    X = df[["Unit price", "Quantity"]]
    y = df["Total"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -- 예측 결과 --
    y_pred = model.predict(X_test)
    r2_score = model.score(X_test, y_test)

    st.subheader("1) 모델 학습 결과")
    st.write(f"회귀 계수: {model.coef_}")
    st.write(f"절편: {model.intercept_:.2f}")
    st.metric("R² (테스트 세트)", f"{r2_score:.2f}")

    st.subheader("2) 실제값 vs 예측값 (시각화)")

    # 테스트 세트의 (실제값, 예측값) -> 하나의 DF에서 melt
    comparison_df = pd.DataFrame(
        {
            "index": range(len(y_test)),  # 그래프용 index
            "실제값": y_test.values,  # y_test가 Series이므로 .values 사용
            "예측값": y_pred,
        }
    )

    # melt()하여 "타입"(실제 / 예측), "값" 으로 변환
    df_plot = comparison_df.melt(
        id_vars="index",
        value_vars=["실제값", "예측값"],
        var_name="타입",
        value_name="값",
    )

    # 사용자 입력에 대한 예측값 (추가할 데이터)
    user_input_df = None  # 나중에 예측 후 추가

    # 사용자 입력 폼
    st.subheader("3) 사용자 입력으로 예측해보기")
    with st.form("predict_form"):
        unit_price = st.number_input(
            "단가(Unit price)", min_value=0.0, value=10.0, step=1.0
        )
        quantity = st.number_input("수량(Quantity)", min_value=1, value=5)
        submitted = st.form_submit_button("예측하기")

    if submitted:
        new_data = pd.DataFrame({"Unit price": [unit_price], "Quantity": [quantity]})
        pred_val = model.predict(new_data)[0]
        st.success(f"예측된 총 판매금액(Total): 약 ${pred_val:,.2f}")

        # 사용자 입력값을 그래프에 추가하기 위해 임의 index 부여
        new_index = df_plot["index"].max() + 1
        user_input_df = pd.DataFrame(
            {"index": [new_index], "타입": ["사용자 입력값"], "값": [pred_val]}
        )

    # df_plot 에 user_input_df를 concat
    if user_input_df is not None:
        df_plot = pd.concat([df_plot, user_input_df], ignore_index=True)

    # Plotly Scatter: x= index, y= 값, color= 타입
    fig_result = px.scatter(
        df_plot,
        x="index",
        y="값",
        color="타입",
        title="실제값 vs 예측값 (다른 색상 표시) + 사용자 입력값",
        labels={"index": "테스트 데이터 인덱스", "값": "Total 금액"},
        width=800,
        height=500,
    )

    # 조금 더 보기 좋게 마커 크기 등 조정
    fig_result.update_traces(
        marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey"))
    )
    st.plotly_chart(fig_result, use_container_width=True)

    # 표로도 일부 샘플 확인
    st.markdown("#### 실제값 vs 예측값 (상위 10개 표)")
    st.dataframe(comparison_df.head(10))
