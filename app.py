import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ============================================================
# ETF V8 实盘助手
# ============================================================

st.set_page_config(
    page_title="ETF V8 实盘助手",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# 基本参数
# ============================================================

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000
TRANCHE_COUNT = 5
ADD_STEP = 0.03

ETF_NAMES = {
    "159209": "红利质量ETF",
    "159399": "现金流ETF",
    "159581": "红利ETF"
}

# ============================================================
# 文件读取
# ============================================================

def load_csv(path):
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ============================================================
# 读取ETF最新数据
# ============================================================

def load_etf(code):

    path = f"data/etf_{code}_signals.csv"

    df = load_csv(path)

    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# 读取V8结果
# ============================================================

def load_v8():

    path = "data/etf_v8_final_result.csv"

    df = load_csv(path)

    if df is None or df.empty:
        return None

    return df


# ============================================================
# 读取交易记录
# ============================================================

def load_trades():

    path = "data/trades.csv"

    if not os.path.exists(path):
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "action",
                "price",
                "amount"
            ]
        )

    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "action",
                "price",
                "amount"
            ]
        )

    return df


# ============================================================
# 计算建仓档位
# ============================================================

def calculate_levels(price):

    levels = []

    for i in range(TRANCHE_COUNT):

        level_price = price * ((1 - ADD_STEP) ** i)

        levels.append({
            "档位": i + 1,
            "价格": level_price,
            "金额": TRANCHE_AMOUNT
        })

    return pd.DataFrame(levels)


# ============================================================
# 计算实际持仓
# ============================================================

def calculate_position(code, trades):

    if trades.empty:
        return {
            "shares": 0,
            "cost": 0,
            "buy_amount": 0,
            "sell_amount": 0
        }

    df = trades[trades["code"].astype(str) == str(code)].copy()

    if df.empty:
        return {
            "shares": 0,
            "cost": 0,
            "buy_amount": 0,
            "sell_amount": 0
        }

    shares = 0
    cost = 0
    buy_amount = 0
    sell_amount = 0

    for _, row in df.iterrows():

        action = str(row["action"]).upper()

        price = float(row["price"])
        amount = float(row["amount"])

        if action == "BUY":

            buy_shares = amount / price

            shares += buy_shares
            cost += amount
            buy_amount += amount

        elif action == "SELL":

            sell_shares = amount / price

            shares -= sell_shares
            sell_amount += amount

            if shares < 0:
                shares = 0

    return {
        "shares": shares,
        "cost": cost,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount
    }


# ============================================================
# 页面标题
# ============================================================

st.title("📈 ETF V8 实盘助手")

st.caption(
    "V8模型冻结｜严格样本外验证优先｜工作日自动更新"
)

st.divider()


# ============================================================
# 更新时间
# ============================================================

st.info(
    f"页面打开时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


# ============================================================
# 读取数据
# ============================================================

v8 = load_v8()
trades = load_trades()


# ============================================================
# V8最终排名
# ============================================================

st.header("🏆 V8 最终模型排名")

if v8 is None:

    st.error(
        "暂时找不到 data/etf_v8_final_result.csv"
    )

else:

    show_cols = [
        "final_rank",
        "code",
        "name",
        "strategy_return",
        "strategy_drawdown",
        "strategy_sharpe",
        "validation_windows",
        "validation_win_rate",
        "validation_return",
        "validation_sharpe",
        "v8_score",
        "rating"
    ]

    available_cols = [
        c for c in show_cols
        if c in v8.columns
    ]

    ranking = v8[available_cols].copy()

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 当前核心ETF
# ============================================================

st.header("🎯 当前核心ETF")

core_code = "159581"

core_df = load_etf(core_code)

if core_df is None:

    st.error("无法读取159581数据")

else:

    latest = core_df.iloc[-1]

    latest_price = float(latest["price"])
    latest_date = latest["date"]

    position = calculate_position(
        core_code,
        trades
    )

    shares = position["shares"]
    cost = position["cost"]

    market_value = shares * latest_price

    profit = market_value - cost

    if cost > 0:
        profit_pct = profit / cost * 100
    else:
        profit_pct = 0

    # --------------------------------------------------------
    # 均线
    # --------------------------------------------------------

    ma20 = core_df["price"].rolling(20).mean().iloc[-1]
    ma60 = core_df["price"].rolling(60).mean().iloc[-1]

    if latest_price > ma20 and latest_price > ma60:
        trend = "🟢 强势"

    elif latest_price > ma60:
        trend = "🟡 中性"

    else:
        trend = "🔴 弱势"

    # --------------------------------------------------------
    # 建仓档位
    # --------------------------------------------------------

    if cost <= 0:
        current_level = 0
    else:
        current_level = min(
            int(round(cost / TRANCHE_AMOUNT)),
            TRANCHE_COUNT
        )

    # --------------------------------------------------------
    # 下一档价格
    # --------------------------------------------------------

    next_level = current_level + 1

    if next_level <= TRANCHE_COUNT:

        next_price = latest_price * (
            (1 - ADD_STEP) ** current_level
        )

    else:

        next_price = None

    # --------------------------------------------------------
    # V8评级
    # --------------------------------------------------------

    rating = "未知"

    if v8 is not None:

        row = v8[
            v8["code"].astype(str) == core_code
        ]

        if not row.empty:

            rating = str(
                row.iloc[0].get(
                    "rating",
                    "未知"
                )
            )

    # --------------------------------------------------------
    # 页面
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "最新价格",
        f"{latest_price:.4f}"
    )

    c2.metric(
        "20日均线",
        f"{ma20:.4f}"
    )

    c3.metric(
        "60日均线",
        f"{ma60:.4f}"
    )

    c4.metric(
        "趋势",
        trend
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "当前投入",
        f"¥{cost:,.0f}"
    )

    c2.metric(
        "当前市值",
        f"¥{market_value:,.0f}"
    )

    c3.metric(
        "实盘收益",
        f"¥{profit:,.0f}"
    )

    c4.metric(
        "实盘收益率",
        f"{profit_pct:.2f}%"
    )

    st.write("")

    st.subheader("📌 当前模型状态")

    st.write(
        f"**V8评级：{rating}**"
    )

    st.write(
        f"当前建仓：**{current_level} / {TRANCHE_COUNT}档**"
    )

    if current_level < TRANCHE_COUNT:

        if current_level == 0:

            st.success(
                "🟢 可以进行首次建仓"
            )

        else:

            st.info(
                f"下一档参考价格："
                f"**{next_price:.4f}**"
            )

    else:

        st.warning(
            "已完成5档建仓"
        )


# ============================================================
# 完整价格计划
# ============================================================

st.header("📊 3%分批建仓价格计划")

if core_df is not None:

    levels = calculate_levels(
        float(core_df.iloc[-1]["price"])
    )

    levels["价格"] = levels["价格"].map(
        lambda x: f"{x:.4f}"
    )

    levels["金额"] = levels["金额"].map(
        lambda x: f"¥{x:,.0f}"
    )

    st.dataframe(
        levels,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "实际首次成交后，建议以后续实际成交价重新计算价格计划。"
    )


# ============================================================
# 三只ETF当前状态
# ============================================================

st.header("👀 三只ETF实时观察")

rows = []

for code, name in ETF_NAMES.items():

    df = load_etf(code)

    if df is None or df.empty:
        continue

    latest = df.iloc[-1]

    price = float(latest["price"])

    ma20 = df["price"].rolling(20).mean().iloc[-1]
    ma60 = df["price"].rolling(60).mean().iloc[-1]

    if price > ma20 and price > ma60:
        trend = "🟢 强势"
    elif price > ma60:
        trend = "🟡 中性"
    else:
        trend = "🔴 弱势"

    rating = "—"

    if v8 is not None:

        row = v8[
            v8["code"].astype(str) == code
        ]

        if not row.empty:
            rating = str(
                row.iloc[0].get(
                    "rating",
                    "—"
                )
            )

    rows.append({
        "代码": code,
        "ETF": name,
        "最新价格": price,
        "趋势": trend,
        "V8评级": rating,
        "最新日期": latest["date"].strftime(
            "%Y-%m-%d"
        )
    })

if rows:

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 实盘交易记录
# ============================================================

st.header("🧾 实盘交易记录")

if trades.empty:

    st.info(
        "目前还没有真实成交记录。"
    )

else:

    st.dataframe(
        trades,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 说明
# ============================================================

st.divider()

st.caption(
    "说明：V8模型只负责模型评价与操作参考，"
    "实际交易由投资者自行决定。"
)

st.caption(
    "当前模型核心ETF：159581 红利ETF。"
)
