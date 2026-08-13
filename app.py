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

CORE_CODE = "159581"

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
# ETF行情
# ============================================================

def load_etf(code):

    path = f"data/etf_{code}_signals.csv"

    df = load_csv(path)

    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "price"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


# ============================================================
# V8结果
# ============================================================

def load_v8():

    path = "data/etf_v8_final_result.csv"

    df = load_csv(path)

    if df is None or df.empty:
        return None

    if "code" in df.columns:
        df["code"] = (
            df["code"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

    return df


# ============================================================
# 读取交易记录
#
# 优先读取根目录 trades.csv
# 同时兼容旧的 data/trades.csv
# ============================================================

def load_trades():

    possible_paths = [
        "trades.csv",
        "data/trades.csv"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            try:

                df = pd.read_csv(path)

                if not df.empty:
                    return df

            except Exception:
                pass

    return pd.DataFrame(
        columns=[
            "date",
            "code",
            "action",
            "price",
            "amount"
        ]
    )


# ============================================================
# 读取历史实际成交价格
# ============================================================

def get_actual_first_price(
    code,
    trades
):

    if trades.empty:
        return None

    if "code" not in trades.columns:
        return None

    df = trades[
        trades["code"].astype(str) == str(code)
    ].copy()

    if df.empty:
        return None

    if "action" not in df.columns:
        return None

    buys = df[
        df["action"]
        .astype(str)
        .str.upper() == "BUY"
    ].copy()

    if buys.empty:
        return None

    if "date" in buys.columns:

        buys["date"] = pd.to_datetime(
            buys["date"],
            errors="coerce"
        )

        buys = buys.sort_values(
            "date"
        )

    try:
        return float(
            buys.iloc[0]["price"]
        )
    except Exception:
        return None


# ============================================================
# 根据首次实际成交价格计算固定3%档位
# ============================================================

def calculate_fixed_levels(
    base_price
):

    levels = []

    for i in range(TRANCHE_COUNT):

        price = (
            base_price *
            ((1 - ADD_STEP) ** i)
        )

        levels.append({
            "档位": i + 1,
            "触发价格": price,
            "金额": TRANCHE_AMOUNT
        })

    return pd.DataFrame(levels)


# ============================================================
# 计算实际持仓
# ============================================================

def calculate_position(
    code,
    trades
):

    result = {
        "shares": 0.0,
        "cost": 0.0,
        "buy_amount": 0.0,
        "sell_amount": 0.0
    }

    if trades.empty:
        return result

    if "code" not in trades.columns:
        return result

    df = trades[
        trades["code"].astype(str) == str(code)
    ].copy()

    if df.empty:
        return result

    for _, row in df.iterrows():

        try:

            action = str(
                row.get("action", "")
            ).upper()

            price = float(
                row.get("price", 0)
            )

            amount = float(
                row.get("amount", 0)
            )

            if price <= 0 or amount <= 0:
                continue

            shares = amount / price

            if action == "BUY":

                result["shares"] += shares
                result["cost"] += amount
                result["buy_amount"] += amount

            elif action == "SELL":

                result["shares"] -= shares
                result["sell_amount"] += amount

        except Exception:
            continue

    result["shares"] = max(
        result["shares"],
        0
    )

    return result


# ============================================================
# 计算建仓档位
# ============================================================

def get_current_level(
    cost
):

    if cost <= 0:
        return 0

    level = int(
        round(
            cost / TRANCHE_AMOUNT
        )
    )

    return min(
        max(level, 0),
        TRANCHE_COUNT
    )


# ============================================================
# 获取V8评级
# ============================================================

def get_rating(
    code,
    v8
):

    if v8 is None:
        return "未知"

    if "code" not in v8.columns:
        return "未知"

    row = v8[
        v8["code"].astype(str) == str(code)
    ]

    if row.empty:
        return "未知"

    return str(
        row.iloc[0].get(
            "rating",
            "未知"
        )
    )


# ============================================================
# 趋势判断
# ============================================================

def get_trend(df):

    if len(df) < 60:
        return "数据不足", None, None

    ma20 = (
        df["price"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    ma60 = (
        df["price"]
        .rolling(60)
        .mean()
        .iloc[-1]
    )

    price = float(
        df.iloc[-1]["price"]
    )

    if price > ma20 and price > ma60:

        trend = "🟢 强势"

    elif price > ma60:

        trend = "🟡 中性"

    else:

        trend = "🔴 弱势"

    return trend, ma20, ma60


# ============================================================
# 页面标题
# ============================================================

st.title(
    "📈 ETF V8 实盘助手"
)

st.caption(
    "V8模型冻结｜严格样本外验证优先｜工作日自动更新"
)

st.divider()


# ============================================================
# 当前时间
# ============================================================

st.info(
    "App打开时间："
    + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


# ============================================================
# 读取数据
# ============================================================

v8 = load_v8()
trades = load_trades()


# ============================================================
# V8最终排名
# ============================================================

st.header(
    "🏆 V8最终模型排名"
)

if v8 is None:

    st.error(
        "找不到 data/etf_v8_final_result.csv"
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
        c
        for c in show_cols
        if c in v8.columns
    ]

    ranking = v8[
        available_cols
    ].copy()

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 今日操作结论
# ============================================================

st.header(
    "🚦 今日实盘操作"
)

core_df = load_etf(
    CORE_CODE
)

if core_df is None:

    st.error(
        "无法读取159581行情数据"
    )

else:

    latest = core_df.iloc[-1]

    latest_price = float(
        latest["price"]
    )

    latest_date = latest["date"]

    trend, ma20, ma60 = get_trend(
        core_df
    )

    position = calculate_position(
        CORE_CODE,
        trades
    )

    shares = position["shares"]
    cost = position["cost"]

    market_value = (
        shares *
        latest_price
    )

    profit = (
        market_value -
        cost
    )

    if cost > 0:

        profit_pct = (
            profit /
            cost *
            100
        )

    else:

        profit_pct = 0


    # --------------------------------------------------------
    # V8评级
    # --------------------------------------------------------

    rating = get_rating(
        CORE_CODE,
        v8
    )


    # --------------------------------------------------------
    # 当前档位
    # --------------------------------------------------------

    current_level = get_current_level(
        cost
    )


    # --------------------------------------------------------
    # 首次实际成交价格
    # --------------------------------------------------------

    first_buy_price = get_actual_first_price(
        CORE_CODE,
        trades
    )


    # --------------------------------------------------------
    # 价格计划基准
    #
    # 如果已经真实成交：
    # 使用首次实际成交价
    #
    # 如果还没有成交：
    # 使用当前价格作为参考
    # --------------------------------------------------------

    if first_buy_price is not None:

        base_price = first_buy_price
        base_text = (
            "首次实际成交价"
        )

    else:

        base_price = latest_price
        base_text = (
            "当前价格参考"
        )


    levels = calculate_fixed_levels(
        base_price
    )


    # --------------------------------------------------------
    # 下一档
    # --------------------------------------------------------

    if current_level < TRANCHE_COUNT:

        next_level_number = (
            current_level + 1
        )

        next_price = float(
            levels.iloc[
                current_level
            ]["触发价格"]
        )

    else:

        next_level_number = None
        next_price = None


    # --------------------------------------------------------
    # 今日操作逻辑
    # --------------------------------------------------------

    if "A" in rating:

        if current_level == 0:

            action_text = (
                "🟢 首次建仓"
            )

            action_detail = (
                f"建议买入 ¥{TRANCHE_AMOUNT:,.0f}"
            )

        elif current_level < TRANCHE_COUNT:

            if next_price is not None:

                if latest_price <= next_price:

                    action_text = (
                        "🟢 触发加仓"
                    )

                    action_detail = (
                        f"第{next_level_number}档"
                        f"买入 ¥{TRANCHE_AMOUNT:,.0f}"
                    )

                else:

                    action_text = (
                        "🟡 暂不加仓"
                    )

                    action_detail = (
                        f"等待价格 ≤ "
                        f"{next_price:.4f}"
                    )

            else:

                action_text = (
                    "🟡 暂不加仓"
                )

                action_detail = (
                    "等待下一档价格"
                )

        else:

            action_text = (
                "🟡 已完成建仓"
            )

            action_detail = (
                "当前不再增加仓位"
            )

    else:

        action_text = (
            "🔴 暂缓买入"
        )

        action_detail = (
            "V8评级不是A，不执行新建仓"
        )


    # --------------------------------------------------------
    # 大号操作结论
    # --------------------------------------------------------

    if "🟢" in action_text:

        st.success(
            f"### {action_text}\n\n"
            f"**{action_detail}**"
        )

    elif "🔴" in action_text:

        st.error(
            f"### {action_text}\n\n"
            f"**{action_detail}**"
        )

    else:

        st.warning(
            f"### {action_text}\n\n"
            f"**{action_detail}**"
        )


    # --------------------------------------------------------
    # 核心数据
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "最新价格",
        f"{latest_price:.4f}"
    )

    c2.metric(
        "20日均线",
        f"{ma20:.4f}"
        if ma20 is not None
        else "-"
    )

    c3.metric(
        "60日均线",
        f"{ma60:.4f}"
        if ma60 is not None
        else "-"
    )

    c4.metric(
        "趋势",
        trend
    )


    # --------------------------------------------------------
    # 持仓
    # --------------------------------------------------------

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
        "浮动盈亏",
        f"¥{profit:,.0f}"
    )

    c4.metric(
        "收益率",
        f"{profit_pct:.2f}%"
    )


    # --------------------------------------------------------
    # 模型状态
    # --------------------------------------------------------

    st.subheader(
        "📌 模型状态"
    )

    st.write(
        f"**V8评级：{rating}**"
    )

    st.write(
        f"当前建仓："
        f"**{current_level} / "
        f"{TRANCHE_COUNT}档**"
    )

    st.write(
        f"数据日期："
        f"**{latest_date.strftime('%Y-%m-%d')}**"
    )


    # --------------------------------------------------------
    # 下一档
    # --------------------------------------------------------

    if next_price is not None:

        st.info(
            f"📍 第{next_level_number}档触发价格："
            f"**{next_price:.4f}**\n\n"
            f"达到该价格后："
            f"**买入 ¥{TRANCHE_AMOUNT:,.0f}**"
        )

    else:

        st.success(
            "已经完成5档建仓。"
        )


# ============================================================
# 完整固定价格计划
# ============================================================

st.header(
    "📊 3%分批建仓价格计划"
)

if core_df is not None:

    latest_price = float(
        core_df.iloc[-1]["price"]
    )

    first_buy_price = get_actual_first_price(
        CORE_CODE,
        trades
    )

    if first_buy_price is not None:

        base_price = first_buy_price

        st.success(
            f"价格计划已经锁定。"
            f"基准：首次实际成交价 "
            f"**{first_buy_price:.4f}**"
        )

    else:

        base_price = latest_price

        st.info(
            f"目前尚未记录实际成交。"
            f"当前价格 **{latest_price:.4f}** "
            f"仅作为首次建仓参考。"
        )

    levels = calculate_fixed_levels(
        base_price
    )

    levels["触发价格"] = levels[
        "触发价格"
    ].map(
        lambda x: f"{x:.4f}"
    )

    levels["金额"] = levels[
        "金额"
    ].map(
        lambda x: f"¥{x:,.0f}"
    )

    st.dataframe(
        levels,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "规则：首次实际成交后，后续4档价格按照首次实际成交价固定计算，不随每日价格变化。"
    )


# ============================================================
# 三只ETF当前状态
# ============================================================

st.header(
    "👀 三只ETF当前状态"
)

rows = []

for code, name in ETF_NAMES.items():

    df = load_etf(code)

    if df is None or df.empty:
        continue

    latest = df.iloc[-1]

    price = float(
        latest["price"]
    )

    trend, ma20, ma60 = get_trend(
        df
    )

    rating = get_rating(
        code,
        v8
    )

    rows.append({
        "代码": code,
        "ETF": name,
        "最新价格": f"{price:.4f}",
        "趋势": trend,
        "V8评级": rating,
        "最新日期": latest[
            "date"
        ].strftime("%Y-%m-%d")
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

st.header(
    "🧾 实盘交易记录"
)

if trades.empty:

    st.info(
        "目前没有读取到交易记录。"
        "如果你已经成交，请把实际成交记录写入 trades.csv。"
    )

else:

    st.dataframe(
        trades,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 资金状态
# ============================================================

st.header(
    "💰 资金状态"
)

core_position = calculate_position(
    CORE_CODE,
    trades
)

used_capital = core_position[
    "cost"
]

remaining_capital = (
    INITIAL_CAPITAL -
    used_capital
)

if remaining_capital < 0:
    remaining_capital = 0

c1, c2, c3 = st.columns(3)

c1.metric(
    "总资金",
    f"¥{INITIAL_CAPITAL:,.0f}"
)

c2.metric(
    "已投入",
    f"¥{used_capital:,.0f}"
)

c3.metric(
    "剩余资金",
    f"¥{remaining_capital:,.0f}"
)


# ============================================================
# 使用说明
# ============================================================

st.divider()

st.subheader(
    "📖 每天怎么看"
)

st.markdown(
    """
**每天只需要重点看「🚦 今日实盘操作」。**

- 🟢 **首次建仓** → 可以考虑买入20,000元
- 🟢 **触发加仓** → 达到对应价格，可以考虑加仓20,000元
- 🟡 **暂不加仓** → 继续等待
- 🔴 **暂缓买入** → 不执行新的建仓

价格计划以**首次实际成交价格**为基准锁定。

V8负责模型评价和交易参考，实际交易仍由投资者自行确认。
"""
)

st.caption(
    "ETF V8 实盘助手｜模型冻结版"
)
