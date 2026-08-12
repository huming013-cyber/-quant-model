import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

LIST_FILE = "stock_list.csv"
DATA_DIR = "data"

INITIAL_CAPITAL = 100000
FEE_RATE = 0.0005

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# A股代码转换
# ============================================================

def convert_code(asset_type, code):

    code = str(code).strip()

    if asset_type.lower() == "etf":

        if code.startswith(("15", "16")):
            return code + ".SZ"

        if code.startswith(("51", "56", "58")):
            return code + ".SS"

        return code

    if code.startswith(("60", "68", "51", "58")):
        return code + ".SS"

    if code.startswith(("00", "30", "12", "15", "16")):
        return code + ".SZ"

    return code


# ============================================================
# 技术指标
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    df["RET20"] = close.pct_change(20)
    df["RET60"] = close.pct_change(60)
    df["RET120"] = close.pct_change(120)

    df["HIGH20"] = close.rolling(20).max()
    df["HIGH120"] = close.rolling(120).max()

    df["DRAWDOWN20"] = close / df["HIGH20"] - 1
    df["DRAWDOWN120"] = close / df["HIGH120"] - 1

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["SIGNAL"]
    )

    # 成交量
    if "Volume" in df.columns:

        df["VOL20"] = df["Volume"].rolling(20).mean()

        df["VOLUME_RATIO"] = (
            df["Volume"] / df["VOL20"]
        )

    else:

        df["VOLUME_RATIO"] = np.nan

    # 波动率
    daily_return = close.pct_change()

    df["VOLATILITY20"] = (
        daily_return.rolling(20).std()
        * np.sqrt(252)
    )

    return df


# ============================================================
# 单日评分
# ============================================================

def get_score(row):

    score = 0

    price = row["Close"]

    # -------------------------
    # 趋势 25分
    # -------------------------

    if price > row["MA20"]:
        score += 8

    if price > row["MA60"]:
        score += 7

    if row["MA20"] > row["MA60"]:
        score += 5

    if row["MA60"] > row["MA120"]:
        score += 5

    # -------------------------
    # 动量 20分
    # -------------------------

    if row["RET20"] > 0:
        score += 8

    if row["RET20"] > 0.05:
        score += 2

    if row["RET60"] > 0:
        score += 5

    if row["RET120"] > 0:
        score += 5

    # -------------------------
    # 回撤 15分
    # -------------------------

    if row["DRAWDOWN20"] >= -0.03:
        score += 8

    elif row["DRAWDOWN20"] >= -0.08:
        score += 5

    elif row["DRAWDOWN20"] >= -0.15:
        score += 2

    if row["DRAWDOWN120"] >= -0.10:
        score += 7

    elif row["DRAWDOWN120"] >= -0.20:
        score += 4

    # -------------------------
    # RSI 10分
    # -------------------------

    rsi = row["RSI14"]

    if 50 <= rsi <= 65:
        score += 10

    elif 45 <= rsi < 50:
        score += 8

    elif 65 < rsi <= 70:
        score += 8

    elif 35 <= rsi < 45:
        score += 5

    elif rsi < 35:
        score += 3

    elif rsi > 75:
        score += 2

    else:
        score += 5

    # -------------------------
    # MACD 10分
    # -------------------------

    if row["MACD"] > row["SIGNAL"]:
        score += 7

    if row["MACD_HIST"] > 0:
        score += 3

    # -------------------------
    # 成交量 8分
    # -------------------------

    volume_ratio = row["VOLUME_RATIO"]

    if pd.isna(volume_ratio):

        score += 5

    elif 0.8 <= volume_ratio <= 1.8:

        score += 8

    elif volume_ratio > 1.8:

        score += 6

    else:

        score += 4

    # -------------------------
    # 稳定性 10分
    # -------------------------

    volatility = row["VOLATILITY20"]

    if pd.isna(volatility):

        score += 5

    elif volatility < 0.20:

        score += 10

    elif volatility < 0.30:

        score += 8

    elif volatility < 0.40:

        score += 6

    elif volatility < 0.50:

        score += 4

    else:

        score += 2

    return min(score, 100)


# ============================================================
# 生成每日信号
# ============================================================

def generate_signals(df):

    df = df.copy()

    df["score"] = np.nan

    for i in range(len(df)):

        row = df.iloc[i]

        required = [
            "MA20",
            "MA60",
            "MA120",
            "RET20",
            "RET60",
            "RET120",
            "DRAWDOWN20",
            "DRAWDOWN120",
            "RSI14",
            "MACD",
            "SIGNAL",
            "MACD_HIST",
            "VOLATILITY20"
        ]

        if any(
            pd.isna(row[x])
            for x in required
        ):
            continue

        df.iloc[i, df.columns.get_loc("score")] = (
            get_score(row)
        )

    # --------------------------------------------------------
    # 交易规则
    #
    # >=75：进入持仓
    # <60 ：退出持仓
    # 60~74：保持之前仓位
    # --------------------------------------------------------

    position = []
    current = 0

    for score in df["score"]:

        if pd.isna(score):

            position.append(0)
            continue

        if score >= 75:

            current = 1

        elif score < 60:

            current = 0

        position.append(current)

    df["position"] = position

    # --------------------------------------------------------
    # 防止未来函数
    #
    # 今天产生信号
    # 下一交易日执行
    # --------------------------------------------------------

    df["position_used"] = (
        df["position"].shift(1).fillna(0)
    )

    return df


# ============================================================
# 回测
# ============================================================

def run_backtest(df):

    df = df.copy()

    close = df["Close"]

    daily_return = close.pct_change().fillna(0)

    # 策略收益
    strategy_return = (
        daily_return
        * df["position_used"]
    )

    # 计算换仓
    trades = (
        df["position_used"]
        .diff()
        .abs()
        .fillna(0)
    )

    # 交易成本
    strategy_return = (
        strategy_return
        - trades * FEE_RATE
    )

    df["strategy_return"] = strategy_return

    # 策略净值
    df["strategy_equity"] = (
        1 + df["strategy_return"]
    ).cumprod()

    # 买入持有
    df["buy_hold_return"] = daily_return

    df["buy_hold_equity"] = (
        1 + df["buy_hold_return"]
    ).cumprod()

    return df


# ============================================================
# 计算指标
# ============================================================

def calculate_metrics(df):

    strategy_equity = df["strategy_equity"]

    buy_hold_equity = df["buy_hold_equity"]

    days = len(df)

    years = days / 252

    if years <= 0:
        years = 1

    strategy_total = (
        strategy_equity.iloc[-1] - 1
    )

    buy_hold_total = (
        buy_hold_equity.iloc[-1] - 1
    )

    strategy_annual = (
        (1 + strategy_total)
        ** (1 / years)
        - 1
    )

    buy_hold_annual = (
        (1 + buy_hold_total)
        ** (1 / years)
        - 1
    )

    # 最大回撤
    rolling_max = (
        strategy_equity.cummax()
    )

    drawdown = (
        strategy_equity / rolling_max - 1
    )

    max_drawdown = drawdown.min()

    # 夏普
    daily_std = (
        df["strategy_return"].std()
    )

    if daily_std > 0:

        sharpe = (
            df["strategy_return"].mean()
            / daily_std
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    trade_count = int(
        df["position_used"]
        .diff()
        .abs()
        .sum()
        / 2
    )

    return {

        "strategy_total_return": round(
            strategy_total * 100,
            2
        ),

        "buy_hold_total_return": round(
            buy_hold_total * 100,
            2
        ),

        "strategy_annual_return": round(
            strategy_annual * 100,
            2
        ),

        "buy_hold_annual_return": round(
            buy_hold_annual * 100,
            2
        ),

        "max_drawdown": round(
            max_drawdown * 100,
            2
        ),

        "sharpe": round(
            sharpe,
            2
        ),

        "trade_count": trade_count
    }


# ============================================================
# 分析单个标的
# ============================================================

def backtest_asset(
    asset_type,
    code,
    name
):

    yahoo_code = convert_code(
        asset_type,
        code
    )

    print()
    print("=" * 60)
    print(name)
    print(yahoo_code)

    try:

        df = yf.download(
            yahoo_code,
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:

            print("❌ 数据获取失败")

            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        if "Close" not in df.columns:

            print("❌ 缺少Close")

            return None

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 200:

            print("❌ 数据不足")

            return None

        df = calculate_indicators(df)

        df = generate_signals(df)

        df = run_backtest(df)

        metrics = calculate_metrics(df)

        print(
            "策略收益：",
            metrics["strategy_total_return"],
            "%"
        )

        print(
            "买入持有：",
            metrics["buy_hold_total_return"],
            "%"
        )

        print(
            "最大回撤：",
            metrics["max_drawdown"],
            "%"
        )

        print(
            "夏普：",
            metrics["sharpe"]
        )

        print(
            "交易次数：",
            metrics["trade_count"]
        )

        return metrics

    except Exception as e:

        print(
            "❌ 回测失败：",
            e
        )

        return None


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 60)
    print("V3 历史回测")
    print("=" * 60)

    stocks = pd.read_csv(
        LIST_FILE
    )

    results = []

    for _, row in stocks.iterrows():

        asset_type = str(
            row["type"]
        ).strip()

        code = str(
            row["code"]
        ).strip()

        name = str(
            row["name"]
        ).strip()

        metrics = backtest_asset(
            asset_type,
            code,
            name
        )

        if metrics is not None:

            result = {
                "type": asset_type,
                "code": code,
                "name": name,
                **metrics
            }

            results.append(result)

        time.sleep(1)

    if not results:

        print("❌ 没有回测结果")

        return

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "strategy_annual_return",
        ascending=False
    )

    result_df.insert(
        0,
        "rank",
        range(
            1,
            len(result_df) + 1
        )
    )

    output = os.path.join(
        DATA_DIR,
        "backtest_result.csv"
    )

    result_df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("V3 回测完成")
    print("=" * 60)

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print(
        "结果文件：",
        output
    )


if __name__ == "__main__":
    main()
