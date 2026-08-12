import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# ETF V5：10万元资金管理模型
#
# 目标：
# 不是预测每天涨跌，而是测试：
#
# 1. 买入持有
# 2. 原始20%×5分批建仓
# 3. V5智能分批建仓
#
# V5核心：
# - 第一笔：趋势正常才建仓
# - 回撤3%/6%/9%/12%不是无条件加仓
# - 必须同时满足趋势没有严重破坏
# - 如果出现明显下跌趋势，暂停加仓
# - 趋势恢复后重新允许加仓
#
# 初始资金：100000
# 每档：20000
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000

TRANCHE = 20000

FEE_RATE = 0.0005

ETF_LIST = [
    {
        "code": "159209",
        "name": "红利质量ETF",
        "yahoo": "159209.SZ"
    },
    {
        "code": "159399",
        "name": "现金流ETF",
        "yahoo": "159399.SZ"
    },
    {
        "code": "159581",
        "name": "红利ETF",
        "yahoo": "159581.SZ"
    }
]

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 获取历史数据
# ============================================================

def download_data(symbol):

    print(f"获取数据：{symbol}")

    try:

        df = yf.download(
            symbol,
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return None

        df = df.dropna(subset=["Close"])

        if len(df) < 200:
            return None

        return df

    except Exception as e:

        print(f"数据获取失败：{e}")

        return None


# ============================================================
# 指标
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    # 均线
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # 均线趋势
    df["MA20_SLOPE"] = (
        df["MA20"] / df["MA20"].shift(10) - 1
    )

    df["MA60_SLOPE"] = (
        df["MA60"] / df["MA60"].shift(20) - 1
    )

    # 动量
    df["RET20"] = close.pct_change(20)
    df["RET60"] = close.pct_change(60)
    df["RET120"] = close.pct_change(120)

    # 高点
    df["HIGH20"] = close.rolling(20).max()
    df["HIGH60"] = close.rolling(60).max()
    df["HIGH120"] = close.rolling(120).max()

    # 回撤
    df["DD20"] = close / df["HIGH20"] - 1
    df["DD60"] = close / df["HIGH60"] - 1
    df["DD120"] = close / df["HIGH120"] - 1

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - 100 / (1 + rs)

    # 波动率
    daily_return = close.pct_change()

    df["VOLATILITY20"] = (
        daily_return.rolling(20).std()
        * np.sqrt(252)
    )

    return df


# ============================================================
# V5趋势判断
# ============================================================

def trend_ok(row):

    required = [
        "Close",
        "MA20",
        "MA60",
        "MA120",
        "MA60_SLOPE",
        "RET60",
        "RET120"
    ]

    if any(pd.isna(row[x]) for x in required):
        return False

    price = row["Close"]

    # 核心趋势
    if price < row["MA120"]:
        return False

    # 中期趋势不能明显向下
    if row["MA60_SLOPE"] < -0.03:
        return False

    # 60日和120日动量不能同时严重为负
    if (
        row["RET60"] < -0.08
        and row["RET120"] < -0.05
    ):
        return False

    return True


# ============================================================
# V5风险判断
# ============================================================

def risk_high(row):

    required = [
        "Close",
        "MA60",
        "MA120",
        "MA60_SLOPE",
        "RET20",
        "RET60"
    ]

    if any(pd.isna(row[x]) for x in required):
        return True

    price = row["Close"]

    # 跌破120日均线
    if price < row["MA120"]:
        return True

    # MA60明显下降
    if row["MA60_SLOPE"] < -0.05:
        return True

    # 短中期同时恶化
    if (
        row["RET20"] < -0.08
        and row["RET60"] < -0.12
    ):
        return True

    return False


# ============================================================
# 策略1：买入持有
# ============================================================

def backtest_buy_hold(df):

    close = df["Close"]

    daily_return = close.pct_change().fillna(0)

    equity = INITIAL_CAPITAL * (
        1 + daily_return
    ).cumprod()

    return equity


# ============================================================
# 策略2：
# 你的原始机械分批策略
#
# 第一次：任意时间买2万
# 之后：
# -3%
# -6%
# -9%
# -12%
#
# 用于作为基准。
# ============================================================

def backtest_original_tranche(df):

    close = df["Close"]

    cash = INITIAL_CAPITAL

    shares = 0.0

    first_price = None

    tranche_count = 0

    equity_curve = []

    for price in close:

        price = float(price)

        # 第一次直接建仓
        if tranche_count == 0:

            amount = TRANCHE

            fee = amount * FEE_RATE

            if cash >= amount + fee:

                shares += amount / price

                cash -= amount + fee

                first_price = price

                tranche_count = 1

        # 后续回撤加仓
        elif tranche_count < 5:

            drawdown = (
                price / first_price - 1
            )

            target = -0.03 * tranche_count

            if drawdown <= target:

                amount = TRANCHE

                fee = amount * FEE_RATE

                if cash >= amount + fee:

                    shares += amount / price

                    cash -= amount + fee

                    tranche_count += 1

        equity_curve.append(
            cash + shares * price
        )

    return pd.Series(
        equity_curve,
        index=df.index
    )


# ============================================================
# 策略3：
# V5智能分批建仓
# ============================================================

def backtest_v5(df):

    close = df["Close"]

    cash = INITIAL_CAPITAL

    shares = 0.0

    first_entry_price = None

    tranche_count = 0

    equity_curve = []

    entries = 0

    paused_days = 0

    risk_days = 0

    # --------------------------------------------------------
    # 每一档对应：
    #
    # 第1档：趋势允许
    # 第2档：-3%
    # 第3档：-6%
    # 第4档：-9%
    # 第5档：-12%
    #
    # 但必须通过趋势检查。
    # --------------------------------------------------------

    for i in range(len(df)):

        price = float(close.iloc[i])

        row = df.iloc[i]

        trend = trend_ok(row)

        risk = risk_high(row)

        if risk:
            risk_days += 1

        # ====================================================
        # 第一档
        # ====================================================

        if tranche_count == 0:

            # 必须有正常趋势
            if trend and not risk:

                amount = TRANCHE

                fee = amount * FEE_RATE

                if cash >= amount + fee:

                    shares += amount / price

                    cash -= amount + fee

                    first_entry_price = price

                    tranche_count = 1

                    entries += 1

        # ====================================================
        # 后续加仓
        # ====================================================

        elif (
            tranche_count < 5
            and first_entry_price is not None
        ):

            drawdown = (
                price / first_entry_price - 1
            )

            target = -0.03 * tranche_count

            # ------------------------------------------------
            # 只有达到回撤档位才考虑加仓
            # ------------------------------------------------

            if drawdown <= target:

                # 风险高：
                # 暂停加仓
                if risk:

                    paused_days += 1

                else:

                    # ------------------------------------------------
                    # 加仓额仍为2万元
                    # ------------------------------------------------

                    amount = TRANCHE

                    fee = amount * FEE_RATE

                    if cash >= amount + fee:

                        shares += amount / price

                        cash -= amount + fee

                        tranche_count += 1

                        entries += 1

        # ====================================================
        # 风险退出
        #
        # 不是普通下跌就卖。
        #
        # 必须：
        # 1. 跌破MA120
        # 2. 风险条件持续
        #
        # 简化处理：
        # 当天风险严重时，全部退出。
        # ====================================================

        if (
            shares > 0
            and risk
            and price < row["MA120"]
        ):

            value = shares * price

            fee = value * FEE_RATE

            cash += value - fee

            shares = 0

            first_entry_price = None

            tranche_count = 0

        equity_curve.append(
            cash + shares * price
        )

    equity = pd.Series(
        equity_curve,
        index=df.index
    )

    return equity, entries, paused_days, risk_days


# ============================================================
# 指标
# ============================================================

def calculate_metrics(equity):

    equity = equity.dropna()

    if len(equity) < 2:

        return {
            "total_return": 0,
            "annual_return": 0,
            "max_drawdown": 0,
            "sharpe": 0
        }

    total_return = (
        equity.iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    years = len(equity) / 252

    if years <= 0:
        years = 1

    annual_return = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )

    peak = equity.cummax()

    drawdown = equity / peak - 1

    max_drawdown = drawdown.min()

    daily_return = equity.pct_change().fillna(0)

    std = daily_return.std()

    if std > 0:

        sharpe = (
            daily_return.mean()
            / std
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    return {
        "total_return": round(
            total_return * 100,
            2
        ),
        "annual_return": round(
            annual_return * 100,
            2
        ),
        "max_drawdown": round(
            max_drawdown * 100,
            2
        ),
        "sharpe": round(
            sharpe,
            2
        )
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print("ETF V5 资金管理模型")

    print("=" * 70)

    results = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        print()

        print(
            f"========== {code} {name} =========="
        )

        df = download_data(
            item["yahoo"]
        )

        if df is None:

            print("数据获取失败")

            continue

        df = calculate_indicators(df)

        # ----------------------------------------------------
        # 买入持有
        # ----------------------------------------------------

        bh = backtest_buy_hold(df)

        bh_metrics = calculate_metrics(bh)

        # ----------------------------------------------------
        # 原始分批
        # ----------------------------------------------------

        original = backtest_original_tranche(df)

        original_metrics = calculate_metrics(
            original
        )

        # ----------------------------------------------------
        # V5
        # ----------------------------------------------------

        v5, entries, paused, risk_days = (
            backtest_v5(df)
        )

        v5_metrics = calculate_metrics(
            v5
        )

        # ----------------------------------------------------
        # 结果
        # ----------------------------------------------------

        result = {

            "code": code,

            "name": name,

            "buy_hold_return":
                bh_metrics["total_return"],

            "buy_hold_annual":
                bh_metrics["annual_return"],

            "buy_hold_drawdown":
                bh_metrics["max_drawdown"],

            "buy_hold_sharpe":
                bh_metrics["sharpe"],

            "original_tranche_return":
                original_metrics["total_return"],

            "original_tranche_annual":
                original_metrics["annual_return"],

            "original_tranche_drawdown":
                original_metrics["max_drawdown"],

            "original_tranche_sharpe":
                original_metrics["sharpe"],

            "v5_return":
                v5_metrics["total_return"],

            "v5_annual":
                v5_metrics["annual_return"],

            "v5_drawdown":
                v5_metrics["max_drawdown"],

            "v5_sharpe":
                v5_metrics["sharpe"],

            "v5_entries":
                entries,

            "v5_paused_days":
                paused,

            "v5_risk_days":
                risk_days
        }

        results.append(result)

        # ----------------------------------------------------
        # 保存每日V5数据
        # ----------------------------------------------------

        output = pd.DataFrame({

            "date": df.index,

            "close": df["Close"].values,

            "ma20": df["MA20"].values,

            "ma60": df["MA60"].values,

            "ma120": df["MA120"].values,

            "ret20": df["RET20"].values,

            "ret60": df["RET60"].values,

            "ret120": df["RET120"].values,

            "dd60": df["DD60"].values,

            "dd120": df["DD120"].values,

            "rsi14": df["RSI14"].values

        })

        output.to_csv(
            os.path.join(
                DATA_DIR,
                f"etf_v5_{code}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"买入持有："
            f"{bh_metrics['total_return']}%"
        )

        print(
            f"原始分批："
            f"{original_metrics['total_return']}%"
        )

        print(
            f"V5："
            f"{v5_metrics['total_return']}%"
        )

    # ========================================================
    # 保存结果
    # ========================================================

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v5_result.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print("=" * 70)

    print("V5 回测完成")

    print("=" * 70)

    print()

    print(
        result_df.to_string(
            index=False
        )
    )

    print()

    print(
        "结果文件：data/etf_v5_result.csv"
    )


if __name__ == "__main__":

    main()
