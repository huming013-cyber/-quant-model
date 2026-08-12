import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# ETF V6：10万元分批建仓优化模型
#
# 核心思想：
# 不预测涨跌，不使用复杂择时。
#
# 自动测试不同的回撤加仓间距：
# 2%、3%、4%、5%、6%、8%、10%
#
# 总资金：100000元
# 每次投入：20000元
# 最多：5档
#
# 比较：
# 1. 买入持有
# 2. 不同回撤间距的分批建仓
#
# 最后自动寻找：
# 每只ETF历史回测中表现最好的加仓间距
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000

TRANCHE = 20000

FEE_RATE = 0.0005

# 要测试的回撤幅度
DRAWDOWN_LEVELS = [
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10
]

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
# 获取数据
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

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 100:
            return None

        return df

    except Exception as e:

        print(
            f"数据获取失败：{e}"
        )

        return None


# ============================================================
# 买入持有
# ============================================================

def backtest_buy_hold(df):

    close = df["Close"]

    daily_return = (
        close
        .pct_change()
        .fillna(0)
    )

    equity = (
        INITIAL_CAPITAL
        * (1 + daily_return).cumprod()
    )

    return equity


# ============================================================
# 分批建仓
#
# 规则：
#
# 第一次：
# 直接投入2万元
#
# 后面：
#
# 回撤达到指定幅度
# 就增加2万元
#
# 例如3%：
#
# 2万元
# ↓ -3%
# 再2万元
# ↓ -6%
# 再2万元
# ↓ -9%
# 再2万元
# ↓ -12%
# 再2万元
#
# 一共最多10万元。
# ============================================================

def backtest_tranche(
    df,
    drawdown_step
):

    close = df["Close"]

    cash = INITIAL_CAPITAL

    shares = 0.0

    first_price = None

    tranche_count = 0

    equity_curve = []

    entry_count = 0

    for price in close:

        price = float(price)

        # ====================================================
        # 第一档
        # ====================================================

        if tranche_count == 0:

            amount = TRANCHE

            fee = (
                amount
                * FEE_RATE
            )

            if cash >= amount + fee:

                shares += (
                    amount / price
                )

                cash -= (
                    amount + fee
                )

                first_price = price

                tranche_count = 1

                entry_count += 1

        # ====================================================
        # 后续加仓
        # ====================================================

        elif (
            tranche_count < 5
            and first_price is not None
        ):

            drawdown = (
                price / first_price
                - 1
            )

            # 第2档：
            # -step
            #
            # 第3档：
            # -2*step
            #
            # 第4档：
            # -3*step
            #
            # 第5档：
            # -4*step

            target = (
                -drawdown_step
                * tranche_count
            )

            if drawdown <= target:

                amount = TRANCHE

                fee = (
                    amount
                    * FEE_RATE
                )

                if cash >= amount + fee:

                    shares += (
                        amount / price
                    )

                    cash -= (
                        amount + fee
                    )

                    tranche_count += 1

                    entry_count += 1

        # ====================================================
        # 每日资产
        # ====================================================

        equity_curve.append(
            cash + shares * price
        )

    equity = pd.Series(
        equity_curve,
        index=df.index
    )

    return equity, entry_count


# ============================================================
# 计算指标
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

    years = (
        len(equity)
        / 252
    )

    if years <= 0:

        years = 1

    annual_return = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )

    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    daily_return = (
        equity
        .pct_change()
        .fillna(0)
    )

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
# 选择最佳策略
#
# 不单纯按照收益率选择。
#
# V6评分：
#
# 50%：年化收益
# 30%：Sharpe
# 20%：最大回撤
#
# 回撤越小越好。
# ============================================================

def strategy_score(metrics):

    annual = metrics["annual_return"]

    sharpe = metrics["sharpe"]

    drawdown = abs(
        metrics["max_drawdown"]
    )

    score = (
        annual * 0.50
        + sharpe * 10 * 0.30
        - drawdown * 0.20
    )

    return round(
        score,
        3
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print(
        "ETF V6：10万元分批建仓优化模型"
    )

    print("=" * 70)

    all_results = []

    best_results = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        print()

        print(
            "=" * 70
        )

        print(
            f"{code} {name}"
        )

        print(
            "=" * 70
        )

        df = download_data(
            item["yahoo"]
        )

        if df is None:

            print(
                "数据获取失败，跳过"
            )

            continue

        # ====================================================
        # 买入持有
        # ====================================================

        buy_hold = (
            backtest_buy_hold(df)
        )

        bh_metrics = (
            calculate_metrics(
                buy_hold
            )
        )

        print()

        print(
            "买入持有："
            f"{bh_metrics['total_return']}%"
        )

        # ====================================================
        # 测试所有加仓间距
        # ====================================================

        for level in DRAWDOWN_LEVELS:

            equity, entries = (
                backtest_tranche(
                    df,
                    level
                )
            )

            metrics = (
                calculate_metrics(
                    equity
                )
            )

            score = (
                strategy_score(
                    metrics
                )
            )

            result = {

                "code": code,

                "name": name,

                "drawdown_step":
                    f"{level * 100:.0f}%",

                "total_return":
                    metrics["total_return"],

                "annual_return":
                    metrics["annual_return"],

                "max_drawdown":
                    metrics["max_drawdown"],

                "sharpe":
                    metrics["sharpe"],

                "entries":
                    entries,

                "score":
                    score
            }

            all_results.append(
                result
            )

            print(
                f"加仓间距 "
                f"{level * 100:.0f}%："
                f"收益 "
                f"{metrics['total_return']}% | "
                f"年化 "
                f"{metrics['annual_return']}% | "
                f"回撤 "
                f"{metrics['max_drawdown']}% | "
                f"Sharpe "
                f"{metrics['sharpe']} | "
                f"评分 "
                f"{score}"
            )

        # ====================================================
        # 找最佳
        # ====================================================

        current_results = [
            x
            for x in all_results
            if x["code"] == code
        ]

        current_df = pd.DataFrame(
            current_results
        )

        best = current_df.sort_values(
            "score",
            ascending=False
        ).iloc[0]

        best_result = {

            "code": code,

            "name": name,

            "best_drawdown_step":
                best["drawdown_step"],

            "best_return":
                best["total_return"],

            "best_annual":
                best["annual_return"],

            "best_drawdown":
                best["max_drawdown"],

            "best_sharpe":
                best["sharpe"],

            "best_entries":
                best["entries"],

            "best_score":
                best["score"],

            "buy_hold_return":
                bh_metrics["total_return"],

            "buy_hold_annual":
                bh_metrics["annual_return"],

            "buy_hold_drawdown":
                bh_metrics["max_drawdown"],

            "buy_hold_sharpe":
                bh_metrics["sharpe"]
        }

        best_results.append(
            best_result
        )

        print()

        print(
            "★ 最佳加仓间距："
            f"{best['drawdown_step']}"
        )

        print(
            "★ 最佳策略收益："
            f"{best['total_return']}%"
        )

    # ========================================================
    # 保存全部测试结果
    # ========================================================

    all_df = pd.DataFrame(
        all_results
    )

    all_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v6_all_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 保存最佳结果
    # ========================================================

    best_df = pd.DataFrame(
        best_results
    )

    best_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v6_best_result.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print("=" * 70)

    print(
        "V6回测完成"
    )

    print("=" * 70)

    print()

    print(
        "最佳策略汇总："
    )

    print()

    print(
        best_df.to_string(
            index=False
        )
    )

    print()

    print(
        "已生成："
    )

    print(
        "data/etf_v6_all_results.csv"
    )

    print(
        "data/etf_v6_best_result.csv"
    )


if __name__ == "__main__":

    main()
