import os
import math
import pandas as pd
import numpy as np


# ============================================================
# ETF V7.5
# 3%分批建仓 + 极端风险保护
# ============================================================

INITIAL_CAPITAL = 100000.0
TRANCHE_AMOUNT = 20000.0
ADD_STEP = 0.03

MAX_TRANCHES = 5

# ============================================================
# 极端风险保护参数
# ============================================================

MA_SHORT = 20
MA_LONG = 60

# 价格跌破60日均线超过8%
EXTREME_MA_GAP = 0.08

# 从持仓期间最高价回撤15%
EXTREME_DRAWDOWN = 0.15


ETF_LIST = [
    ("159209", "红利质量ETF"),
    ("159399", "现金流ETF"),
    ("159581", "红利ETF"),
]


# ============================================================
# 读取数据
# ============================================================

def load_data(code):
    path = f"data/etf_{code}_signals.csv"

    print(f"读取本地数据：{path}")

    if not os.path.exists(path):
        print("文件不存在")
        return None

    try:
        df = pd.read_csv(path)

        print(f"原始字段：{list(df.columns)}")

        required_columns = ["date", "price"]

        for col in required_columns:
            if col not in df.columns:
                print(f"缺少字段：{col}")
                return None

        # ----------------------------------------------------
        # 日期
        # ----------------------------------------------------
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # ----------------------------------------------------
        # 价格
        # ----------------------------------------------------
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # 删除无效数据
        df = df.dropna(subset=["date", "price"])

        # 价格必须大于0
        df = df[df["price"] > 0]

        # ----------------------------------------------------
        # 排序
        # ----------------------------------------------------
        df = df.sort_values("date")

        # ====================================================
        # 关键修复
        # ====================================================
        # 防止：
        # KeyError: 0
        #
        # 后面所有循环都使用新的连续整数索引
        # ====================================================

        df = df.reset_index(drop=True)

        # ----------------------------------------------------
        # 计算均线
        # ----------------------------------------------------

        df["ma20"] = df["price"].rolling(
            MA_SHORT,
            min_periods=1
        ).mean()

        df["ma60"] = df["price"].rolling(
            MA_LONG,
            min_periods=1
        ).mean()

        print(f"成功读取 {len(df)} 条数据")

        if len(df) > 0:
            print(
                f"数据范围："
                f"{df['date'].iloc[0].date()} → "
                f"{df['date'].iloc[-1].date()}"
            )

        return df

    except Exception as e:
        print(f"读取数据失败：{e}")
        return None


# ============================================================
# 计算最大回撤
# ============================================================

def calculate_max_drawdown(equity):

    if len(equity) == 0:
        return 0.0

    equity = np.array(equity, dtype=float)

    peak = np.maximum.accumulate(equity)

    drawdown = equity / peak - 1.0

    return float(drawdown.min() * 100)


# ============================================================
# Sharpe
# ============================================================

def calculate_sharpe(equity):

    if len(equity) < 2:
        return 0.0

    equity = pd.Series(equity, dtype=float)

    returns = equity.pct_change().dropna()

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or pd.isna(std):
        return 0.0

    sharpe = (
        returns.mean()
        / std
        * math.sqrt(252)
    )

    if pd.isna(sharpe) or np.isinf(sharpe):
        return 0.0

    return float(sharpe)


# ============================================================
# 买入持有
# ============================================================

def run_buy_hold(df):

    if df is None or len(df) < 2:
        return None

    initial_price = float(df["price"].iloc[0])

    equity = []

    for i in range(len(df)):

        price = float(df["price"].iloc[i])

        value = INITIAL_CAPITAL * price / initial_price

        equity.append(value)

    final_value = equity[-1]

    total_return = (
        final_value / INITIAL_CAPITAL - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    if days > 0:

        annual_return = (
            (final_value / INITIAL_CAPITAL)
            ** (365 / days)
            - 1
        ) * 100

    else:

        annual_return = 0.0

    max_drawdown = calculate_max_drawdown(equity)

    sharpe = calculate_sharpe(equity)

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
        "equity": equity,
    }


# ============================================================
# 3%原始分批建仓
# ============================================================

def run_original_strategy(df):

    if df is None or len(df) < 2:
        return None

    cash = INITIAL_CAPITAL

    shares = 0.0

    tranche_count = 0

    last_buy_price = None

    equity_curve = []

    capital_usage = []

    buy_count = 0

    for i in range(len(df)):

        # ====================================================
        # 使用 iloc
        # 不再使用 df.loc[i]
        # ====================================================

        price = float(df["price"].iloc[i])

        # ----------------------------------------------------
        # 第一次建仓
        # ----------------------------------------------------

        if tranche_count == 0:

            buy_amount = min(
                TRANCHE_AMOUNT,
                cash
            )

            shares += buy_amount / price

            cash -= buy_amount

            tranche_count += 1

            last_buy_price = price

            buy_count += 1

        # ----------------------------------------------------
        # 后续每下跌3%加仓
        # ----------------------------------------------------

        elif (
            tranche_count < MAX_TRANCHES
            and last_buy_price is not None
            and price <= last_buy_price * (1 - ADD_STEP)
        ):

            buy_amount = min(
                TRANCHE_AMOUNT,
                cash
            )

            if buy_amount > 0:

                shares += buy_amount / price

                cash -= buy_amount

                tranche_count += 1

                last_buy_price = price

                buy_count += 1

        # ----------------------------------------------------
        # 计算账户权益
        # ----------------------------------------------------

        market_value = shares * price

        total_equity = cash + market_value

        equity_curve.append(total_equity)

        usage = (
            market_value / INITIAL_CAPITAL * 100
        )

        capital_usage.append(usage)

    final_equity = equity_curve[-1]

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    if days > 0:

        annual_return = (
            (final_equity / INITIAL_CAPITAL)
            ** (365 / days)
            - 1
        ) * 100

    else:

        annual_return = 0.0

    max_drawdown = calculate_max_drawdown(
        equity_curve
    )

    sharpe = calculate_sharpe(
        equity_curve
    )

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
        "equity": equity_curve,
        "capital_usage": (
            np.mean(capital_usage)
            if capital_usage
            else 0
        ),
        "buy_count": buy_count,
    }


# ============================================================
# V7.5 风险保护策略
# ============================================================

def run_v75_strategy(df):

    if df is None or len(df) < 2:
        return None

    cash = INITIAL_CAPITAL

    shares = 0.0

    tranche_count = 0

    last_buy_price = None

    highest_price = None

    risk_protection = False

    risk_count = 0

    buy_count = 0

    sell_count = 0

    equity_curve = []

    capital_usage = []

    # --------------------------------------------------------
    # 风险保护重新允许建仓的条件
    # --------------------------------------------------------

    recovery_days = 0

    for i in range(len(df)):

        price = float(df["price"].iloc[i])

        ma20 = float(df["ma20"].iloc[i])

        ma60 = float(df["ma60"].iloc[i])

        # ====================================================
        # 更新持仓期间最高价
        # ====================================================

        if highest_price is None:

            highest_price = price

        else:

            highest_price = max(
                highest_price,
                price
            )

        # ====================================================
        # 极端风险判断
        # ====================================================

        ma_gap = 0.0

        if ma60 > 0:

            ma_gap = (
                price / ma60 - 1
            )

        price_drawdown = 0.0

        if highest_price > 0:

            price_drawdown = (
                price / highest_price - 1
            )

        extreme_risk = (

            ma_gap <= -EXTREME_MA_GAP

            or

            price_drawdown <= -EXTREME_DRAWDOWN
        )

        # ====================================================
        # 触发极端风险保护
        # ====================================================

        if (
            extreme_risk
            and shares > 0
        ):

            # 全部卖出
            cash += shares * price

            shares = 0.0

            tranche_count = 0

            last_buy_price = None

            risk_protection = True

            recovery_days = 0

            risk_count += 1

            sell_count += 1

        # ====================================================
        # 风险保护期间
        # ====================================================

        if risk_protection:

            # 必须连续恢复到安全状态
            safe_trend = (

                price >= ma60

                and

                ma20 >= ma60
            )

            if safe_trend:

                recovery_days += 1

            else:

                recovery_days = 0

            # 连续5个交易日恢复
            if recovery_days >= 5:

                risk_protection = False

                highest_price = price

                last_buy_price = None

                tranche_count = 0

        # ====================================================
        # 正常建仓
        # ====================================================

        if not risk_protection:

            # ------------------------------------------------
            # 第一次建仓
            # ------------------------------------------------

            if tranche_count == 0:

                buy_amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if buy_amount > 0:

                    shares += (
                        buy_amount / price
                    )

                    cash -= buy_amount

                    tranche_count += 1

                    last_buy_price = price

                    highest_price = price

                    buy_count += 1

            # ------------------------------------------------
            # 每下跌3%增加一档
            # ------------------------------------------------

            elif (
                tranche_count < MAX_TRANCHES
                and last_buy_price is not None
                and price <= (
                    last_buy_price
                    * (1 - ADD_STEP)
                )
            ):

                buy_amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if buy_amount > 0:

                    shares += (
                        buy_amount / price
                    )

                    cash -= buy_amount

                    tranche_count += 1

                    last_buy_price = price

                    buy_count += 1

        # ====================================================
        # 每日账户权益
        # ====================================================

        market_value = shares * price

        total_equity = (
            cash + market_value
        )

        equity_curve.append(
            total_equity
        )

        usage = (
            market_value
            / INITIAL_CAPITAL
            * 100
        )

        capital_usage.append(
            usage
        )

    # ========================================================
    # 最终结果
    # ========================================================

    final_equity = equity_curve[-1]

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    if days > 0:

        annual_return = (
            (final_equity / INITIAL_CAPITAL)
            ** (365 / days)
            - 1
        ) * 100

    else:

        annual_return = 0.0

    max_drawdown = calculate_max_drawdown(
        equity_curve
    )

    sharpe = calculate_sharpe(
        equity_curve
    )

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
        "equity": equity_curve,
        "capital_usage": (
            np.mean(capital_usage)
            if capital_usage
            else 0
        ),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "risk_count": risk_count,
    }


# ============================================================
# 打印单个ETF
# ============================================================

def print_result(
    code,
    name,
    df,
    buy_hold,
    original,
    v75
):

    print()
    print("=" * 70)
    print(f"{code} {name}")
    print("=" * 70)

    print(
        f"数据范围："
        f"{df['date'].iloc[0].date()} → "
        f"{df['date'].iloc[-1].date()}"
    )

    print(
        f"数据条数：{len(df)}"
    )

    # ========================================================
    # 买入持有
    # ========================================================

    print()
    print("【A 买入持有】")

    print(
        f"收益：{buy_hold['return']:.2f}%"
    )

    print(
        f"年化：{buy_hold['annual']:.2f}%"
    )

    print(
        f"最大回撤：{buy_hold['drawdown']:.2f}%"
    )

    print(
        f"Sharpe：{buy_hold['sharpe']:.2f}"
    )

    # ========================================================
    # 原始3%分批
    # ========================================================

    print()
    print("【B 原始3%分批建仓】")

    print(
        f"收益：{original['return']:.2f}%"
    )

    print(
        f"年化：{original['annual']:.2f}%"
    )

    print(
        f"最大回撤：{original['drawdown']:.2f}%"
    )

    print(
        f"Sharpe：{original['sharpe']:.2f}"
    )

    print(
        f"平均资金使用率："
        f"{original['capital_usage']:.2f}%"
    )

    print(
        f"建仓次数："
        f"{original['buy_count']}"
    )

    # ========================================================
    # V7.5
    # ========================================================

    print()
    print(
        "【C V7.5：3%分批 + 极端风险保护】"
    )

    print(
        f"收益：{v75['return']:.2f}%"
    )

    print(
        f"年化：{v75['annual']:.2f}%"
    )

    print(
        f"最大回撤：{v75['drawdown']:.2f}%"
    )

    print(
        f"Sharpe：{v75['sharpe']:.2f}"
    )

    print(
        f"平均资金使用率："
        f"{v75['capital_usage']:.2f}%"
    )

    print(
        f"建仓次数："
        f"{v75['buy_count']}"
    )

    print(
        f"风险保护触发次数："
        f"{v75['risk_count']}"
    )

    print(
        f"风险保护卖出次数："
        f"{v75['sell_count']}"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ETF V7.5：3%分批建仓 + 极端风险保护"
    )
    print("=" * 70)

    print()
    print(
        "参数："
    )

    print(
        f"初始资金：{INITIAL_CAPITAL:,.0f} 元"
    )

    print(
        f"每档金额：{TRANCHE_AMOUNT:,.0f} 元"
    )

    print(
        f"加仓间距：{ADD_STEP * 100:.0f}%"
    )

    print(
        f"最大档位：{MAX_TRANCHES}档"
    )

    print(
        f"极端风险：价格低于60日均线"
        f"{EXTREME_MA_GAP * 100:.0f}%"
        f"或从高点回撤"
        f"{EXTREME_DRAWDOWN * 100:.0f}%"
    )

    results = []

    for code, name in ETF_LIST:

        print()
        print("=" * 70)
        print(f"{code} {name}")
        print("=" * 70)

        df = load_data(code)

        if df is None:

            print("数据读取失败")
            continue

        if len(df) < 2:

            print("数据不足")
            continue

        # ====================================================
        # 三种策略
        # ====================================================

        buy_hold = run_buy_hold(df)

        original = run_original_strategy(df)

        v75 = run_v75_strategy(df)

        if (
            buy_hold is None
            or original is None
            or v75 is None
        ):

            print("回测失败")
            continue

        # ====================================================
        # 输出
        # ====================================================

        print_result(
            code,
            name,
            df,
            buy_hold,
            original,
            v75
        )

        # ====================================================
        # 结果保存
        # ====================================================

        result = {

            "code": code,

            "name": name,

            "data_start":
                df["date"].iloc[0].strftime(
                    "%Y-%m-%d"
                ),

            "data_end":
                df["date"].iloc[-1].strftime(
                    "%Y-%m-%d"
                ),

            "data_days":
                len(df),

            # ------------------------------------------------
            # 买入持有
            # ------------------------------------------------

            "buy_hold_return":
                round(
                    buy_hold["return"],
                    2
                ),

            "buy_hold_annual":
                round(
                    buy_hold["annual"],
                    2
                ),

            "buy_hold_drawdown":
                round(
                    buy_hold["drawdown"],
                    2
                ),

            "buy_hold_sharpe":
                round(
                    buy_hold["sharpe"],
                    2
                ),

            # ------------------------------------------------
            # 原始3%
            # ------------------------------------------------

            "original_return":
                round(
                    original["return"],
                    2
                ),

            "original_annual":
                round(
                    original["annual"],
                    2
                ),

            "original_drawdown":
                round(
                    original["drawdown"],
                    2
                ),

            "original_sharpe":
                round(
                    original["sharpe"],
                    2
                ),

            "original_capital_usage":
                round(
                    original["capital_usage"],
                    2
                ),

            # ------------------------------------------------
            # V7.5
            # ------------------------------------------------

            "v75_return":
                round(
                    v75["return"],
                    2
                ),

            "v75_annual":
                round(
                    v75["annual"],
                    2
                ),

            "v75_drawdown":
                round(
                    v75["drawdown"],
                    2
                ),

            "v75_sharpe":
                round(
                    v75["sharpe"],
                    2
                ),

            "v75_capital_usage":
                round(
                    v75["capital_usage"],
                    2
                ),

            "v75_buy_count":
                v75["buy_count"],

            "v75_sell_count":
                v75["sell_count"],

            "risk_protection_count":
                v75["risk_count"],

            # ------------------------------------------------
            # V7.5相对买入持有
            # ------------------------------------------------

            "return_vs_buy_hold":
                round(
                    v75["return"]
                    - buy_hold["return"],
                    2
                ),

            "drawdown_improvement":
                round(
                    v75["drawdown"]
                    - buy_hold["drawdown"],
                    2
                ),

            "sharpe_improvement":
                round(
                    v75["sharpe"]
                    - buy_hold["sharpe"],
                    2
                ),
        }

        results.append(result)

    # ========================================================
    # 汇总
    # ========================================================

    print()
    print("=" * 70)
    print("V7.5最终汇总")
    print("=" * 70)

    if len(results) == 0:

        print("没有成功完成回测")

        return

    result_df = pd.DataFrame(results)

    # ========================================================
    # 排名
    # ========================================================

    def calculate_score(row):

        score = 0.0

        # 收益
        score += row["v75_return"] * 1.0

        # Sharpe
        score += row["v75_sharpe"] * 10.0

        # 回撤改善
        score += (
            row["drawdown_improvement"]
            * 0.5
        )

        # 样本外/风险保护模型不能只看收益
        if (
            row["v75_return"]
            > row["buy_hold_return"]
        ):
            score += 5

        if (
            row["v75_sharpe"]
            > row["buy_hold_sharpe"]
        ):
            score += 5

        return round(score, 4)

    result_df["model_score"] = (
        result_df.apply(
            calculate_score,
            axis=1
        )
    )

    result_df = result_df.sort_values(
        "model_score",
        ascending=False
    ).reset_index(drop=True)

    result_df.insert(
        0,
        "final_rank",
        range(1, len(result_df) + 1)
    )

    # ========================================================
    # 打印最终排名
    # ========================================================

    print()

    columns_to_show = [
        "final_rank",
        "code",
        "name",
        "buy_hold_return",
        "original_return",
        "v75_return",
        "buy_hold_drawdown",
        "original_drawdown",
        "v75_drawdown",
        "buy_hold_sharpe",
        "original_sharpe",
        "v75_sharpe",
        "risk_protection_count",
        "model_score",
    ]

    print(
        result_df[
            columns_to_show
        ].to_string(index=False)
    )

    # ========================================================
    # 保存CSV
    # ========================================================

    os.makedirs(
        "data",
        exist_ok=True
    )

    output_path = (
        "data/etf_v7_5_result.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)

    print(
        f"V7.5结果已保存：{output_path}"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
