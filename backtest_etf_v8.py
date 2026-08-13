import os
import pandas as pd
import numpy as np

# ============================================================
# ETF V8
# 模型冻结 + 严格样本外最终评价
#
# 核心原则：
# 1. 不再根据完整历史选择新参数
# 2. 样本外结果优先于历史回测
# 3. 验证窗口不足时降低可信度
# 4. 买入持有仅作为基准
# 5. 最终输出三只ETF排名
# 6. 给出10万元配置建议
# ============================================================

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000

ETF_LIST = [
    ("159209", "红利质量ETF"),
    ("159399", "现金流ETF"),
    ("159581", "红利ETF"),
]

DATA_DIR = "data"


# ============================================================
# 工具函数
# ============================================================

def load_data(code):
    filename = os.path.join(
        DATA_DIR,
        f"etf_{code}_signals.csv"
    )

    print(f"读取本地数据：{filename}")

    if not os.path.exists(filename):
        print("数据文件不存在")
        return None

    df = pd.read_csv(filename)

    print(f"原始字段：{list(df.columns)}")

    required = ["date", "price"]

    for col in required:
        if col not in df.columns:
            print(f"缺少字段：{col}")
            return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date")
    df = df.drop_duplicates("date")
    df = df.reset_index(drop=True)

    if len(df) < 30:
        print("数据不足")
        return None

    print(f"成功读取 {len(df)} 条数据")
    print(
        f"数据范围："
        f"{df['date'].iloc[0].date()} → "
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 买入持有
# ============================================================

def buy_hold(df):

    start_price = df["price"].iloc[0]
    end_price = df["price"].iloc[-1]

    total_return = (
        end_price / start_price - 1
    ) * 100

    days = (
        df["date"].iloc[-1] -
        df["date"].iloc[0]
    ).days

    if days > 0:
        annual_return = (
            (end_price / start_price)
            ** (365 / days)
            - 1
        ) * 100
    else:
        annual_return = 0

    equity = df["price"] / start_price

    drawdown = (
        equity / equity.cummax() - 1
    )

    max_drawdown = drawdown.min() * 100

    daily_return = df["price"].pct_change().dropna()

    if daily_return.std() > 0:
        sharpe = (
            daily_return.mean()
            / daily_return.std()
            * np.sqrt(252)
        )
    else:
        sharpe = 0

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
    }


# ============================================================
# 3%分批建仓
# ============================================================

def tranche_strategy(
    df,
    step=0.03
):

    prices = df["price"].values

    cash = INITIAL_CAPITAL
    position = 0.0

    first_price = prices[0]

    next_buy_price = first_price
    buy_count = 0

    equity_curve = []

    for price in prices:

        # ----------------------------------------------------
        # 首次建仓
        # ----------------------------------------------------

        if buy_count == 0:

            if price <= next_buy_price:

                amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if amount > 0:

                    position += (
                        amount / price
                    )

                    cash -= amount

                    buy_count += 1

                    next_buy_price = (
                        price * (1 - step)
                    )

        # ----------------------------------------------------
        # 后续加仓
        # ----------------------------------------------------

        elif buy_count < 5:

            if price <= next_buy_price:

                amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if amount > 0:

                    position += (
                        amount / price
                    )

                    cash -= amount

                    buy_count += 1

                    next_buy_price = (
                        price * (1 - step)
                    )

        equity = (
            cash +
            position * price
        )

        equity_curve.append(equity)

    final_equity = equity_curve[-1]

    total_return = (
        final_equity /
        INITIAL_CAPITAL
        - 1
    ) * 100

    days = (
        df["date"].iloc[-1] -
        df["date"].iloc[0]
    ).days

    if days > 0:

        annual_return = (
            (final_equity / INITIAL_CAPITAL)
            ** (365 / days)
            - 1
        ) * 100

    else:
        annual_return = 0

    equity_series = pd.Series(
        equity_curve
    )

    drawdown = (
        equity_series /
        equity_series.cummax()
        - 1
    )

    max_drawdown = (
        drawdown.min() * 100
    )

    daily_equity = (
        equity_series.pct_change()
        .dropna()
    )

    if daily_equity.std() > 0:

        sharpe = (
            daily_equity.mean()
            /
            daily_equity.std()
            *
            np.sqrt(252)
        )

    else:
        sharpe = 0

    average_usage = (
        np.mean(
            [
                (
                    INITIAL_CAPITAL - c
                ) / INITIAL_CAPITAL
                for c in
                [
                    max(
                        0,
                        INITIAL_CAPITAL -
                        min(
                            INITIAL_CAPITAL,
                            TRANCHE_AMOUNT *
                            buy_count
                        )
                    )
                ]
            ]
        )
    )

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
        "entries": buy_count,
    }


# ============================================================
# 根据已有V7结果判断
# ============================================================

def get_v7_information(code):

    # --------------------------------------------------------
    # 这里使用已经完成的 V7.3 / V7.6 样本外结果
    # --------------------------------------------------------

    if code == "159209":

        return {
            "windows": 1,
            "win_rate": 0.0,
            "oos_return": -3.56,
            "oos_annual": -8.56,
            "oos_drawdown": -12.75,
            "oos_sharpe": -0.55,
            "parameter_stability": 100.0,
        }

    if code == "159399":

        return {
            "windows": 1,
            "win_rate": 0.0,
            "oos_return": -10.68,
            "oos_annual": -23.35,
            "oos_drawdown": -18.31,
            "oos_sharpe": -1.89,
            "parameter_stability": 100.0,
        }

    if code == "159581":

        return {
            "windows": 3,
            "win_rate": 100.0,
            "oos_return": 1.79,
            "oos_annual": 3.62,
            "oos_drawdown": -2.50,
            "oos_sharpe": 0.80,
            "parameter_stability": 66.67,
        }

    return {
        "windows": 0,
        "win_rate": 0,
        "oos_return": 0,
        "oos_annual": 0,
        "oos_drawdown": 0,
        "oos_sharpe": 0,
        "parameter_stability": 0,
    }


# ============================================================
# V8评分
# ============================================================

def calculate_v8_score(info):

    windows = info["windows"]
    win_rate = info["win_rate"]
    oos_return = info["oos_return"]
    oos_sharpe = info["oos_sharpe"]
    stability = info["parameter_stability"]

    # --------------------------------------------------------
    # 样本外收益
    # --------------------------------------------------------

    score_return = oos_return * 2

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    score_sharpe = oos_sharpe * 10

    # --------------------------------------------------------
    # 胜率
    # --------------------------------------------------------

    score_win = win_rate * 0.05

    # --------------------------------------------------------
    # 参数稳定性
    # --------------------------------------------------------

    score_stability = stability * 0.02

    # --------------------------------------------------------
    # 样本窗口数量奖励
    # --------------------------------------------------------

    score_windows = min(
        windows,
        3
    ) * 2

    # --------------------------------------------------------
    # 样本不足惩罚
    # --------------------------------------------------------

    if windows < 2:
        penalty = 10

    else:
        penalty = 0

    score = (
        score_return
        + score_sharpe
        + score_win
        + score_stability
        + score_windows
        - penalty
    )

    return round(score, 4)


# ============================================================
# 最终评级
# ============================================================

def rating(info, score):

    windows = info["windows"]
    win_rate = info["win_rate"]
    sharpe = info["oos_sharpe"]
    oos_return = info["oos_return"]

    if (
        windows >= 3
        and win_rate >= 66
        and oos_return > 0
        and sharpe > 0.5
    ):
        return "A：优先"

    if (
        windows >= 2
        and win_rate >= 50
        and oos_return > 0
        and sharpe > 0
    ):
        return "B：观察"

    if (
        oos_return > 0
        and sharpe > 0
    ):
        return "C：谨慎"

    return "D：暂缓"


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ETF V8：模型冻结 + 严格样本外最终评价")
    print("=" * 70)

    print()
    print("核心原则：")
    print("1. 不再使用验证期数据调参")
    print("2. 样本外结果优先")
    print("3. 买入持有仅作为基准")
    print("4. 验证窗口不足降低可信度")
    print("5. 不因为历史收益漂亮而直接买入")
    print()

    results = []

    for code, name in ETF_LIST:

        print()
        print("=" * 70)
        print(code, name)
        print("=" * 70)

        df = load_data(code)

        if df is None:
            continue

        # ----------------------------------------------------
        # 买入持有
        # ----------------------------------------------------

        bh = buy_hold(df)

        print()
        print("【买入持有基准】")

        print(
            f"收益：{bh['return']:.2f}%"
        )

        print(
            f"年化：{bh['annual']:.2f}%"
        )

        print(
            f"最大回撤：{bh['drawdown']:.2f}%"
        )

        print(
            f"Sharpe：{bh['sharpe']:.2f}"
        )

        # ----------------------------------------------------
        # 3%分批
        # ----------------------------------------------------

        strategy = tranche_strategy(
            df,
            step=0.03
        )

        print()
        print("【冻结3%分批策略】")

        print(
            f"收益：{strategy['return']:.2f}%"
        )

        print(
            f"年化：{strategy['annual']:.2f}%"
        )

        print(
            f"最大回撤：{strategy['drawdown']:.2f}%"
        )

        print(
            f"Sharpe：{strategy['sharpe']:.2f}"
        )

        print(
            f"建仓档位：{strategy['entries']}/5"
        )

        # ----------------------------------------------------
        # V7样本外
        # ----------------------------------------------------

        info = get_v7_information(
            code
        )

        score = calculate_v8_score(
            info
        )

        final_rating = rating(
            info,
            score
        )

        print()
        print("【严格样本外结果】")

        print(
            f"验证窗口："
            f"{info['windows']}"
        )

        if info["windows"] == 0:

            print(
                "样本外数据不足"
            )

        else:

            print(
                f"胜率："
                f"{info['win_rate']:.2f}%"
            )

            print(
                f"平均收益："
                f"{info['oos_return']:.2f}%"
            )

            print(
                f"平均年化："
                f"{info['oos_annual']:.2f}%"
            )

            print(
                f"平均最大回撤："
                f"{info['oos_drawdown']:.2f}%"
            )

            print(
                f"平均Sharpe："
                f"{info['oos_sharpe']:.2f}"
            )

            print(
                f"参数稳定性："
                f"{info['parameter_stability']:.2f}%"
            )

        print()
        print("【V8最终评价】")

        print(
            f"V8模型评分："
            f"{score:.4f}"
        )

        print(
            f"最终评级："
            f"{final_rating}"
        )

        results.append({

            "code": code,
            "name": name,

            "buy_hold_return":
                round(
                    bh["return"],
                    2
                ),

            "buy_hold_annual":
                round(
                    bh["annual"],
                    2
                ),

            "buy_hold_drawdown":
                round(
                    bh["drawdown"],
                    2
                ),

            "buy_hold_sharpe":
                round(
                    bh["sharpe"],
                    2
                ),

            "strategy_return":
                round(
                    strategy["return"],
                    2
                ),

            "strategy_annual":
                round(
                    strategy["annual"],
                    2
                ),

            "strategy_drawdown":
                round(
                    strategy["drawdown"],
                    2
                ),

            "strategy_sharpe":
                round(
                    strategy["sharpe"],
                    2
                ),

            "validation_windows":
                info["windows"],

            "validation_win_rate":
                info["win_rate"],

            "validation_return":
                info["oos_return"],

            "validation_annual":
                info["oos_annual"],

            "validation_drawdown":
                info["oos_drawdown"],

            "validation_sharpe":
                info["oos_sharpe"],

            "parameter_stability":
                info[
                    "parameter_stability"
                ],

            "v8_score":
                score,

            "rating":
                final_rating,
        })

    # ========================================================
    # 最终排名
    # ========================================================

    if not results:
        print("没有可用结果")
        return

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "v8_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    result_df[
        "final_rank"
    ] = np.arange(
        1,
        len(result_df) + 1
    )

    # ========================================================
    # 打印最终排名
    # ========================================================

    print()
    print("=" * 70)
    print("V8最终排名")
    print("=" * 70)

    print(
        result_df[
            [
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
                "parameter_stability",
                "v8_score",
                "rating",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # 10万元配置建议
    # ========================================================

    print()
    print("=" * 70)
    print("V8：10万元最终配置建议")
    print("=" * 70)

    # --------------------------------------------------------
    # 只允许真正通过样本外验证的ETF进入核心组合
    # --------------------------------------------------------

    qualified = result_df[
        (
            result_df[
                "validation_windows"
            ] >= 2
        )
        &
        (
            result_df[
                "validation_win_rate"
            ] >= 50
        )
        &
        (
            result_df[
                "validation_return"
            ] > 0
        )
        &
        (
            result_df[
                "validation_sharpe"
            ] > 0
        )
    ]

    if len(qualified) == 0:

        print()
        print(
            "当前没有ETF达到核心组合标准。"
        )

        print(
            "建议：100,000元暂不强制配置。"
        )

    elif len(qualified) == 1:

        row = qualified.iloc[0]

        print()
        print(
            f"核心ETF："
            f"{row['code']} "
            f"{row['name']}"
        )

        print(
            "建议最高配置：100,000元"
        )

        print(
            "首次建仓：20,000元"
        )

        print(
            "之后按照3%分批策略执行"
        )

        print()
        print(
            "其他ETF：暂缓"
        )

    else:

        # ----------------------------------------------------
        # 多个合格ETF按照V8评分比例分配
        # ----------------------------------------------------

        total_score = qualified[
            "v8_score"
        ].sum()

        print()

        for _, row in qualified.iterrows():

            weight = (
                row["v8_score"]
                / total_score
            )

            amount = (
                INITIAL_CAPITAL
                * weight
            )

            amount = (
                int(amount / 10000)
                * 10000
            )

            print(
                f"{row['code']} "
                f"{row['name']}："
                f"{amount:,} 元"
            )

    # ========================================================
    # 保存CSV
    # ========================================================

    output_file = os.path.join(
        DATA_DIR,
        "etf_v8_final_result.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("V8完成")
    print("=" * 70)

    print()
    print(
        f"结果已保存："
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
