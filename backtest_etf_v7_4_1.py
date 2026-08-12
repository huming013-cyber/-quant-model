import os
import pandas as pd
import numpy as np

# ============================================================
# ETF V7.4.1
# 建仓信号严格滚动样本外验证
#
# 目的：
# 验证首次建仓是否应该加入趋势过滤：
#
# A：原始分批策略
# B：价格 > MA20 > MA60 才允许首次建仓
# C：原始 signal + 趋势过滤
#
# 注意：
# 3%分批加仓逻辑保持不变。
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000

STEPS = [
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10
]

ETF_LIST = {
    "159209": "红利质量ETF",
    "159399": "现金流ETF",
    "159581": "红利ETF"
}


# ============================================================
# 读取数据
# ============================================================

def load_data(code):

    path = os.path.join(
        DATA_DIR,
        f"etf_{code}_signals.csv"
    )

    print(
        f"读取本地数据：{path}"
    )

    if not os.path.exists(path):

        print(
            f"❌ 找不到：{path}"
        )

        return None

    try:

        df = pd.read_csv(path)

        print(
            f"原始字段：{list(df.columns)}"
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        if "score" in df.columns:

            df["score"] = pd.to_numeric(
                df["score"],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "date",
                "price"
            ]
        )

        df = df[
            df["price"] > 0
        ]

        df = df.sort_values(
            "date"
        )

        df = df.drop_duplicates(
            "date",
            keep="last"
        )

        # ====================================================
        # 技术指标
        # ====================================================

        df["ma20"] = (
            df["price"]
            .rolling(20)
            .mean()
        )

        df["ma60"] = (
            df["price"]
            .rolling(60)
            .mean()
        )

        # ====================================================
        # 趋势过滤
        # ====================================================

        df["trend_strong"] = (
            (df["price"] > df["ma20"])
            &
            (df["ma20"] > df["ma60"])
        )

        print(
            f"成功读取 {len(df)} 条数据"
        )

        print(
            f"数据范围："
            f"{df['date'].iloc[0].strftime('%Y-%m-%d')}"
            f" → "
            f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
        )

        return df

    except Exception as e:

        print(
            f"❌ 数据读取失败：{e}"
        )

        return None


# ============================================================
# 最大回撤
# ============================================================

def max_drawdown(values):

    values = pd.Series(
        values
    )

    peak = values.cummax()

    drawdown = (
        values / peak - 1
    )

    return drawdown.min() * 100


# ============================================================
# Sharpe
# ============================================================

def sharpe_ratio(values):

    values = pd.Series(
        values
    )

    returns = values.pct_change()

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    returns = returns.dropna()

    if len(returns) < 2:

        return 0

    std = returns.std()

    if std == 0 or pd.isna(std):

        return 0

    return (
        returns.mean()
        /
        std
        *
        np.sqrt(252)
    )


# ============================================================
# 回测
#
# entry_mode：
#
# "original"
# 原始策略：
# 第一天直接建仓
#
# "trend"
# 只有 price > MA20 > MA60 才首次建仓
#
# "signal_trend"
# 原始 signal + 趋势过滤
# ============================================================

def run_strategy(
    df,
    step,
    entry_mode
):

    cash = INITIAL_CAPITAL

    shares = 0

    invested = 0

    first_buy_price = None

    entry_done = False

    next_add_price = None

    portfolio_values = []

    capital_used = []

    signals = []

    for i in range(len(df)):

        row = df.iloc[i]

        price = row["price"]

        # ====================================================
        # 默认信号
        # ====================================================

        action = "HOLD"

        # ====================================================
        # 首次建仓
        # ====================================================

        if not entry_done:

            allow_entry = False

            # -----------------------------------------------
            # A：原始策略
            # -----------------------------------------------

            if entry_mode == "original":

                allow_entry = True

            # -----------------------------------------------
            # B：趋势过滤
            # -----------------------------------------------

            elif entry_mode == "trend":

                if bool(row["trend_strong"]):

                    allow_entry = True

            # -----------------------------------------------
            # C：signal + 趋势
            # -----------------------------------------------

            elif entry_mode == "signal_trend":

                signal_ok = False

                if "signal" in df.columns:

                    signal_value = str(
                        row["signal"]
                    ).lower()

                    if signal_value in [
                        "buy",
                        "买入",
                        "1",
                        "true",
                        "strong_buy",
                        "加仓"
                    ]:

                        signal_ok = True

                # 如果没有明确买入signal，
                # score较高时允许作为辅助条件
                if (
                    not signal_ok
                    and "score" in df.columns
                ):

                    score = row["score"]

                    if (
                        pd.notna(score)
                        and score >= 0.5
                    ):

                        signal_ok = True

                allow_entry = (
                    signal_ok
                    and bool(row["trend_strong"])
                )

            # =================================================
            # 执行首次建仓
            # =================================================

            if allow_entry:

                buy_amount = TRANCHE_AMOUNT

                buy_amount = min(
                    buy_amount,
                    cash
                )

                if buy_amount > 0:

                    shares += (
                        buy_amount
                        / price
                    )

                    cash -= buy_amount

                    invested += buy_amount

                    first_buy_price = price

                    entry_done = True

                    next_add_price = (
                        first_buy_price
                        * (1 - step)
                    )

                    action = "BUY_FIRST"

        # ====================================================
        # 已经建仓
        # ====================================================

        else:

            # -----------------------------------------------
            # 是否达到下一次加仓价格
            # -----------------------------------------------

            if (
                next_add_price is not None
                and price <= next_add_price
                and invested < INITIAL_CAPITAL
            ):

                buy_amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if buy_amount > 0:

                    shares += (
                        buy_amount
                        / price
                    )

                    cash -= buy_amount

                    invested += buy_amount

                    action = "BUY_ADD"

                    # 下一次加仓价格
                    next_add_price = (
                        next_add_price
                        * (1 - step)
                    )

        # ====================================================
        # 每日资产
        # ====================================================

        portfolio_value = (
            cash
            +
            shares * price
        )

        portfolio_values.append(
            portfolio_value
        )

        capital_used.append(
            invested
            / INITIAL_CAPITAL
            * 100
        )

        signals.append(
            action
        )

    # ========================================================
    # 结果
    # ========================================================

    values = pd.Series(
        portfolio_values
    )

    total_return = (
        values.iloc[-1]
        /
        INITIAL_CAPITAL
        - 1
    ) * 100

    days = len(df)

    if days > 1:

        annual_return = (
            (
                values.iloc[-1]
                /
                INITIAL_CAPITAL
            )
            **
            (252 / days)
            - 1
        ) * 100

    else:

        annual_return = 0

    drawdown = max_drawdown(
        values
    )

    sharpe = sharpe_ratio(
        values
    )

    avg_usage = np.mean(
        capital_used
    )

    return {

        "return": total_return,

        "annual": annual_return,

        "drawdown": drawdown,

        "sharpe": sharpe,

        "capital_usage":
            avg_usage,

        "final_value":
            values.iloc[-1],

        "signals":
            signals

    }


# ============================================================
# 找到训练期最佳参数
#
# 注意：
# 只能使用训练数据。
# ============================================================

def select_best_step(
    train_df,
    entry_mode
):

    results = []

    for step in STEPS:

        result = run_strategy(
            train_df,
            step,
            entry_mode
        )

        # 综合评分：
        # 收益 + Sharpe + 回撤控制
        score = (
            result["annual"] * 0.45
            +
            result["sharpe"] * 10 * 0.35
            +
            result["drawdown"] * 0.20
        )

        results.append(
            {
                "step": step,
                "score": score,
                "return":
                    result["return"],
                "annual":
                    result["annual"],
                "drawdown":
                    result["drawdown"],
                "sharpe":
                    result["sharpe"]
            }
        )

    results_df = pd.DataFrame(
        results
    )

    best = results_df.loc[
        results_df["score"].idxmax()
    ]

    return (
        float(best["step"]),
        results_df
    )


# ============================================================
# 严格滚动样本外验证
# ============================================================

def rolling_validation(
    df,
    entry_mode
):

    n = len(df)

    # --------------------------------------------------------
    # 至少需要：
    # 约一年训练 + 半年验证
    # --------------------------------------------------------

    train_days = 252

    validation_days = 126

    windows = []

    start = 0

    while (
        start
        + train_days
        + validation_days
        <= n
    ):

        train_start = start

        train_end = (
            start
            + train_days
        )

        validation_end = (
            train_end
            + validation_days
        )

        train_df = df.iloc[
            train_start:train_end
        ].copy()

        validation_df = df.iloc[
            train_end:validation_end
        ].copy()

        # ----------------------------------------------------
        # 训练期选择参数
        # ----------------------------------------------------

        best_step, _ = (
            select_best_step(
                train_df,
                entry_mode
            )
        )

        # ----------------------------------------------------
        # 样本外验证
        # ----------------------------------------------------

        validation_result = (
            run_strategy(
                validation_df,
                best_step,
                entry_mode
            )
        )

        windows.append(
            {
                "train_start":
                    train_df["date"].iloc[0],

                "train_end":
                    train_df["date"].iloc[-1],

                "validation_start":
                    validation_df["date"].iloc[0],

                "validation_end":
                    validation_df["date"].iloc[-1],

                "step":
                    best_step,

                "return":
                    validation_result["return"],

                "annual":
                    validation_result["annual"],

                "drawdown":
                    validation_result["drawdown"],

                "sharpe":
                    validation_result["sharpe"],

                "capital_usage":
                    validation_result["capital_usage"]
            }
        )

        # ----------------------------------------------------
        # 滚动
        # ----------------------------------------------------

        start += validation_days

    return windows


# ============================================================
# 打印窗口
# ============================================================

def print_windows(
    windows,
    mode_name
):

    print()

    print(
        f"【{mode_name}】"
    )

    if len(windows) == 0:

        print(
            "没有足够数据形成样本外窗口"
        )

        return

    for i, w in enumerate(
        windows,
        start=1
    ):

        print()

        print(
            f"窗口 {i}"
        )

        print(
            f"训练期："
            f"{w['train_start'].strftime('%Y-%m-%d')}"
            f" → "
            f"{w['train_end'].strftime('%Y-%m-%d')}"
        )

        print(
            f"验证期："
            f"{w['validation_start'].strftime('%Y-%m-%d')}"
            f" → "
            f"{w['validation_end'].strftime('%Y-%m-%d')}"
        )

        print(
            f"训练期选择："
            f"{w['step'] * 100:.0f}%"
        )

        print(
            f"验证期收益："
            f"{w['return']:.2f}%"
        )

        print(
            f"验证期年化："
            f"{w['annual']:.2f}%"
        )

        print(
            f"验证期最大回撤："
            f"{w['drawdown']:.2f}%"
        )

        print(
            f"验证期Sharpe："
            f"{w['sharpe']:.2f}"
        )

        print(
            f"平均资金使用率："
            f"{w['capital_usage']:.2f}%"
        )


# ============================================================
# 汇总
# ============================================================

def summarize(
    windows
):

    if len(windows) == 0:

        return {

            "windows": 0,

            "win_rate": 0,

            "average_return": 0,

            "average_annual": 0,

            "average_drawdown": 0,

            "average_sharpe": 0,

            "average_capital_usage": 0,

            "most_common_step": 0,

            "parameter_stability": 0

        }

    df = pd.DataFrame(
        windows
    )

    positive = (
        df["return"] > 0
    ).sum()

    win_rate = (
        positive
        /
        len(df)
        *
        100
    )

    most_common_step = (
        df["step"]
        .value_counts()
        .index[0]
    )

    parameter_stability = (
        (
            df["step"]
            ==
            most_common_step
        ).mean()
        *
        100
    )

    return {

        "windows":
            len(df),

        "win_rate":
            win_rate,

        "average_return":
            df["return"].mean(),

        "average_annual":
            df["annual"].mean(),

        "average_drawdown":
            df["drawdown"].mean(),

        "average_sharpe":
            df["sharpe"].mean(),

        "average_capital_usage":
            df["capital_usage"].mean(),

        "most_common_step":
            most_common_step,

        "parameter_stability":
            parameter_stability

    }


# ============================================================
# 主程序
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "ETF V7.4.1：建仓信号严格样本外验证"
    )

    print("=" * 70)

    print()

    print(
        "目标：验证趋势过滤是否真的有效"
    )

    print()

    all_results = []

    for code, name in ETF_LIST.items():

        print()

        print("=" * 70)

        print(
            f"{code} {name}"
        )

        print("=" * 70)

        df = load_data(
            code
        )

        if df is None:

            continue

        if len(df) < 378:

            print()

            print(
                "数据不足，无法形成"
                "至少一个完整训练+验证窗口"
            )

            continue

        # ====================================================
        # 三种策略
        # ====================================================

        modes = [

            (
                "A 原始策略",
                "original"
            ),

            (
                "B 趋势过滤",
                "trend"
            ),

            (
                "C Signal + 趋势过滤",
                "signal_trend"
            )

        ]

        mode_summaries = {}

        for mode_name, mode in modes:

            windows = rolling_validation(
                df,
                mode
            )

            print_windows(
                windows,
                mode_name
            )

            summary = summarize(
                windows
            )

            mode_summaries[
                mode
            ] = summary

            print()

            print(
                f"{mode_name}汇总："
            )

            print(
                f"验证窗口："
                f"{summary['windows']}"
            )

            print(
                f"胜率："
                f"{summary['win_rate']:.2f}%"
            )

            print(
                f"平均收益："
                f"{summary['average_return']:.2f}%"
            )

            print(
                f"平均年化："
                f"{summary['average_annual']:.2f}%"
            )

            print(
                f"平均最大回撤："
                f"{summary['average_drawdown']:.2f}%"
            )

            print(
                f"平均Sharpe："
                f"{summary['average_sharpe']:.2f}"
            )

            print(
                f"平均资金使用率："
                f"{summary['average_capital_usage']:.2f}%"
            )

            print(
                f"最常出现参数："
                f"{summary['most_common_step'] * 100:.0f}%"
            )

            print(
                f"参数稳定性："
                f"{summary['parameter_stability']:.2f}%"
            )

            all_results.append(
                {
                    "code":
                        code,

                    "name":
                        name,

                    "mode":
                        mode_name,

                    "windows":
                        summary["windows"],

                    "win_rate":
                        summary["win_rate"],

                    "average_return":
                        summary["average_return"],

                    "average_annual":
                        summary["average_annual"],

                    "average_drawdown":
                        summary["average_drawdown"],

                    "average_sharpe":
                        summary["average_sharpe"],

                    "average_capital_usage":
                        summary[
                            "average_capital_usage"
                        ],

                    "most_common_step":
                        summary[
                            "most_common_step"
                        ],

                    "parameter_stability":
                        summary[
                            "parameter_stability"
                        ]
                }
            )

    # ========================================================
    # 最终比较
    # ========================================================

    print()

    print("=" * 70)

    print(
        "V7.4.1 建仓过滤器最终比较"
    )

    print("=" * 70)

    if len(all_results) == 0:

        print(
            "没有可用结果"
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 主ETF重点结论
    # ========================================================

    main_results = results_df[
        results_df["code"]
        == MAIN_CODE
        if "MAIN_CODE" in globals()
        else results_df["code"]
        == "159581"
    ]

    print()

    print("=" * 70)

    print(
        "V7.4.1核心结论"
    )

    print("=" * 70)

    print()

    print(
        "注意："
    )

    print(
        "只有当趋势过滤在严格样本外"
    )

    print(
        "同时改善收益、Sharpe或回撤时，"
    )

    print(
        "才允许加入V7.4实盘规则。"
    )

    print()

    print(
        "本程序不会自动修改V7.4实盘策略。"
    )

    print(
        "它只负责验证。"
    )

    print()

    print(
        "V7.4.1完成"
    )


if __name__ == "__main__":

    main()
