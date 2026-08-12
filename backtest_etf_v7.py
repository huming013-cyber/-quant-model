import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# ETF V7.3
# 上市以来完整回测 + 严格滚动样本外验证
#
# 核心原则：
# 1. 只使用ETF真实存在的历史数据
# 2. 不补上市以前的数据
# 3. 不使用验证期数据选择参数
# 4. 每个验证窗口的参数只由训练期决定
# 5. 最后一个验证窗口必须覆盖最新数据
# 6. 同时进行完整历史回测
# ============================================================


DATA_DIR = "data"

INITIAL_CAPITAL = 100000

TRANCHE = 20000

FEE_RATE = 0.0005


# ============================================================
# 测试的加仓间距
# ============================================================

DRAWDOWN_LEVELS = [
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10
]


# ============================================================
# ETF
# ============================================================

ETF_LIST = [

    {
        "code": "159209",
        "name": "红利质量ETF",
        "file": "etf_159209_signals.csv"
    },

    {
        "code": "159399",
        "name": "现金流ETF",
        "file": "etf_159399_signals.csv"
    },

    {
        "code": "159581",
        "name": "红利ETF",
        "file": "etf_159581_signals.csv"
    }

]


# ============================================================
# 读取数据
# ============================================================

def load_data(filename):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    print()
    print(
        f"读取本地数据：{path}"
    )

    if not os.path.exists(path):

        print("文件不存在")

        return None

    try:

        df = pd.read_csv(path)

        if df.empty:

            print("CSV为空")

            return None

        print(
            "原始字段："
            + str(list(df.columns))
        )

        # ----------------------------------------------------
        # 日期
        # ----------------------------------------------------

        date_col = None

        for col in [
            "date",
            "Date",
            "日期",
            "datetime",
            "Datetime",
            "时间"
        ]:

            if col in df.columns:

                date_col = col

                break

        if date_col is None:

            print("找不到日期字段")

            return None

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=[date_col]
        )

        df = df.sort_values(
            date_col
        )

        df = df.set_index(
            date_col
        )

        # ----------------------------------------------------
        # 价格
        # ----------------------------------------------------

        price_col = None

        for col in [
            "price",
            "Price",
            "close",
            "Close",
            "收盘",
            "收盘价"
        ]:

            if col in df.columns:

                price_col = col

                break

        if price_col is None:

            print("找不到价格字段")

            return None

        df["Close"] = pd.to_numeric(
            df[price_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Close"]
        )

        df = df[
            df["Close"] > 0
        ]

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

        print(
            f"成功读取 {len(df)} 条数据"
        )

        print(
            f"数据范围："
            f"{df.index[0].date()}"
            f" → "
            f"{df.index[-1].date()}"
        )

        return df

    except Exception as e:

        print(
            f"数据读取失败：{e}"
        )

        return None


# ============================================================
# 买入持有
# ============================================================

def buy_hold(df):

    prices = df["Close"]

    start_price = prices.iloc[0]

    equity = (
        INITIAL_CAPITAL
        * prices
        / start_price
    )

    return equity


# ============================================================
# 分批建仓
#
# 100000元
#
# 第一笔20000
#
# 每达到一个回撤级别加20000
#
# 最多5笔
# ============================================================

def tranche_strategy(
    df,
    step
):

    prices = df["Close"]

    cash = INITIAL_CAPITAL

    shares = 0.0

    first_price = None

    entries = 0

    equity_list = []

    invested_list = []

    for price in prices:

        price = float(price)

        # ----------------------------------------------------
        # 第一笔
        # ----------------------------------------------------

        if entries == 0:

            amount = TRANCHE

            fee = (
                amount
                * FEE_RATE
            )

            if cash >= amount + fee:

                shares += (
                    amount
                    / price
                )

                cash -= (
                    amount
                    + fee
                )

                first_price = price

                entries = 1

        # ----------------------------------------------------
        # 后续加仓
        # ----------------------------------------------------

        elif entries < 5:

            drawdown = (
                price
                / first_price
                - 1
            )

            target = (
                -step
                * entries
            )

            if drawdown <= target:

                amount = TRANCHE

                fee = (
                    amount
                    * FEE_RATE
                )

                if cash >= amount + fee:

                    shares += (
                        amount
                        / price
                    )

                    cash -= (
                        amount
                        + fee
                    )

                    entries += 1

        equity = (
            cash
            + shares * price
        )

        invested = (
            shares * price
        )

        equity_list.append(
            equity
        )

        invested_list.append(
            invested
        )

    equity = pd.Series(
        equity_list,
        index=prices.index
    )

    invested = pd.Series(
        invested_list,
        index=prices.index
    )

    return (
        equity,
        invested,
        entries
    )


# ============================================================
# 指标
# ============================================================

def metrics(equity):

    equity = equity.dropna()

    if len(equity) < 2:

        return {
            "return": 0,
            "annual": 0,
            "drawdown": 0,
            "sharpe": 0
        }

    total_return = (
        equity.iloc[-1]
        / equity.iloc[0]
        - 1
    )

    days = len(equity)

    years = days / 252

    if years <= 0:

        years = 1

    annual_return = (
        (1 + total_return)
        ** (1 / years)
        - 1
    )

    peak = equity.cummax()

    drawdown = (
        equity
        / peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    daily_return = (
        equity
        .pct_change()
        .dropna()
    )

    if (
        len(daily_return) > 1
        and daily_return.std() > 0
    ):

        sharpe = (
            daily_return.mean()
            / daily_return.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    return {

        "return":
            total_return * 100,

        "annual":
            annual_return * 100,

        "drawdown":
            max_drawdown * 100,

        "sharpe":
            sharpe

    }


# ============================================================
# 参数评分
#
# 注意：
# 只用于训练期
#
# 验证期绝对不参与参数选择
# ============================================================

def score(m):

    return (
        m["annual"] * 0.50
        + m["sharpe"] * 10 * 0.30
        - abs(m["drawdown"]) * 0.20
    )


# ============================================================
# 训练期寻找最佳参数
# ============================================================

def optimize(train_df):

    rows = []

    for step in DRAWDOWN_LEVELS:

        equity, invested, entries = (
            tranche_strategy(
                train_df,
                step
            )
        )

        m = metrics(
            equity
        )

        rows.append({

            "step":
                step,

            "return":
                m["return"],

            "annual":
                m["annual"],

            "drawdown":
                m["drawdown"],

            "sharpe":
                m["sharpe"],

            "score":
                score(m)

        })

    result = pd.DataFrame(
        rows
    )

    best = result.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    return (
        best,
        result
    )


# ============================================================
# 创建严格滚动验证窗口
#
# 设计：
#
# 数据 >= 500：
# 训练252交易日
# 验证126交易日
#
# 数据 < 500：
# 训练约70%
# 验证约30%
#
# 最关键：
#
# 如果最后剩余数据不足一个完整验证窗口，
# 则把最后一个验证窗口右对齐到最新日期。
#
# 这样可以保证最新数据一定进入样本外验证。
# ============================================================

def create_windows(n):

    windows = []

    if n < 250:

        return windows

    if n >= 500:

        train_size = 252

        validation_size = 126

        # ----------------------------------------------------
        # 正常滚动
        # ----------------------------------------------------

        start = 0

        while (
            start
            + train_size
            + validation_size
            <= n
        ):

            train_start = start

            train_end = (
                start
                + train_size
            )

            validation_end = (
                train_end
                + validation_size
            )

            windows.append(
                (
                    train_start,
                    train_end,
                    validation_end
                )
            )

            start += validation_size

        # ----------------------------------------------------
        # 最后窗口右对齐
        #
        # 如果最后一段没有完整验证窗口，
        # 就让验证期覆盖到最后一天。
        # ----------------------------------------------------

        final_validation_end = n

        final_validation_start = (
            n
            - validation_size
        )

        final_train_end = (
            final_validation_start
        )

        final_train_start = (
            final_train_end
            - train_size
        )

        if final_train_start >= 0:

            final_window = (
                final_train_start,
                final_train_end,
                final_validation_end
            )

            if (
                not windows
                or windows[-1]
                != final_window
            ):

                windows.append(
                    final_window
                )

    else:

        # ----------------------------------------------------
        # 历史较短ETF
        #
        # 一个主要样本外窗口，
        # 训练70%，验证30%
        # ----------------------------------------------------

        validation_size = int(
            n * 0.30
        )

        train_size = (
            n
            - validation_size
        )

        windows.append(
            (
                0,
                train_size,
                n
            )
        )

    return windows


# ============================================================
# Walk Forward
# ============================================================

def walk_forward(df):

    n = len(df)

    windows = create_windows(
        n
    )

    if not windows:

        print(
            "历史数据不足，无法验证"
        )

        return None

    results = []

    for i, (
        train_start,
        train_end,
        validation_end
    ) in enumerate(
        windows,
        start=1
    ):

        train = df.iloc[
            train_start:
            train_end
        ]

        validation = df.iloc[
            train_end:
            validation_end
        ]

        if len(validation) < 2:

            continue

        # ----------------------------------------------------
        # 参数只从训练期选择
        # ----------------------------------------------------

        best, candidates = optimize(
            train
        )

        selected_step = float(
            best["step"]
        )

        # ----------------------------------------------------
        # 样本外验证
        # ----------------------------------------------------

        equity, invested, entries = (
            tranche_strategy(
                validation,
                selected_step
            )
        )

        m = metrics(
            equity
        )

        utilization = (
            invested.mean()
            / INITIAL_CAPITAL
            * 100
        )

        result = {

            "window":
                i,

            "train_start":
                train.index[0].strftime(
                    "%Y-%m-%d"
                ),

            "train_end":
                train.index[-1].strftime(
                    "%Y-%m-%d"
                ),

            "validation_start":
                validation.index[0].strftime(
                    "%Y-%m-%d"
                ),

            "validation_end":
                validation.index[-1].strftime(
                    "%Y-%m-%d"
                ),

            "selected_step":
                f"{selected_step * 100:.0f}%",

            "validation_return":
                m["return"],

            "validation_annual":
                m["annual"],

            "validation_drawdown":
                m["drawdown"],

            "validation_sharpe":
                m["sharpe"],

            "capital_utilization":
                utilization,

            "entries":
                entries
        }

        results.append(
            result
        )

        print()

        print(
            f"窗口 {i}"
        )

        print(
            f"训练期："
            f"{train.index[0].date()}"
            f" → "
            f"{train.index[-1].date()}"
        )

        print(
            f"验证期："
            f"{validation.index[0].date()}"
            f" → "
            f"{validation.index[-1].date()}"
        )

        print(
            f"训练期选择："
            f"{selected_step * 100:.0f}%"
        )

        print(
            f"验证期收益："
            f"{m['return']:.2f}%"
        )

        print(
            f"验证期年化："
            f"{m['annual']:.2f}%"
        )

        print(
            f"验证期最大回撤："
            f"{m['drawdown']:.2f}%"
        )

        print(
            f"验证期Sharpe："
            f"{m['sharpe']:.2f}"
        )

        print(
            f"平均资金使用率："
            f"{utilization:.2f}%"
        )

    if not results:

        return None

    return pd.DataFrame(
        results
    )


# ============================================================
# 完整上市以来回测
# ============================================================

def full_backtest(df):

    # --------------------------------------------------------
    # 买入持有
    # --------------------------------------------------------

    bh_equity = buy_hold(
        df
    )

    bh = metrics(
        bh_equity
    )

    # --------------------------------------------------------
    # 分批参数
    # --------------------------------------------------------

    candidates = []

    for step in DRAWDOWN_LEVELS:

        equity, invested, entries = (
            tranche_strategy(
                df,
                step
            )
        )

        m = metrics(
            equity
        )

        candidates.append({

            "step":
                step,

            "return":
                m["return"],

            "annual":
                m["annual"],

            "drawdown":
                m["drawdown"],

            "sharpe":
                m["sharpe"],

            "score":
                score(m),

            "entries":
                entries,

            "utilization":
                invested.mean()
                / INITIAL_CAPITAL
                * 100
        })

    candidate_df = pd.DataFrame(
        candidates
    )

    best = candidate_df.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    return {

        "buy_hold":
            bh,

        "best_step":
            best["step"],

        "best_return":
            best["return"],

        "best_annual":
            best["annual"],

        "best_drawdown":
            best["drawdown"],

        "best_sharpe":
            best["sharpe"],

        "best_score":
            best["score"],

        "best_entries":
            best["entries"],

        "best_utilization":
            best["utilization"],

        "all_candidates":
            candidate_df

    }


# ============================================================
# 综合判断
#
# 不是投资建议。
# 这里只用于量化模型排名。
# ============================================================

def final_model_score(row):

    validation_score = (
        row["validation_win_rate"] * 0.25
        + row["validation_average_sharpe"] * 20 * 0.25
        + row["parameter_stability"] * 0.10
        - abs(
            row["validation_average_drawdown"]
        ) * 0.10
    )

    historical_score = (
        row["full_strategy_annual"] * 0.15
        + row["full_strategy_sharpe"] * 10 * 0.10
        - abs(
            row["full_strategy_drawdown"]
        ) * 0.05
    )

    return (
        validation_score
        + historical_score
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print(
        "ETF V7.3：上市以来完整回测 + 严格滚动样本外验证"
    )

    print("=" * 70)

    all_windows = []

    full_results = []

    summary = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        print()

        print("=" * 70)

        print(
            f"{code} {name}"
        )

        print("=" * 70)

        df = load_data(
            item["file"]
        )

        if df is None:

            print(
                "数据读取失败"
            )

            continue

        # ====================================================
        # 完整历史
        # ====================================================

        print()

        print(
            "【上市以来完整回测】"
        )

        full = full_backtest(
            df
        )

        bh = full[
            "buy_hold"
        ]

        print()

        print(
            f"买入持有收益："
            f"{bh['return']:.2f}%"
        )

        print(
            f"买入持有年化："
            f"{bh['annual']:.2f}%"
        )

        print(
            f"买入持有最大回撤："
            f"{bh['drawdown']:.2f}%"
        )

        print(
            f"买入持有Sharpe："
            f"{bh['sharpe']:.2f}"
        )

        print()

        print(
            f"完整历史最优加仓间距："
            f"{full['best_step'] * 100:.0f}%"
        )

        print(
            f"完整历史策略收益："
            f"{full['best_return']:.2f}%"
        )

        print(
            f"完整历史策略年化："
            f"{full['best_annual']:.2f}%"
        )

        print(
            f"完整历史策略最大回撤："
            f"{full['best_drawdown']:.2f}%"
        )

        print(
            f"完整历史策略Sharpe："
            f"{full['best_sharpe']:.2f}"
        )

        print(
            f"平均资金使用率："
            f"{full['best_utilization']:.2f}%"
        )

        # ----------------------------------------------------
        # 保存参数比较
        # ----------------------------------------------------

        candidate_df = (
            full[
                "all_candidates"
            ].copy()
        )

        candidate_df[
            "code"
        ] = code

        candidate_df[
            "name"
        ] = name

        full_results.append(
            candidate_df
        )

        # ====================================================
        # 样本外
        # ====================================================

        print()

        print(
            "【严格滚动样本外验证】"
        )

        wf = walk_forward(
            df
        )

        if (
            wf is None
            or wf.empty
        ):

            print(
                "无法进行样本外验证"
            )

            continue

        wf[
            "code"
        ] = code

        wf[
            "name"
        ] = name

        all_windows.append(
            wf
        )

        # ====================================================
        # 汇总
        # ====================================================

        windows = len(
            wf
        )

        positive = (
            wf[
                "validation_return"
            ] > 0
        ).sum()

        win_rate = (
            positive
            / windows
            * 100
        )

        average_return = (
            wf[
                "validation_return"
            ].mean()
        )

        average_annual = (
            wf[
                "validation_annual"
            ].mean()
        )

        average_drawdown = (
            wf[
                "validation_drawdown"
            ].mean()
        )

        average_sharpe = (
            wf[
                "validation_sharpe"
            ].mean()
        )

        average_utilization = (
            wf[
                "capital_utilization"
            ].mean()
        )

        counts = (
            wf[
                "selected_step"
            ].value_counts()
        )

        most_common_step = (
            counts.index[0]
        )

        parameter_stability = (
            counts.iloc[0]
            / windows
            * 100
        )

        print()

        print(
            "V7.3汇总："
        )

        print(
            f"验证窗口："
            f"{windows}"
        )

        print(
            f"正收益窗口："
            f"{positive}"
        )

        print(
            f"胜率："
            f"{win_rate:.2f}%"
        )

        print(
            f"平均收益："
            f"{average_return:.2f}%"
        )

        print(
            f"平均年化："
            f"{average_annual:.2f}%"
        )

        print(
            f"平均最大回撤："
            f"{average_drawdown:.2f}%"
        )

        print(
            f"平均Sharpe："
            f"{average_sharpe:.2f}"
        )

        print(
            f"平均资金使用率："
            f"{average_utilization:.2f}%"
        )

        print(
            f"最常出现参数："
            f"{most_common_step}"
        )

        print(
            f"参数稳定性："
            f"{parameter_stability:.2f}%"
        )

        summary.append({

            "code":
                code,

            "name":
                name,

            "data_start":
                df.index[0].strftime(
                    "%Y-%m-%d"
                ),

            "data_end":
                df.index[-1].strftime(
                    "%Y-%m-%d"
                ),

            "data_days":
                len(df),

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

            "best_step":
                f"{full['best_step'] * 100:.0f}%",

            "full_strategy_return":
                round(
                    full["best_return"],
                    2
                ),

            "full_strategy_annual":
                round(
                    full["best_annual"],
                    2
                ),

            "full_strategy_drawdown":
                round(
                    full["best_drawdown"],
                    2
                ),

            "full_strategy_sharpe":
                round(
                    full["best_sharpe"],
                    2
                ),

            "validation_windows":
                windows,

            "validation_win_rate":
                round(
                    win_rate,
                    2
                ),

            "validation_average_return":
                round(
                    average_return,
                    2
                ),

            "validation_average_annual":
                round(
                    average_annual,
                    2
                ),

            "validation_average_drawdown":
                round(
                    average_drawdown,
                    2
                ),

            "validation_average_sharpe":
                round(
                    average_sharpe,
                    2
                ),

            "validation_capital_utilization":
                round(
                    average_utilization,
                    2
                ),

            "validation_most_common_step":
                most_common_step,

            "parameter_stability":
                round(
                    parameter_stability,
                    2
                )
        })

    # ========================================================
    # 生成汇总
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    if not summary_df.empty:

        summary_df[
            "model_score"
        ] = summary_df.apply(
            final_model_score,
            axis=1
        )

        summary_df = (
            summary_df
            .sort_values(
                "model_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        summary_df[
            "final_rank"
        ] = (
            summary_df.index
            + 1
        )

    # ========================================================
    # 保存
    # ========================================================

    if all_windows:

        windows_df = pd.concat(
            all_windows,
            ignore_index=True
        )

        windows_df.to_csv(
            os.path.join(
                DATA_DIR,
                "etf_v7_3_walk_forward.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    if full_results:

        full_df = pd.concat(
            full_results,
            ignore_index=True
        )

        full_df.to_csv(
            os.path.join(
                DATA_DIR,
                "etf_v7_3_full_backtest.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    summary_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v7_3_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 最终排名
    # ========================================================

    print()

    print("=" * 70)

    print(
        "V7.3最终排名"
    )

    print("=" * 70)

    if not summary_df.empty:

        print()

        print(
            summary_df[
                [
                    "final_rank",
                    "code",
                    "name",
                    "best_step",
                    "full_strategy_return",
                    "full_strategy_annual",
                    "full_strategy_drawdown",
                    "full_strategy_sharpe",
                    "validation_windows",
                    "validation_win_rate",
                    "validation_average_return",
                    "validation_average_annual",
                    "validation_average_drawdown",
                    "validation_average_sharpe",
                    "validation_most_common_step",
                    "parameter_stability",
                    "model_score"
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print("=" * 70)

    print(
        "V7.3完成"
    )

    print("=" * 70)

    print()

    print(
        "生成文件："
    )

    print(
        "data/etf_v7_3_summary.csv"
    )

    print(
        "data/etf_v7_3_full_backtest.csv"
    )

    print(
        "data/etf_v7_3_walk_forward.csv"
    )


if __name__ == "__main__":

    main()
