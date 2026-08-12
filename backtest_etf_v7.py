import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# ETF V7.2
# 上市以来完整回测 + 样本外滚动验证
#
# 数据来源：
# data/etf_159209_signals.csv
# data/etf_159399_signals.csv
# data/etf_159581_signals.csv
#
# 重要：
# 1. 不人为补充上市以前的数据
# 2. 每只ETF按照自己的真实历史长度计算
# 3. 历史不足时自动减少验证窗口
# 4. 训练期选择参数
# 5. 验证期锁定参数，不重新优化
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
# ETF列表
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
# 读取本地数据
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

        print(
            "文件不存在"
        )

        return None

    try:

        df = pd.read_csv(
            path
        )

        if df.empty:

            print(
                "CSV为空"
            )

            return None

        print(
            "原始字段："
            + str(
                list(df.columns)
            )
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

            print(
                "找不到日期字段"
            )

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

            print(
                "找不到价格字段"
            )

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

        # ----------------------------------------------------
        # 去重
        # ----------------------------------------------------

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
# 总资金100000
#
# 20000第一笔
#
# 后面每跌一个指定百分比加20000
#
# 最多5笔
# ============================================================

def tranche_strategy(
    df,
    step
):

    prices = df["Close"]

    cash = INITIAL_CAPITAL

    shares = 0

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

        # ----------------------------------------------------
        # 每日资产
        # ----------------------------------------------------

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
# 策略评分
#
# 年化收益
# Sharpe
# 最大回撤
# ============================================================

def score(m):

    return (
        m["annual"] * 0.5
        + m["sharpe"] * 10 * 0.3
        - abs(m["drawdown"]) * 0.2
    )


# ============================================================
# 找训练期最佳参数
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

            "step": step,

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
# 动态确定训练/验证窗口
#
# 不再强制252+126
#
# 原则：
#
# 数据 >= 500：
# 训练252 + 验证126
#
# 数据 >= 360：
# 训练约65% + 验证约35%
#
# 数据 < 360：
# 尽可能70%训练 + 30%验证
# ============================================================

def get_window_size(n):

    if n >= 500:

        train_size = 252

        validation_size = 126

    elif n >= 360:

        validation_size = int(
            n * 0.30
        )

        train_size = (
            n
            - validation_size
        )

    else:

        validation_size = int(
            n * 0.30
        )

        train_size = (
            n
            - validation_size
        )

    return (
        train_size,
        validation_size
    )


# ============================================================
# Walk Forward
# ============================================================

def walk_forward(df):

    n = len(df)

    if n < 250:

        print(
            "历史数据少于250个交易日"
        )

        return None

    train_size, validation_size = (
        get_window_size(n)
    )

    results = []

    start = 0

    window = 1

    # --------------------------------------------------------
    # 为避免只有一个窗口时重复使用大量数据，
    # 这里采用滚动验证。
    # --------------------------------------------------------

    while True:

        train_start = start

        train_end = (
            start
            + train_size
        )

        validation_end = (
            train_end
            + validation_size
        )

        if validation_end > n:

            break

        train = df.iloc[
            train_start:
            train_end
        ]

        validation = df.iloc[
            train_end:
            validation_end
        ]

        # ----------------------------------------------------
        # 训练期优化
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
                window,

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
            f"窗口 {window}"
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

        # ----------------------------------------------------
        # 滚动
        # ----------------------------------------------------

        if n >= 500:

            step_forward = validation_size

        else:

            # 对短历史ETF只做一次主要样本外验证
            step_forward = validation_size

        start += step_forward

        window += 1

    return pd.DataFrame(
        results
    )


# ============================================================
# 完整上市以来回测
# ============================================================

def full_backtest(df):

    result = {}

    # --------------------------------------------------------
    # 买入持有
    # --------------------------------------------------------

    bh_equity = buy_hold(
        df
    )

    bh = metrics(
        bh_equity
    )

    result["buy_hold"] = bh

    # --------------------------------------------------------
    # 所有分批参数
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

    result["best_step"] = (
        best["step"]
    )

    result["best_return"] = (
        best["return"]
    )

    result["best_annual"] = (
        best["annual"]
    )

    result["best_drawdown"] = (
        best["drawdown"]
    )

    result["best_sharpe"] = (
        best["sharpe"]
    )

    result["best_score"] = (
        best["score"]
    )

    result["best_entries"] = (
        best["entries"]
    )

    result["best_utilization"] = (
        best["utilization"]
    )

    result["all_candidates"] = (
        candidate_df
    )

    return result


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print(
        "ETF V7.2：上市以来完整回测 + 样本外验证"
    )

    print("=" * 70)

    all_windows = []

    summary = []

    full_results = []

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
        # 完整历史回测
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
            f"V6/V7最优加仓间距："
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

        # ====================================================
        # 保存完整参数比较
        # ====================================================

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
        # Walk Forward
        # ====================================================

        print()

        print(
            "【样本外滚动验证】"
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

        most_common = (
            counts.index[0]
        )

        stability = (
            counts.iloc[0]
            / windows
            * 100
        )

        print()

        print(
            "V7.2汇总："
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
            f"{most_common}"
        )

        print(
            f"参数稳定性："
            f"{stability:.2f}%"
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
                most_common,

            "parameter_stability":
                round(
                    stability,
                    2
                )
        })

    # ========================================================
    # 保存结果
    # ========================================================

    if all_windows:

        windows_df = pd.concat(
            all_windows,
            ignore_index=True
        )

        windows_df.to_csv(
            os.path.join(
                DATA_DIR,
                "etf_v7_2_walk_forward.csv"
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
                "etf_v7_2_full_backtest.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v7_2_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 最终结果
    # ========================================================

    print()

    print("=" * 70)

    print(
        "V7.2完成"
    )

    print("=" * 70)

    print()

    if not summary_df.empty:

        print(
            summary_df.to_string(
                index=False
            )
        )

    print()

    print(
        "生成文件："
    )

    print(
        "data/etf_v7_2_summary.csv"
    )

    print(
        "data/etf_v7_2_full_backtest.csv"
    )

    print(
        "data/etf_v7_2_walk_forward.csv"
    )


if __name__ == "__main__":

    main()
