import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# ETF V7.1
#
# 重要：
# V7.1 不再访问 Yahoo Finance
#
# 直接读取 update_data.py 已经生成的：
#
# data/etf_159209-signals.csv
# data/etf_159399-signals.csv
# data/etf_159581-signals.csv
#
# 目的：
# 验证 V6 找出的分批加仓参数是否具有样本外稳定性。
#
# 测试参数：
# 2%、3%、4%、5%、6%、8%、10%
#
# 总资金：100000
# 每档：20000
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000
TRANCHE = 20000
FEE_RATE = 0.0005

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
        "file": "etf_159209-signals.csv"
    },
    {
        "code": "159399",
        "name": "现金流ETF",
        "file": "etf_159399-signals.csv"
    },
    {
        "code": "159581",
        "name": "红利ETF",
        "file": "etf_159581-signals.csv"
    }
]


# ============================================================
# 读取本地数据
# ============================================================

def load_local_data(filename):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    print(
        f"读取本地数据：{path}"
    )

    if not os.path.exists(path):

        print(
            f"文件不存在：{path}"
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
            + str(list(df.columns))
        )

        # ----------------------------------------------------
        # 自动寻找日期列
        # ----------------------------------------------------

        date_candidates = [
            "date",
            "Date",
            "日期",
            "datetime",
            "Datetime",
            "时间"
        ]

        date_col = None

        for col in date_candidates:

            if col in df.columns:

                date_col = col
                break

        if date_col is not None:

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
        # 自动寻找收盘价
        # ----------------------------------------------------

        close_candidates = [
            "close",
            "Close",
            "收盘",
            "收盘价",
            "price",
            "Price"
        ]

        close_col = None

        for col in close_candidates:

            if col in df.columns:

                close_col = col
                break

        # ----------------------------------------------------
        # 如果没有标准close字段，
        # 尝试寻找包含close的字段
        # ----------------------------------------------------

        if close_col is None:

            for col in df.columns:

                col_lower = str(
                    col
                ).lower()

                if (
                    "close" in col_lower
                    or "收盘" in str(col)
                ):

                    close_col = col
                    break

        if close_col is None:

            print(
                "找不到收盘价字段"
            )

            print(
                "当前字段："
                + str(list(df.columns))
            )

            return None

        # ----------------------------------------------------
        # 标准化Close
        # ----------------------------------------------------

        df["Close"] = pd.to_numeric(
            df[close_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Close"]
        )

        df = df[
            df["Close"] > 0
        ]

        # ----------------------------------------------------
        # 删除重复日期
        # ----------------------------------------------------

        if df.index.duplicated().any():

            df = df[
                ~df.index.duplicated(
                    keep="last"
                )
            ]

        print(
            f"成功读取 {len(df)} 条数据"
        )

        if len(df) > 0:

            print(
                f"数据范围："
                f"{df.index[0]} "
                f"→ "
                f"{df.index[-1]}"
            )

        return df

    except Exception as e:

        print(
            f"读取数据失败：{e}"
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
# 第一次：20,000
#
# 之后：
#
# 2%：
# -2%
# -4%
# -6%
# -8%
#
# 3%：
# -3%
# -6%
# -9%
# -12%
#
# 以此类推。
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

    invested_curve = []

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

        equity_value = (
            cash
            + shares * price
        )

        invested_value = (
            shares * price
        )

        equity_curve.append(
            equity_value
        )

        invested_curve.append(
            invested_value
        )

    equity = pd.Series(
        equity_curve,
        index=df.index
    )

    invested = pd.Series(
        invested_curve,
        index=df.index
    )

    return (
        equity,
        invested,
        entry_count
    )


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

        "total_return":
            total_return * 100,

        "annual_return":
            annual_return * 100,

        "max_drawdown":
            max_drawdown * 100,

        "sharpe":
            sharpe
    }


# ============================================================
# 策略评分
# ============================================================

def strategy_score(metrics):

    annual = metrics[
        "annual_return"
    ]

    sharpe = metrics[
        "sharpe"
    ]

    drawdown = abs(
        metrics[
            "max_drawdown"
        ]
    )

    score = (
        annual * 0.50
        + sharpe * 10 * 0.30
        - drawdown * 0.20
    )

    return score


# ============================================================
# 训练期选择参数
# ============================================================

def find_best_parameter(
    train_df
):

    candidates = []

    for level in DRAWDOWN_LEVELS:

        equity, invested, entries = (
            backtest_tranche(
                train_df,
                level
            )
        )

        metrics = calculate_metrics(
            equity
        )

        score = strategy_score(
            metrics
        )

        candidates.append({

            "level":
                level,

            "return":
                metrics[
                    "total_return"
                ],

            "annual":
                metrics[
                    "annual_return"
                ],

            "drawdown":
                metrics[
                    "max_drawdown"
                ],

            "sharpe":
                metrics[
                    "sharpe"
                ],

            "score":
                score,

            "entries":
                entries
        })

    candidates_df = pd.DataFrame(
        candidates
    )

    best = candidates_df.sort_values(
        "score",
        ascending=False
    ).iloc[0]

    return (
        best,
        candidates_df
    )


# ============================================================
# 验证期
# ============================================================

def evaluate_validation(
    validation_df,
    level
):

    equity, invested, entries = (
        backtest_tranche(
            validation_df,
            level
        )
    )

    metrics = calculate_metrics(
        equity
    )

    average_invested = (
        invested.mean()
    )

    capital_utilization = (
        average_invested
        / INITIAL_CAPITAL
        * 100
    )

    return {

        "return":
            metrics[
                "total_return"
            ],

        "annual":
            metrics[
                "annual_return"
            ],

        "drawdown":
            metrics[
                "max_drawdown"
            ],

        "sharpe":
            metrics[
                "sharpe"
            ],

        "entries":
            entries,

        "capital_utilization":
            capital_utilization
    }


# ============================================================
# Walk Forward
#
# 每次：
#
# 训练252个交易日
# 验证126个交易日
#
# 然后向前滚动126天。
#
# 参数只允许由训练期决定。
# ============================================================

def walk_forward_validation(
    df
):

    df = df.copy()

    n = len(df)

    if n < 500:

        return None

    train_size = 252

    validation_size = 126

    results = []

    start = 0

    window_id = 1

    while (
        start
        + train_size
        + validation_size
        <= n
    ):

        train_df = df.iloc[
            start:
            start + train_size
        ]

        validation_df = df.iloc[
            start + train_size:
            start + train_size
            + validation_size
        ]

        # ====================================================
        # 训练期选择参数
        # ====================================================

        best, candidates = (
            find_best_parameter(
                train_df
            )
        )

        best_level = float(
            best["level"]
        )

        # ====================================================
        # 验证期使用锁定参数
        # ====================================================

        validation = (
            evaluate_validation(
                validation_df,
                best_level
            )
        )

        result = {

            "window":
                window_id,

            "train_start":
                str(
                    train_df.index[
                        0
                    ]
                ),

            "train_end":
                str(
                    train_df.index[
                        -1
                    ]
                ),

            "validation_start":
                str(
                    validation_df.index[
                        0
                    ]
                ),

            "validation_end":
                str(
                    validation_df.index[
                        -1
                    ]
                ),

            "selected_step":
                f"{best_level * 100:.0f}%",

            "validation_return":
                validation[
                    "return"
                ],

            "validation_annual":
                validation[
                    "annual"
                ],

            "validation_drawdown":
                validation[
                    "drawdown"
                ],

            "validation_sharpe":
                validation[
                    "sharpe"
                ],

            "validation_entries":
                validation[
                    "entries"
                ],

            "capital_utilization":
                validation[
                    "capital_utilization"
                ]
        }

        results.append(
            result
        )

        print()

        print(
            f"窗口 {window_id}"
        )

        print(
            f"训练期："
            f"{train_df.index[0]}"
            f" → "
            f"{train_df.index[-1]}"
        )

        print(
            f"验证期："
            f"{validation_df.index[0]}"
            f" → "
            f"{validation_df.index[-1]}"
        )

        print(
            "训练期选择："
            f"{best_level * 100:.0f}%"
        )

        print(
            "验证期收益："
            f"{validation['return']:.2f}%"
        )

        print(
            "验证期年化："
            f"{validation['annual']:.2f}%"
        )

        print(
            "验证期最大回撤："
            f"{validation['drawdown']:.2f}%"
        )

        print(
            "验证期Sharpe："
            f"{validation['sharpe']:.2f}"
        )

        print(
            "平均资金使用率："
            f"{validation['capital_utilization']:.2f}%"
        )

        start += validation_size

        window_id += 1

    return pd.DataFrame(
        results
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print(
        "ETF V7.1：本地数据滚动验证模型"
    )

    print("=" * 70)

    all_results = []

    summary_results = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        filename = item["file"]

        print()

        print("=" * 70)

        print(
            f"{code} {name}"
        )

        print("=" * 70)

        df = load_local_data(
            filename
        )

        if df is None:

            print(
                "数据读取失败，跳过"
            )

            continue

        result_df = (
            walk_forward_validation(
                df
            )
        )

        if (
            result_df is None
            or result_df.empty
        ):

            print(
                "数据不足，无法验证"
            )

            continue

        result_df[
            "code"
        ] = code

        result_df[
            "name"
        ] = name

        all_results.append(
            result_df
        )

        # ====================================================
        # 汇总
        # ====================================================

        average_return = (
            result_df[
                "validation_return"
            ].mean()
        )

        average_annual = (
            result_df[
                "validation_annual"
            ].mean()
        )

        average_drawdown = (
            result_df[
                "validation_drawdown"
            ].mean()
        )

        average_sharpe = (
            result_df[
                "validation_sharpe"
            ].mean()
        )

        average_utilization = (
            result_df[
                "capital_utilization"
            ].mean()
        )

        positive_windows = (
            (
                result_df[
                    "validation_return"
                ] > 0
            ).sum()
        )

        total_windows = len(
            result_df
        )

        win_rate = (
            positive_windows
            / total_windows
            * 100
        )

        parameter_counts = (
            result_df[
                "selected_step"
            ]
            .value_counts()
        )

        most_common_parameter = (
            parameter_counts
            .index[0]
        )

        parameter_stability = (
            parameter_counts.iloc[0]
            / total_windows
            * 100
        )

        summary_results.append({

            "code":
                code,

            "name":
                name,

            "windows":
                total_windows,

            "positive_windows":
                positive_windows,

            "win_rate":
                round(
                    win_rate,
                    2
                ),

            "average_return":
                round(
                    average_return,
                    2
                ),

            "average_annual":
                round(
                    average_annual,
                    2
                ),

            "average_drawdown":
                round(
                    average_drawdown,
                    2
                ),

            "average_sharpe":
                round(
                    average_sharpe,
                    2
                ),

            "average_capital_utilization":
                round(
                    average_utilization,
                    2
                ),

            "most_common_step":
                most_common_parameter,

            "parameter_stability":
                round(
                    parameter_stability,
                    2
                )
        })

        print()

        print(
            "V7.1汇总："
        )

        print(
            f"验证窗口："
            f"{total_windows}"
        )

        print(
            f"正收益窗口："
            f"{positive_windows}"
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
            f"{most_common_parameter}"
        )

        print(
            f"参数稳定性："
            f"{parameter_stability:.2f}%"
        )

    # ========================================================
    # 保存全部窗口
    # ========================================================

    if all_results:

        all_df = pd.concat(
            all_results,
            ignore_index=True
        )

        all_df.to_csv(
            os.path.join(
                DATA_DIR,
                "etf_v7_walk_forward.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 保存汇总
    # ========================================================

    summary_df = pd.DataFrame(
        summary_results
    )

    summary_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_v7_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print("=" * 70)

    print(
        "V7.1滚动验证完成"
    )

    print("=" * 70)

    print()

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
        "data/etf_v7_walk_forward.csv"
    )

    print(
        "data/etf_v7_summary.csv"
    )


if __name__ == "__main__":

    main()
