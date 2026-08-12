import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# ETF V4 专用量化回测
#
# 核心ETF：
# 159209 红利质量ETF
# 159399 现金流ETF
# 159581 红利ETF
#
# 初始资金：
# 100000元
#
# 分5次建仓：
# 20000
# 20000
# 20000
# 20000
# 20000
#
# 加仓逻辑：
# 第一次：趋势信号成立
# 第二次：从首次建仓价回撤3%
# 第三次：回撤6%
# 第四次：回撤9%
# 第五次：回撤12%
#
# 同时测试：
# 1. 买入持有
# 2. 分批建仓
# 3. ETF量化择时
# 4. 三ETF轮动
#
# 所有信号均使用下一交易日执行，避免未来函数。
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
# 下载数据
# ============================================================

def download_data(yahoo_code):

    print(f"正在获取：{yahoo_code}")

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

            print("❌ 没有数据")

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

            print("❌ 历史数据不足")

            return None

        return df

    except Exception as e:

        print(
            f"❌ 下载失败：{e}"
        )

        return None


# ============================================================
# 技术指标
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    # --------------------------------------------------------
    # 均线
    # --------------------------------------------------------

    df["MA20"] = close.rolling(20).mean()

    df["MA60"] = close.rolling(60).mean()

    df["MA120"] = close.rolling(120).mean()

    # --------------------------------------------------------
    # 动量
    # --------------------------------------------------------

    df["RET20"] = close.pct_change(20)

    df["RET60"] = close.pct_change(60)

    df["RET120"] = close.pct_change(120)

    # --------------------------------------------------------
    # 20日高点
    # --------------------------------------------------------

    df["HIGH20"] = close.rolling(20).max()

    df["HIGH60"] = close.rolling(60).max()

    df["HIGH120"] = close.rolling(120).max()

    # --------------------------------------------------------
    # 回撤
    # --------------------------------------------------------

    df["DD20"] = (
        close / df["HIGH20"] - 1
    )

    df["DD60"] = (
        close / df["HIGH60"] - 1
    )

    df["DD120"] = (
        close / df["HIGH120"] - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI14"] = (
        100
        - 100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        ema12 - ema26
    )

    df["SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        - df["SIGNAL"]
    )

    # --------------------------------------------------------
    # 波动率
    # --------------------------------------------------------

    daily_return = close.pct_change()

    df["VOLATILITY20"] = (
        daily_return
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    return df


# ============================================================
# ETF专用评分
#
# 总分100
#
# 趋势       35
# 动量       25
# 回撤       15
# RSI        10
# MACD       5
# 稳定性     10
# ============================================================

def calculate_score(row):

    score = 0

    price = row["Close"]

    # --------------------------------------------------------
    # 趋势 35
    # --------------------------------------------------------

    if price > row["MA20"]:
        score += 8

    if price > row["MA60"]:
        score += 10

    if row["MA20"] > row["MA60"]:
        score += 8

    if row["MA60"] > row["MA120"]:
        score += 9

    # --------------------------------------------------------
    # 动量 25
    # --------------------------------------------------------

    if row["RET20"] > 0:
        score += 8

    if row["RET60"] > 0:
        score += 8

    if row["RET120"] > 0:
        score += 9

    # --------------------------------------------------------
    # 回撤 15
    # --------------------------------------------------------

    dd60 = row["DD60"]

    if dd60 >= -0.05:
        score += 8

    elif dd60 >= -0.10:
        score += 6

    elif dd60 >= -0.15:
        score += 3

    dd120 = row["DD120"]

    if dd120 >= -0.10:
        score += 7

    elif dd120 >= -0.20:
        score += 4

    # --------------------------------------------------------
    # RSI 10
    # --------------------------------------------------------

    rsi = row["RSI14"]

    if 45 <= rsi <= 65:
        score += 10

    elif 40 <= rsi < 45:
        score += 7

    elif 65 < rsi <= 70:
        score += 7

    elif 30 <= rsi < 40:
        score += 5

    elif rsi > 75:
        score += 2

    else:
        score += 4

    # --------------------------------------------------------
    # MACD 5
    # --------------------------------------------------------

    if row["MACD"] > row["SIGNAL"]:
        score += 3

    if row["MACD_HIST"] > 0:
        score += 2

    # --------------------------------------------------------
    # 稳定性 10
    # --------------------------------------------------------

    volatility = row["VOLATILITY20"]

    if volatility < 0.20:
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
# 生成评分
# ============================================================

def add_scores(df):

    df = df.copy()

    df["score"] = np.nan

    required = [
        "MA20",
        "MA60",
        "MA120",
        "RET20",
        "RET60",
        "RET120",
        "DD60",
        "DD120",
        "RSI14",
        "MACD",
        "SIGNAL",
        "MACD_HIST",
        "VOLATILITY20"
    ]

    for i in range(len(df)):

        row = df.iloc[i]

        if any(
            pd.isna(row[x])
            for x in required
        ):
            continue

        df.iloc[
            i,
            df.columns.get_loc("score")
        ] = calculate_score(row)

    return df


# ============================================================
# 生成趋势信号
# ============================================================

def add_signal(df):

    df = df.copy()

    df["signal"] = 0

    for i in range(len(df)):

        row = df.iloc[i]

        score = row["score"]

        if pd.isna(score):

            continue

        # ----------------------------------------------------
        # ETF策略：
        #
        # 75分以上：
        # 趋势允许建仓
        #
        # 60~74：
        # 持有观察
        #
        # 55以下：
        # 风险控制
        # ----------------------------------------------------

        if score >= 75:

            df.iloc[
                i,
                df.columns.get_loc("signal")
            ] = 1

        elif score < 55:

            df.iloc[
                i,
                df.columns.get_loc("signal")
            ] = -1

        else:

            df.iloc[
                i,
                df.columns.get_loc("signal")
            ] = 0

    return df


# ============================================================
# 策略A：买入持有
# ============================================================

def backtest_buy_hold(df):

    close = df["Close"]

    daily_return = (
        close.pct_change()
        .fillna(0)
    )

    equity = (
        1 + daily_return
    ).cumprod()

    return equity


# ============================================================
# 策略B：
# 10万元 + 五档分批建仓
#
# 第一次：
# 趋势信号 >=75
#
# 第二次：
# 首次建仓价 -3%
#
# 第三次：
# 首次建仓价 -6%
#
# 第四次：
# 首次建仓价 -9%
#
# 第五次：
# 首次建仓价 -12%
#
# 每次20000元
# ============================================================

def backtest_tranche(df):

    df = df.copy()

    cash = INITIAL_CAPITAL

    shares = 0.0

    first_entry_price = None

    tranche_count = 0

    equity_curve = []

    entry_count = 0

    exit_count = 0

    for i in range(len(df)):

        price = float(
            df["Close"].iloc[i]
        )

        signal = df["signal"].iloc[i]

        # ====================================================
        # 第一次建仓
        # ====================================================

        if (
            tranche_count == 0
            and signal == 1
        ):

            amount = TRANCHE

            fee = (
                amount
                * FEE_RATE
            )

            if cash >= amount + fee:

                buy_shares = (
                    amount / price
                )

                shares += buy_shares

                cash -= (
                    amount + fee
                )

                first_entry_price = price

                tranche_count = 1

                entry_count += 1

        # ====================================================
        # 后续回撤加仓
        # ====================================================

        elif (
            tranche_count > 0
            and tranche_count < 5
            and first_entry_price is not None
        ):

            drawdown = (
                price
                / first_entry_price
                - 1
            )

            target_level = (
                -0.03
                * tranche_count
            )

            if drawdown <= target_level:

                amount = TRANCHE

                fee = (
                    amount
                    * FEE_RATE
                )

                if cash >= amount + fee:

                    buy_shares = (
                        amount / price
                    )

                    shares += buy_shares

                    cash -= (
                        amount + fee
                    )

                    tranche_count += 1

                    entry_count += 1

        # ====================================================
        # 风险退出
        #
        # 只有在趋势严重破坏时退出
        # 防止ETF因为普通波动频繁交易
        # ====================================================

        if (
            shares > 0
            and signal == -1
            and df["Close"].iloc[i]
            < df["MA120"].iloc[i]
        ):

            value = (
                shares * price
            )

            fee = (
                value
                * FEE_RATE
            )

            cash += (
                value - fee
            )

            shares = 0

            first_entry_price = None

            tranche_count = 0

            exit_count += 1

        equity = (
            cash
            + shares * price
        )

        equity_curve.append(
            equity
        )

    equity_series = pd.Series(
        equity_curve,
        index=df.index
    )

    return (
        equity_series,
        entry_count,
        exit_count
    )


# ============================================================
# 策略C：
# 量化择时
#
# 满仓/空仓
# 但不会因为短期波动频繁交易
# ============================================================

def backtest_timing(df):

    df = df.copy()

    cash = INITIAL_CAPITAL

    shares = 0.0

    equity_curve = []

    position = 0

    trades = 0

    for i in range(len(df)):

        price = float(
            df["Close"].iloc[i]
        )

        signal = df["signal"].iloc[i]

        ma120 = df["MA120"].iloc[i]

        # ----------------------------------------------------
        # 买入
        # ----------------------------------------------------

        if (
            position == 0
            and signal == 1
            and not pd.isna(ma120)
            and price > ma120
        ):

            amount = cash

            fee = (
                amount
                * FEE_RATE
            )

            shares = (
                (amount - fee)
                / price
            )

            cash = 0

            position = 1

            trades += 1

        # ----------------------------------------------------
        # 卖出
        # ----------------------------------------------------

        elif (
            position == 1
            and signal == -1
            and not pd.isna(ma120)
            and price < ma120
        ):

            value = (
                shares * price
            )

            fee = (
                value
                * FEE_RATE
            )

            cash = (
                value - fee
            )

            shares = 0

            position = 0

            trades += 1

        equity = (
            cash
            + shares * price
        )

        equity_curve.append(
            equity
        )

    equity_series = pd.Series(
        equity_curve,
        index=df.index
    )

    return (
        equity_series,
        trades
    )


# ============================================================
# 计算回测指标
# ============================================================

def metrics(
    equity,
    initial_capital=INITIAL_CAPITAL
):

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
        / initial_capital
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

    rolling_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / rolling_max
        - 1
    )

    max_drawdown = drawdown.min()

    daily_return = (
        equity.pct_change()
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
# 三ETF轮动
#
# 每20个交易日重新排名
#
# 只持有当前评分最高ETF
#
# 最高ETF评分 <60：
# 全部现金
# ============================================================

def backtest_rotation(data_dict):

    all_dates = None

    for code, df in data_dict.items():

        dates = df.index

        if all_dates is None:

            all_dates = dates

        else:

            all_dates = all_dates.intersection(
                dates
            )

    all_dates = sorted(
        list(all_dates)
    )

    cash = INITIAL_CAPITAL

    shares = {}

    for code in data_dict:

        shares[code] = 0.0

    current_code = None

    equity_curve = []

    last_rotation_index = -20

    trade_count = 0

    for i, date in enumerate(all_dates):

        # ----------------------------------------------------
        # 每20个交易日轮动一次
        # ----------------------------------------------------

        if (
            i - last_rotation_index >= 20
            or current_code is None
        ):

            candidates = []

            for code, df in data_dict.items():

                if date not in df.index:

                    continue

                row = df.loc[date]

                score = row["score"]

                if pd.isna(score):

                    continue

                candidates.append(
                    (
                        code,
                        score
                    )
                )

            if candidates:

                candidates.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                best_code, best_score = (
                    candidates[0]
                )

                # ------------------------------------------------
                # 评分低于60：
                # 空仓
                # ------------------------------------------------

                if best_score < 60:

                    target_code = None

                else:

                    target_code = best_code

                # ------------------------------------------------
                # 换仓
                # ------------------------------------------------

                if target_code != current_code:

                    # 先卖出旧仓
                    if current_code is not None:

                        old_price = float(
                            data_dict[
                                current_code
                            ].loc[
                                date,
                                "Close"
                            ]
                        )

                        value = (
                            shares[
                                current_code
                            ]
                            * old_price
                        )

                        fee = (
                            value
                            * FEE_RATE
                        )

                        cash += (
                            value - fee
                        )

                        shares[
                            current_code
                        ] = 0

                        trade_count += 1

                    # 买入新仓
                    if target_code is not None:

                        new_price = float(
                            data_dict[
                                target_code
                            ].loc[
                                date,
                                "Close"
                            ]
                        )

                        fee = (
                            cash
                            * FEE_RATE
                        )

                        shares[
                            target_code
                        ] = (
                            (cash - fee)
                            / new_price
                        )

                        cash = 0

                        trade_count += 1

                    current_code = target_code

                last_rotation_index = i

        # ----------------------------------------------------
        # 计算每日资产
        # ----------------------------------------------------

        total_value = cash

        for code in data_dict:

            if date not in data_dict[
                code
            ].index:

                continue

            price = float(
                data_dict[
                    code
                ].loc[
                    date,
                    "Close"
                ]
            )

            total_value += (
                shares[code]
                * price
            )

        equity_curve.append(
            total_value
        )

    equity = pd.Series(
        equity_curve,
        index=all_dates
    )

    return (
        equity,
        trade_count
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)

    print(
        "ETF V4 专用量化回测开始"
    )

    print("=" * 70)

    data_dict = {}

    # ========================================================
    # 获取三只ETF
    # ========================================================

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        yahoo = item["yahoo"]

        print()

        print(
            f"正在处理：{code} {name}"
        )

        df = download_data(
            yahoo
        )

        if df is None:

            continue

        df = calculate_indicators(
            df
        )

        df = add_scores(
            df
        )

        df = add_signal(
            df
        )

        data_dict[code] = df

        time.sleep(1)

    if len(data_dict) == 0:

        print(
            "❌ 三只ETF全部获取失败"
        )

        return

    # ========================================================
    # 单ETF回测
    # ========================================================

    results = []

    equity_files = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        if code not in data_dict:

            continue

        df = data_dict[code]

        print()

        print(
            "=" * 70
        )

        print(
            f"开始回测：{code} {name}"
        )

        # ----------------------------------------------------
        # 买入持有
        # ----------------------------------------------------

        bh_equity_ratio = (
            backtest_buy_hold(df)
        )

        bh_equity = (
            bh_equity_ratio
            * INITIAL_CAPITAL
        )

        bh_metrics = metrics(
            bh_equity
        )

        # ----------------------------------------------------
        # 分批建仓
        # ----------------------------------------------------

        tranche_equity, entries, exits = (
            backtest_tranche(df)
        )

        tranche_metrics = metrics(
            tranche_equity
        )

        # ----------------------------------------------------
        # 量化择时
        # ----------------------------------------------------

        timing_equity, timing_trades = (
            backtest_timing(df)
        )

        timing_metrics = metrics(
            timing_equity
        )

        # ----------------------------------------------------
        # 保存结果
        # ----------------------------------------------------

        results.append({

            "code": code,

            "name": name,

            "buy_hold_return":
                bh_metrics[
                    "total_return"
                ],

            "buy_hold_annual":
                bh_metrics[
                    "annual_return"
                ],

            "buy_hold_drawdown":
                bh_metrics[
                    "max_drawdown"
                ],

            "buy_hold_sharpe":
                bh_metrics[
                    "sharpe"
                ],

            "tranche_return":
                tranche_metrics[
                    "total_return"
                ],

            "tranche_annual":
                tranche_metrics[
                    "annual_return"
                ],

            "tranche_drawdown":
                tranche_metrics[
                    "max_drawdown"
                ],

            "tranche_sharpe":
                tranche_metrics[
                    "sharpe"
                ],

            "tranche_entries":
                entries,

            "tranche_exits":
                exits,

            "timing_return":
                timing_metrics[
                    "total_return"
                ],

            "timing_annual":
                timing_metrics[
                    "annual_return"
                ],

            "timing_drawdown":
                timing_metrics[
                    "max_drawdown"
                ],

            "timing_sharpe":
                timing_metrics[
                    "sharpe"
                ],

            "timing_trades":
                timing_trades
        })

        # ----------------------------------------------------
        # 保存净值
        # ----------------------------------------------------

        equity_df = pd.DataFrame({

            "date": df.index,

            "price": df["Close"].values,

            "score": df["score"].values,

            "signal": df["signal"].values
        })

        equity_df.to_csv(
            os.path.join(
                DATA_DIR,
                f"etf_{code}_signals.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"{name} 买入持有："
            f"{bh_metrics['total_return']}%"
        )

        print(
            f"{name} 分批建仓："
            f"{tranche_metrics['total_return']}%"
        )

        print(
            f"{name} 量化择时："
            f"{timing_metrics['total_return']}%"
        )

    # ========================================================
    # 保存单ETF结果
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_backtest_result.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 三ETF轮动
    # ========================================================

    print()

    print("=" * 70)

    print(
        "开始三ETF轮动回测"
    )

    print("=" * 70)

    rotation_equity, rotation_trades = (
        backtest_rotation(
            data_dict
        )
    )

    rotation_metrics = metrics(
        rotation_equity
    )

    rotation_result = pd.DataFrame([{

        "strategy":
            "三ETF量化轮动",

        "total_return":
            rotation_metrics[
                "total_return"
            ],

        "annual_return":
            rotation_metrics[
                "annual_return"
            ],

        "max_drawdown":
            rotation_metrics[
                "max_drawdown"
            ],

        "sharpe":
            rotation_metrics[
                "sharpe"
            ],

        "trade_count":
            rotation_trades

    }])

    rotation_result.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_rotation_result.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 当前三ETF排名
    # ========================================================

    ranking = []

    for item in ETF_LIST:

        code = item["code"]

        name = item["name"]

        if code not in data_dict:

            continue

        df = data_dict[code]

        last = df.iloc[-1]

        ranking.append({

            "code": code,

            "name": name,

            "date": str(
                df.index[-1].date()
            ),

            "price": round(
                float(last["Close"]),
                4
            ),

            "score": round(
                float(last["score"]),
                2
            ),

            "return20": round(
                float(last["RET20"]) * 100,
                2
            ),

            "return60": round(
                float(last["RET60"]) * 100,
                2
            ),

            "return120": round(
                float(last["RET120"]) * 100,
                2
            ),

            "drawdown60": round(
                float(last["DD60"]) * 100,
                2
            ),

            "rsi14": round(
                float(last["RSI14"]),
                2
            ),

            "volatility20": round(
                float(
                    last[
                        "VOLATILITY20"
                    ]
                ) * 100,
                2
            )
        })

    ranking_df = pd.DataFrame(
        ranking
    )

    ranking_df = ranking_df.sort_values(
        "score",
        ascending=False
    )

    ranking_df.insert(
        0,
        "rank",
        range(
            1,
            len(ranking_df) + 1
        )
    )

    ranking_df.to_csv(
        os.path.join(
            DATA_DIR,
            "etf_ranking.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 最终输出
    # ========================================================

    print()

    print("=" * 70)

    print(
        "ETF V4 回测完成"
    )

    print("=" * 70)

    print()

    print(
        "三只ETF当前排名："
    )

    print(
        ranking_df.to_string(
            index=False
        )
    )

    print()

    print(
        "单ETF历史回测："
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print()

    print(
        "三ETF轮动："
    )

    print(
        rotation_result.to_string(
            index=False
        )
    )

    print()

    print("=" * 70)

    print(
        "结果文件已经保存到 data/"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
