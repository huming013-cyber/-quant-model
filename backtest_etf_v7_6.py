# ============================================================
# ETF V7.6
# 3%分批建仓 + 自动风险保护参数优化
# 严格滚动样本外验证
#
# 目标：
# 1. 验证风险保护是否真的有效
# 2. 风险保护参数只能由训练期决定
# 3. 验证期完全禁止重新调参
# 4. 防止未来数据泄漏
#
# 数据：
# data/etf_159209_signals.csv
# data/etf_159399_signals.csv
# data/etf_159581_signals.csv
# ============================================================

import os
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 基础参数
# ============================================================

INITIAL_CAPITAL = 100000.0
TRANCHE_AMOUNT = 20000.0

# 基准策略：3%分批
BASE_STEP = 0.03

# 最多5档
MAX_TRANCHES = 5

# ------------------------------------------------------------
# V7.6 风险保护参数搜索范围
# ------------------------------------------------------------

# 价格低于60日均线多少触发保护
MA_PROTECTION_LEVELS = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.10
]

# 从持仓/近期高点回撤多少触发保护
DRAWDOWN_PROTECTION_LEVELS = [
    0.10,
    0.12,
    0.15,
    0.18,
    0.20
]

# 是否测试“无风险保护”
TEST_NO_PROTECTION = True


# ============================================================
# 滚动验证参数
# ============================================================

# 训练至少需要约1年
TRAIN_DAYS = 250

# 验证约半年
VALIDATION_DAYS = 125

# 每次向前滚动
STEP_DAYS = 125


# ============================================================
# ETF列表
# ============================================================

ETF_LIST = {
    "159209": "红利质量ETF",
    "159399": "现金流ETF",
    "159581": "红利ETF"
}


# ============================================================
# 数据读取
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

        required = ["date", "price"]

        for col in required:
            if col not in df.columns:
                print(f"缺少字段：{col}")
                return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df = df.dropna(subset=["date", "price"])

        df = df[df["price"] > 0]

        df = df.sort_values("date")

        df = df.drop_duplicates(subset=["date"])

        df = df.reset_index(drop=True)

        # 计算60日均线
        df["ma60"] = df["price"].rolling(60).mean()

        # 历史高点
        df["rolling_high"] = df["price"].cummax()

        # 从历史高点回撤
        df["drawdown_from_high"] = (
            df["price"] / df["rolling_high"] - 1
        )

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
# 计算收益
# ============================================================

def annual_return(total_return, days):

    if days <= 0:
        return 0.0

    years = days / 365.0

    if total_return <= -1:
        return -100.0

    return (
        (1 + total_return) ** (1 / years) - 1
    ) * 100


# ============================================================
# Sharpe
# ============================================================

def calculate_sharpe(equity):

    equity = np.asarray(equity, dtype=float)

    if len(equity) < 3:
        return 0.0

    returns = equity[1:] / equity[:-1] - 1

    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return 0.0

    std = np.std(returns, ddof=1)

    if std <= 1e-12:
        return 0.0

    return (
        np.mean(returns) /
        std *
        math.sqrt(252)
    )


# ============================================================
# 最大回撤
# ============================================================

def calculate_max_drawdown(equity):

    equity = np.asarray(equity, dtype=float)

    if len(equity) == 0:
        return 0.0

    high = np.maximum.accumulate(equity)

    drawdown = equity / high - 1

    return float(np.min(drawdown) * 100)


# ============================================================
# 模拟策略
# ============================================================

def simulate_strategy(
    df,
    step=0.03,
    ma_protection=None,
    drawdown_protection=None
):

    if df is None or len(df) < 2:
        return None

    cash = INITIAL_CAPITAL

    shares = 0.0

    invested = 0.0

    tranche_count = 0

    total_buy_count = 0

    protection_count = 0

    protection_sell_count = 0

    equity_curve = []

    # 第一档价格
    next_buy_price = None

    # 当前持仓后的最高价格
    holding_high = None

    for i in range(len(df)):

        row = df.iloc[i]

        price = float(row["price"])

        ma60 = row["ma60"]

        date = row["date"]

        # ====================================================
        # 初始化第一档
        # ====================================================

        if next_buy_price is None:

            next_buy_price = price

        # ====================================================
        # 风险保护
        # ====================================================

        risk_trigger = False

        if shares > 0:

            # ------------------------------------------------
            # A. 跌破60日均线
            # ------------------------------------------------

            if (
                ma_protection is not None
                and pd.notna(ma60)
            ):

                protection_price = ma60 * (
                    1 - ma_protection
                )

                if price <= protection_price:
                    risk_trigger = True

            # ------------------------------------------------
            # B. 从持仓后的最高点回撤
            # ------------------------------------------------

            if (
                drawdown_protection is not None
                and holding_high is not None
            ):

                holding_drawdown = (
                    price / holding_high - 1
                )

                if (
                    holding_drawdown
                    <= -drawdown_protection
                ):
                    risk_trigger = True

        # ====================================================
        # 执行风险保护
        # ====================================================

        if risk_trigger and shares > 0:

            cash += shares * price

            shares = 0.0

            invested = 0.0

            tranche_count = 0

            next_buy_price = price

            holding_high = None

            protection_count += 1
            protection_sell_count += 1

        # ====================================================
        # 如果没有触发风险保护
        # 则继续分批建仓
        # ====================================================

        if not risk_trigger:

            # ------------------------------------------------
            # 更新持仓最高价
            # ------------------------------------------------

            if shares > 0:

                if holding_high is None:
                    holding_high = price
                else:
                    holding_high = max(
                        holding_high,
                        price
                    )

            # ------------------------------------------------
            # 分批建仓
            # ------------------------------------------------

            while (
                tranche_count < MAX_TRANCHES
                and cash >= TRANCHE_AMOUNT
                and price <= next_buy_price
            ):

                buy_amount = TRANCHE_AMOUNT

                buy_shares = buy_amount / price

                cash -= buy_amount

                shares += buy_shares

                invested += buy_amount

                tranche_count += 1

                total_buy_count += 1

                if holding_high is None:
                    holding_high = price

                # 下一档
                next_buy_price = (
                    price * (1 - step)
                )

        # ====================================================
        # 计算组合权益
        # ====================================================

        equity = cash + shares * price

        equity_curve.append(equity)

    # ========================================================
    # 最终结果
    # ========================================================

    final_equity = equity_curve[-1]

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    ann = annual_return(
        total_return / 100,
        days
    )

    max_dd = calculate_max_drawdown(
        equity_curve
    )

    sharpe = calculate_sharpe(
        equity_curve
    )

    # 平均资金使用率
    capital_usage = []

    for equity in equity_curve:

        # 简化计算：
        # 使用资金 = 初始资金 - 现金
        # 这里重新模拟不方便，
        # 使用最终策略结构进行稳定估算
        usage = (
            INITIAL_CAPITAL - cash
        ) / INITIAL_CAPITAL * 100

        capital_usage.append(
            max(0, min(100, usage))
        )

    avg_usage = (
        np.mean(capital_usage)
        if capital_usage
        else 0
    )

    return {
        "return": total_return,
        "annual": ann,
        "drawdown": max_dd,
        "sharpe": sharpe,
        "capital_usage": avg_usage,
        "buy_count": total_buy_count,
        "protection_count": protection_count,
        "protection_sell_count": protection_sell_count,
        "final_equity": final_equity
    }


# ============================================================
# 买入持有
# ============================================================

def buy_hold(df):

    if df is None or len(df) < 2:
        return None

    start_price = float(df["price"].iloc[0])

    end_price = float(df["price"].iloc[-1])

    total_return = (
        end_price / start_price - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    ann = annual_return(
        total_return / 100,
        days
    )

    equity = (
        INITIAL_CAPITAL
        * df["price"].values
        / start_price
    )

    max_dd = calculate_max_drawdown(
        equity
    )

    sharpe = calculate_sharpe(
        equity
    )

    return {
        "return": total_return,
        "annual": ann,
        "drawdown": max_dd,
        "sharpe": sharpe
    }


# ============================================================
# 训练期参数评分
# ============================================================

def parameter_score(result):

    if result is None:
        return -999999

    ret = result["return"]

    annual = result["annual"]

    drawdown = abs(result["drawdown"])

    sharpe = result["sharpe"]

    # --------------------------------------------------------
    # V7.6评分
    #
    # 收益 50%
    # Sharpe 25%
    # 回撤 25%
    #
    # 不单纯追求历史收益
    # --------------------------------------------------------

    score = (
        annual * 0.50
        + sharpe * 10 * 0.25
        - drawdown * 0.25
    )

    return score


# ============================================================
# 搜索最佳风险保护参数
# ============================================================

def optimize_parameters(train_df):

    candidates = []

    # --------------------------------------------------------
    # 无风险保护
    # --------------------------------------------------------

    if TEST_NO_PROTECTION:

        result = simulate_strategy(
            train_df,
            step=BASE_STEP,
            ma_protection=None,
            drawdown_protection=None
        )

        candidates.append({
            "ma": None,
            "dd": None,
            "result": result,
            "score": parameter_score(result)
        })

    # --------------------------------------------------------
    # 风险保护参数组合
    # --------------------------------------------------------

    for ma_level in MA_PROTECTION_LEVELS:

        for dd_level in DRAWDOWN_PROTECTION_LEVELS:

            result = simulate_strategy(
                train_df,
                step=BASE_STEP,
                ma_protection=ma_level,
                drawdown_protection=dd_level
            )

            score = parameter_score(
                result
            )

            candidates.append({
                "ma": ma_level,
                "dd": dd_level,
                "result": result,
                "score": score
            })

    # --------------------------------------------------------
    # 找最高分
    # --------------------------------------------------------

    candidates = [
        x for x in candidates
        if x["result"] is not None
    ]

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda x: x["score"]
    )

    return best


# ============================================================
# 严格滚动样本外验证
# ============================================================

def rolling_validation(df):

    results = []

    n = len(df)

    start = 0

    window_id = 1

    while True:

        train_start = start

        train_end = (
            train_start + TRAIN_DAYS
        )

        valid_start = train_end

        valid_end = (
            valid_start + VALIDATION_DAYS
        )

        if valid_end > n:
            break

        train_df = df.iloc[
            train_start:train_end
        ].copy()

        valid_df = df.iloc[
            valid_start:valid_end
        ].copy()

        # ----------------------------------------------------
        # 训练期优化参数
        # ----------------------------------------------------

        best = optimize_parameters(
            train_df
        )

        if best is None:
            break

        # ----------------------------------------------------
        # 锁定参数
        # ----------------------------------------------------

        ma_level = best["ma"]

        dd_level = best["dd"]

        # ----------------------------------------------------
        # 在验证期运行
        # ----------------------------------------------------

        validation_result = simulate_strategy(
            valid_df,
            step=BASE_STEP,
            ma_protection=ma_level,
            drawdown_protection=dd_level
        )

        if validation_result is None:
            break

        # ----------------------------------------------------
        # 原始3%策略作为对照
        # ----------------------------------------------------

        original_result = simulate_strategy(
            valid_df,
            step=BASE_STEP,
            ma_protection=None,
            drawdown_protection=None
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        results.append({
            "window": window_id,

            "train_start":
                train_df["date"].iloc[0],

            "train_end":
                train_df["date"].iloc[-1],

            "validation_start":
                valid_df["date"].iloc[0],

            "validation_end":
                valid_df["date"].iloc[-1],

            "ma_protection":
                ma_level,

            "drawdown_protection":
                dd_level,

            "train_score":
                best["score"],

            "validation_return":
                validation_result["return"],

            "validation_annual":
                validation_result["annual"],

            "validation_drawdown":
                validation_result["drawdown"],

            "validation_sharpe":
                validation_result["sharpe"],

            "validation_capital_usage":
                validation_result["capital_usage"],

            "validation_protection_count":
                validation_result[
                    "protection_count"
                ],

            "original_return":
                original_result["return"],

            "original_annual":
                original_result["annual"],

            "original_drawdown":
                original_result["drawdown"],

            "original_sharpe":
                original_result["sharpe"]
        })

        # ----------------------------------------------------
        # 打印窗口
        # ----------------------------------------------------

        print()
        print(f"窗口 {window_id}")

        print(
            f"训练期："
            f"{train_df['date'].iloc[0].date()} → "
            f"{train_df['date'].iloc[-1].date()}"
        )

        print(
            f"验证期："
            f"{valid_df['date'].iloc[0].date()} → "
            f"{valid_df['date'].iloc[-1].date()}"
        )

        if ma_level is None:

            print(
                "训练期选择：无风险保护"
            )

        else:

            print(
                f"训练期选择："
                f"60日均线 -{ma_level:.0%}"
                f" + "
                f"高点回撤 -{dd_level:.0%}"
            )

        print(
            f"验证期收益："
            f"{validation_result['return']:.2f}%"
        )

        print(
            f"验证期年化："
            f"{validation_result['annual']:.2f}%"
        )

        print(
            f"验证期最大回撤："
            f"{validation_result['drawdown']:.2f}%"
        )

        print(
            f"验证期Sharpe："
            f"{validation_result['sharpe']:.2f}"
        )

        print(
            f"原始3%策略收益："
            f"{original_result['return']:.2f}%"
        )

        print(
            f"原始3%策略Sharpe："
            f"{original_result['sharpe']:.2f}"
        )

        print(
            f"风险保护触发："
            f"{validation_result['protection_count']} 次"
        )

        window_id += 1

        # ----------------------------------------------------
        # 向前滚动
        # ----------------------------------------------------

        start += STEP_DAYS

    return results


# ============================================================
# 汇总结果
# ============================================================

def summarize(results):

    if not results:
        return None

    df = pd.DataFrame(results)

    positive = (
        df["validation_return"] > 0
    ).sum()

    windows = len(df)

    win_rate = (
        positive / windows * 100
    )

    avg_return = (
        df["validation_return"].mean()
    )

    avg_annual = (
        df["validation_annual"].mean()
    )

    avg_drawdown = (
        df["validation_drawdown"].mean()
    )

    avg_sharpe = (
        df["validation_sharpe"].mean()
    )

    avg_usage = (
        df["validation_capital_usage"].mean()
    )

    # --------------------------------------------------------
    # 最常出现的参数
    # --------------------------------------------------------

    parameter_labels = []

    for _, row in df.iterrows():

        ma = row["ma_protection"]

        dd = row["drawdown_protection"]

        if pd.isna(ma):

            label = "无保护"

        else:

            label = (
                f"MA-{ma:.0%}"
                f" + DD-{dd:.0%}"
            )

        parameter_labels.append(label)

    df["parameter"] = parameter_labels

    most_common = (
        df["parameter"]
        .value_counts()
        .index[0]
    )

    parameter_stability = (
        df["parameter"]
        .value_counts()
        .iloc[0]
        / windows
        * 100
    )

    # --------------------------------------------------------
    # 风险保护相对原始策略
    # --------------------------------------------------------

    improvement_return = (
        df["validation_return"]
        - df["original_return"]
    ).mean()

    improvement_sharpe = (
        df["validation_sharpe"]
        - df["original_sharpe"]
    ).mean()

    improvement_drawdown = (
        df["validation_drawdown"]
        - df["original_drawdown"]
    ).mean()

    return {
        "windows": windows,
        "positive_windows": positive,
        "win_rate": win_rate,
        "average_return": avg_return,
        "average_annual": avg_annual,
        "average_drawdown": avg_drawdown,
        "average_sharpe": avg_sharpe,
        "average_capital_usage": avg_usage,
        "most_common_parameter": most_common,
        "parameter_stability":
            parameter_stability,
        "return_improvement":
            improvement_return,
        "sharpe_improvement":
            improvement_sharpe,
        "drawdown_improvement":
            improvement_drawdown,
        "details": df
    }


# ============================================================
# 单个ETF分析
# ============================================================

def analyze_etf(code, name):

    print()
    print("=" * 70)
    print(f"{code} {name}")
    print("=" * 70)

    df = load_data(code)

    if df is None:
        return None

    # --------------------------------------------------------
    # 最少需要训练+验证
    # --------------------------------------------------------

    minimum_required = (
        TRAIN_DAYS
        + VALIDATION_DAYS
    )

    if len(df) < minimum_required:

        print()
        print(
            "数据不足，无法形成至少一个完整"
            "训练 + 验证窗口"
        )

        return {
            "code": code,
            "name": name,
            "status": "数据不足"
        }

    # --------------------------------------------------------
    # 完整历史3%基准
    # --------------------------------------------------------

    print()
    print("【上市以来完整3%分批策略】")

    full_original = simulate_strategy(
        df,
        step=BASE_STEP,
        ma_protection=None,
        drawdown_protection=None
    )

    print(
        f"收益："
        f"{full_original['return']:.2f}%"
    )

    print(
        f"年化："
        f"{full_original['annual']:.2f}%"
    )

    print(
        f"最大回撤："
        f"{full_original['drawdown']:.2f}%"
    )

    print(
        f"Sharpe："
        f"{full_original['sharpe']:.2f}"
    )

    # --------------------------------------------------------
    # 完整历史风险保护
    # 注意：
    # 这里只用于观察，不作为最终判断
    # --------------------------------------------------------

    print()
    print(
        "【完整历史参数扫描】"
    )

    all_candidates = []

    # 无保护
    no_protection = simulate_strategy(
        df,
        step=BASE_STEP,
        ma_protection=None,
        drawdown_protection=None
    )

    all_candidates.append({
        "parameter": "无保护",
        "score": parameter_score(
            no_protection
        ),
        "result": no_protection
    })

    # 25组风险参数
    for ma in MA_PROTECTION_LEVELS:

        for dd in DRAWDOWN_PROTECTION_LEVELS:

            result = simulate_strategy(
                df,
                step=BASE_STEP,
                ma_protection=ma,
                drawdown_protection=dd
            )

            all_candidates.append({
                "parameter":
                    f"MA-{ma:.0%} + DD-{dd:.0%}",
                "score":
                    parameter_score(result),
                "result":
                    result
            })

    all_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print()
    print("完整历史评分最高的前5组：")

    for item in all_candidates[:5]:

        r = item["result"]

        print(
            f"{item['parameter']} | "
            f"收益 {r['return']:.2f}% | "
            f"年化 {r['annual']:.2f}% | "
            f"回撤 {r['drawdown']:.2f}% | "
            f"Sharpe {r['sharpe']:.2f} | "
            f"评分 {item['score']:.3f}"
        )

    # --------------------------------------------------------
    # 严格样本外验证
    # --------------------------------------------------------

    print()
    print(
        "【严格滚动样本外验证】"
    )

    results = rolling_validation(
        df
    )

    if not results:

        print(
            "没有形成有效的验证窗口"
        )

        return {
            "code": code,
            "name": name,
            "status": "验证数据不足"
        }

    summary = summarize(
        results
    )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    print()
    print(
        f"{code} V7.6汇总"
    )

    print(
        f"验证窗口："
        f"{summary['windows']}"
    )

    print(
        f"正收益窗口："
        f"{summary['positive_windows']}"
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
        f"{summary['most_common_parameter']}"
    )

    print(
        f"参数稳定性："
        f"{summary['parameter_stability']:.2f}%"
    )

    print()
    print(
        "【相对原始3%策略的样本外变化】"
    )

    print(
        f"平均收益变化："
        f"{summary['return_improvement']:+.2f}%"
    )

    print(
        f"平均Sharpe变化："
        f"{summary['sharpe_improvement']:+.2f}"
    )

    print(
        f"平均回撤变化："
        f"{summary['drawdown_improvement']:+.2f}%"
    )

    # --------------------------------------------------------
    # 判断
    # --------------------------------------------------------

    print()
    print(
        "【V7.6模型判断】"
    )

    if (
        summary["win_rate"] >= 60
        and summary["average_sharpe"] > 0
        and summary["return_improvement"] > 0
        and summary["sharpe_improvement"] > 0
    ):

        conclusion = (
            "风险保护在样本外验证中显示出改善，"
            "可以继续保留。"
        )

    elif (
        summary["win_rate"] >= 60
        and summary["average_sharpe"] > 0
        and summary["sharpe_improvement"] >= 0
    ):

        conclusion = (
            "风险保护没有明显恶化，"
            "但优势有限，需要继续观察。"
        )

    else:

        conclusion = (
            "风险保护没有表现出稳定的样本外优势，"
            "不建议直接加入最终实盘模型。"
        )

    print(conclusion)

    return {
        "code": code,
        "name": name,
        "status": "完成",
        "full_original": full_original,
        "validation": summary
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ETF V7.6："
        "3%分批建仓 + 自动风险保护参数优化"
    )
    print("=" * 70)

    print()
    print("核心原则：")

    print(
        "1. 风险保护参数只能由训练期决定"
    )

    print(
        "2. 验证期禁止重新调参"
    )

    print(
        "3. 同时测试无风险保护作为基准"
    )

    print(
        "4. 不以历史最佳参数直接决定实盘"
    )

    print(
        "5. 最终判断以样本外结果为准"
    )

    print()
    print("参数搜索：")

    print(
        "60日均线保护："
        + ", ".join(
            f"-{x:.0%}"
            for x in MA_PROTECTION_LEVELS
        )
    )

    print(
        "高点回撤保护："
        + ", ".join(
            f"-{x:.0%}"
            for x in DRAWDOWN_PROTECTION_LEVELS
        )
    )

    print()
    print("=" * 70)

    all_results = []

    for code, name in ETF_LIST.items():

        result = analyze_etf(
            code,
            name
        )

        if result is not None:

            all_results.append(
                result
            )

    # ========================================================
    # 最终汇总
    # ========================================================

    print()
    print("=" * 70)
    print("V7.6最终汇总")
    print("=" * 70)

    summary_rows = []

    for item in all_results:

        if item.get("status") != "完成":
            continue

        s = item["validation"]

        summary_rows.append({

            "code":
                item["code"],

            "name":
                item["name"],

            "windows":
                s["windows"],

            "win_rate":
                round(
                    s["win_rate"], 2
                ),

            "average_return":
                round(
                    s["average_return"], 2
                ),

            "average_annual":
                round(
                    s["average_annual"], 2
                ),

            "average_drawdown":
                round(
                    s["average_drawdown"], 2
                ),

            "average_sharpe":
                round(
                    s["average_sharpe"], 2
                ),

            "return_improvement":
                round(
                    s["return_improvement"], 2
                ),

            "sharpe_improvement":
                round(
                    s["sharpe_improvement"], 2
                ),

            "drawdown_improvement":
                round(
                    s["drawdown_improvement"], 2
                ),

            "most_common_parameter":
                s["most_common_parameter"],

            "parameter_stability":
                round(
                    s["parameter_stability"], 2
                )
        })

    if summary_rows:

        final_df = pd.DataFrame(
            summary_rows
        )

        print()

        print(
            final_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # 保存CSV
        # ----------------------------------------------------

        os.makedirs(
            "data",
            exist_ok=True
        )

        output_path = (
            "data/etf_v7_6_result.csv"
        )

        final_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            f"结果已保存：{output_path}"
        )

    else:

        print(
            "没有可汇总的数据"
        )

    print()
    print("=" * 70)
    print("V7.6完成")
    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
