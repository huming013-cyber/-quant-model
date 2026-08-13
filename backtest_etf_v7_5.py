import pandas as pd
import numpy as np
import os
from datetime import timedelta


# ============================================================
# ETF V7.5
# 最终验证版：
# 3%分批建仓 + 极端风险保护 + 严格样本外验证
# ============================================================

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000

# 正常加仓间距
NORMAL_STEP = 0.03

# 极端风险保护：
# 如果价格相对于最近一次成交价在很短时间内快速下跌，
# 暂停下一档加仓，避免快速暴跌过程中一次性打满仓位
EXTREME_DROP_1D = 0.05
EXTREME_DROP_3D = 0.08

# 风险保护暂停交易天数
PROTECTION_DAYS = 5

ETF_LIST = {
    "159209": "红利质量ETF",
    "159399": "现金流ETF",
    "159581": "红利ETF",
}


# ============================================================
# 读取数据
# ============================================================

def load_data(code):

    file_path = f"data/etf_{code}_signals.csv"

    print(f"读取本地数据：{file_path}")

    if not os.path.exists(file_path):
        print("文件不存在")
        return None

    df = pd.read_csv(file_path)

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
    df = df.drop_duplicates(subset=["date"])

    df = df.reset_index(drop=True)

    print(f"成功读取 {len(df)} 条数据")

    if len(df) > 0:
        print(
            f"数据范围："
            f"{df['date'].iloc[0].date()} → "
            f"{df['date'].iloc[-1].date()}"
        )

    return df


# ============================================================
# 计算最大回撤
# ============================================================

def calculate_max_drawdown(equity):

    equity = pd.Series(equity).astype(float)

    if len(equity) == 0:
        return 0

    running_max = equity.cummax()

    drawdown = equity / running_max - 1

    return drawdown.min() * 100


# ============================================================
# Sharpe
# ============================================================

def calculate_sharpe(equity):

    equity = pd.Series(equity).astype(float)

    if len(equity) < 2:
        return 0

    returns = equity.pct_change().dropna()

    if returns.std() == 0:
        return 0

    return (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )


# ============================================================
# 计算年化收益
# ============================================================

def calculate_annual_return(total_return, days):

    if days <= 0:
        return 0

    years = days / 365

    if years <= 0:
        return 0

    return (
        (1 + total_return / 100) ** (1 / years) - 1
    ) * 100


# ============================================================
# 极端下跌检测
# ============================================================

def extreme_drop_detected(df, i):

    if i <= 0:
        return False

    current_price = df.loc[i, "price"]

    previous_price = df.loc[i - 1, "price"]

    # 单日跌幅
    one_day_drop = (
        current_price / previous_price - 1
    )

    if one_day_drop <= -EXTREME_DROP_1D:
        return True

    # 最近3个交易日跌幅
    if i >= 3:

        price_3d = df.loc[i - 3, "price"]

        three_day_drop = (
            current_price / price_3d - 1
        )

        if three_day_drop <= -EXTREME_DROP_3D:
            return True

    return False


# ============================================================
# 运行分批策略
# ============================================================

def run_strategy(
    df,
    step=0.03,
    use_protection=True
):

    cash = INITIAL_CAPITAL

    shares = 0

    invested = 0

    tranche_count = 0

    first_buy_price = None

    last_buy_price = None

    protection_until = None

    equity_curve = []

    capital_usage = []

    trades = []

    for i in range(len(df)):

        date = df.loc[i, "date"]
        price = float(df.loc[i, "price"])

        # ----------------------------------------------------
        # 风险保护检查
        # ----------------------------------------------------

        extreme = False

        if use_protection:

            extreme = extreme_drop_detected(df, i)

            if extreme:

                protection_until = (
                    date + timedelta(days=PROTECTION_DAYS)
                )

        protection_active = False

        if protection_until is not None:

            if date <= protection_until:
                protection_active = True

        # ----------------------------------------------------
        # 首次建仓
        # ----------------------------------------------------

        if tranche_count == 0:

            if not protection_active:

                amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if amount > 0:

                    buy_shares = int(
                        amount / price / 100
                    ) * 100

                    if buy_shares > 0:

                        cost = buy_shares * price

                        shares += buy_shares

                        cash -= cost

                        invested += cost

                        tranche_count += 1

                        first_buy_price = price

                        last_buy_price = price

                        trades.append({
                            "date": date,
                            "type": "BUY",
                            "price": price,
                            "amount": cost,
                            "tranche": tranche_count
                        })

        # ----------------------------------------------------
        # 后续加仓
        # ----------------------------------------------------

        else:

            target_price = (
                last_buy_price * (1 - step)
            )

            if (
                price <= target_price
                and
                not protection_active
                and
                tranche_count < 5
            ):

                amount = min(
                    TRANCHE_AMOUNT,
                    cash
                )

                if amount > 0:

                    buy_shares = int(
                        amount / price / 100
                    ) * 100

                    if buy_shares > 0:

                        cost = buy_shares * price

                        shares += buy_shares

                        cash -= cost

                        invested += cost

                        tranche_count += 1

                        last_buy_price = price

                        trades.append({
                            "date": date,
                            "type": "BUY",
                            "price": price,
                            "amount": cost,
                            "tranche": tranche_count
                        })

        # ----------------------------------------------------
        # 每日资产
        # ----------------------------------------------------

        market_value = shares * price

        equity = cash + market_value

        equity_curve.append(equity)

        if INITIAL_CAPITAL > 0:

            capital_usage.append(
                invested / INITIAL_CAPITAL * 100
            )

    # ========================================================
    # 最终结果
    # ========================================================

    final_price = float(df["price"].iloc[-1])

    final_equity = (
        cash + shares * final_price
    )

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    annual_return = calculate_annual_return(
        total_return,
        days
    )

    max_drawdown = calculate_max_drawdown(
        equity_curve
    )

    sharpe = calculate_sharpe(
        equity_curve
    )

    avg_usage = (
        np.mean(capital_usage)
        if capital_usage
        else 0
    )

    return {
        "return": total_return,
        "annual": annual_return,
        "drawdown": max_drawdown,
        "sharpe": sharpe,
        "capital_usage": avg_usage,
        "tranches": tranche_count,
        "trades": trades,
        "equity": equity_curve,
    }


# ============================================================
# 买入持有
# ============================================================

def buy_hold(df):

    start_price = float(df["price"].iloc[0])

    end_price = float(df["price"].iloc[-1])

    total_return = (
        end_price / start_price - 1
    ) * 100

    equity = (
        INITIAL_CAPITAL
        * df["price"]
        / start_price
    )

    days = (
        df["date"].iloc[-1]
        - df["date"].iloc[0]
    ).days

    annual = calculate_annual_return(
        total_return,
        days
    )

    drawdown = calculate_max_drawdown(
        equity
    )

    sharpe = calculate_sharpe(
        equity
    )

    return {
        "return": total_return,
        "annual": annual,
        "drawdown": drawdown,
        "sharpe": sharpe,
    }


# ============================================================
# 样本外验证
# ============================================================

def rolling_validation(df):

    results = []

    # 至少需要约一年训练 + 半年验证
    MIN_TRAIN_DAYS = 250
    VALIDATION_DAYS = 125

    if len(df) < MIN_TRAIN_DAYS + VALIDATION_DAYS:

        print(
            "数据不足，无法形成至少一个完整训练+验证窗口"
        )

        return results

    start = 0

    window = 1

    while True:

        train_end = start + MIN_TRAIN_DAYS

        validation_end = (
            train_end + VALIDATION_DAYS
        )

        if validation_end >= len(df):

            break

        train = df.iloc[
            start:train_end
        ].copy()

        validation = df.iloc[
            train_end:validation_end
        ].copy()

        print()
        print(f"窗口 {window}")

        print(
            f"训练期："
            f"{train['date'].iloc[0].date()} → "
            f"{train['date'].iloc[-1].date()}"
        )

        print(
            f"验证期："
            f"{validation['date'].iloc[0].date()} → "
            f"{validation['date'].iloc[-1].date()}"
        )

        # ----------------------------------------------------
        # 训练阶段：
        # 只比较 2%、3%、4%
        # 避免参数过度优化
        # ----------------------------------------------------

        candidates = [0.02, 0.03, 0.04]

        train_results = []

        for step in candidates:

            result = run_strategy(
                train,
                step=step,
                use_protection=True
            )

            train_results.append(
                (
                    step,
                    result
                )
            )

        # 综合评分
        best_step = None
        best_score = -999999

        for step, result in train_results:

            score = (
                result["annual"]
                + result["sharpe"] * 5
                + result["drawdown"] * 0.2
            )

            if score > best_score:

                best_score = score
                best_step = step

        # ----------------------------------------------------
        # 样本外验证
        # ----------------------------------------------------

        validation_result = run_strategy(
            validation,
            step=best_step,
            use_protection=True
        )

        print(
            f"训练期选择："
            f"{best_step * 100:.0f}%"
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
            f"平均资金使用率："
            f"{validation_result['capital_usage']:.2f}%"
        )

        results.append({
            "window": window,
            "step": best_step,
            "return": validation_result["return"],
            "annual": validation_result["annual"],
            "drawdown": validation_result["drawdown"],
            "sharpe": validation_result["sharpe"],
            "capital_usage": validation_result["capital_usage"],
        })

        # 滚动前进约半年
        start += VALIDATION_DAYS

        window += 1

    return results


# ============================================================
# 汇总
# ============================================================

def summarize_validation(results):

    if not results:

        return None

    df = pd.DataFrame(results)

    positive_windows = (
        df["return"] > 0
    ).sum()

    windows = len(df)

    win_rate = (
        positive_windows
        / windows
        * 100
    )

    average_return = df["return"].mean()

    average_annual = df["annual"].mean()

    average_drawdown = df["drawdown"].mean()

    average_sharpe = df["sharpe"].mean()

    average_usage = df["capital_usage"].mean()

    most_common_step = (
        df["step"]
        .round(4)
        .mode()
        .iloc[0]
    )

    parameter_stability = (
        (df["step"] == most_common_step).mean()
        * 100
    )

    return {
        "windows": windows,
        "positive_windows": positive_windows,
        "win_rate": win_rate,
        "average_return": average_return,
        "average_annual": average_annual,
        "average_drawdown": average_drawdown,
        "average_sharpe": average_sharpe,
        "average_usage": average_usage,
        "most_common_step": most_common_step,
        "parameter_stability": parameter_stability,
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ETF V7.5：3%分批建仓 + 极端风险保护")
    print("=" * 70)
    print()
    print("目标：")
    print("1. 验证3%分批建仓")
    print("2. 验证极端风险保护")
    print("3. 严格滚动样本外验证")
    print("4. 不使用MA20/MA60趋势过滤")
    print()

    final_results = []

    for code, name in ETF_LIST.items():

        print()
        print("=" * 70)
        print(f"{code} {name}")
        print("=" * 70)

        df = load_data(code)

        if df is None or len(df) < 2:

            continue

        # ----------------------------------------------------
        # 上市以来完整回测
        # ----------------------------------------------------

        print()
        print("【上市以来完整回测】")

        buy_hold_result = buy_hold(df)

        strategy_result = run_strategy(
            df,
            step=NORMAL_STEP,
            use_protection=True
        )

        print(
            f"买入持有收益："
            f"{buy_hold_result['return']:.2f}%"
        )

        print(
            f"买入持有年化："
            f"{buy_hold_result['annual']:.2f}%"
        )

        print(
            f"买入持有最大回撤："
            f"{buy_hold_result['drawdown']:.2f}%"
        )

        print(
            f"买入持有Sharpe："
            f"{buy_hold_result['sharpe']:.2f}"
        )

        print()

        print("V7.5策略：")

        print(
            f"固定加仓间距："
            f"{NORMAL_STEP * 100:.0f}%"
        )

        print(
            f"策略收益："
            f"{strategy_result['return']:.2f}%"
        )

        print(
            f"策略年化："
            f"{strategy_result['annual']:.2f}%"
        )

        print(
            f"策略最大回撤："
            f"{strategy_result['drawdown']:.2f}%"
        )

        print(
            f"策略Sharpe："
            f"{strategy_result['sharpe']:.2f}"
        )

        print(
            f"平均资金使用率："
            f"{strategy_result['capital_usage']:.2f}%"
        )

        print(
            f"最终建仓档位："
            f"{strategy_result['tranches']} / 5"
        )

        # ----------------------------------------------------
        # 样本外
        # ----------------------------------------------------

        print()
        print("【严格滚动样本外验证】")

        validation_results = rolling_validation(df)

        summary = summarize_validation(
            validation_results
        )

        if summary is None:

            continue

        print()
        print("V7.5汇总：")

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
            f"{summary['average_usage']:.2f}%"
        )

        print(
            f"最常出现参数："
            f"{summary['most_common_step'] * 100:.0f}%"
        )

        print(
            f"参数稳定性："
            f"{summary['parameter_stability']:.2f}%"
        )

        final_results.append({
            "code": code,
            "name": name,
            "buy_hold_return":
                buy_hold_result["return"],
            "strategy_return":
                strategy_result["return"],
            "strategy_annual":
                strategy_result["annual"],
            "strategy_drawdown":
                strategy_result["drawdown"],
            "strategy_sharpe":
                strategy_result["sharpe"],
            "validation_windows":
                summary["windows"],
            "validation_win_rate":
                summary["win_rate"],
            "validation_average_return":
                summary["average_return"],
            "validation_average_annual":
                summary["average_annual"],
            "validation_average_drawdown":
                summary["average_drawdown"],
            "validation_average_sharpe":
                summary["average_sharpe"],
            "validation_usage":
                summary["average_usage"],
        })

    # ========================================================
    # 最终汇总
    # ========================================================

    print()
    print("=" * 70)
    print("V7.5最终汇总")
    print("=" * 70)

    if final_results:

        result_df = pd.DataFrame(
            final_results
        )

        print(
            result_df.to_string(
                index=False
            )
        )

        os.makedirs(
            "data",
            exist_ok=True
        )

        output_file = (
            "data/etf_v7_5_result.csv"
        )

        result_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            f"结果已保存：{output_file}"
        )

    print()
    print("=" * 70)
    print("V7.5完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
