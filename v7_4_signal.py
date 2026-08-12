import os
import pandas as pd
import numpy as np

# ============================================================
# ETF V7.4 实盘执行信号
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000

# 当前模型核心参数
MAIN_CODE = "159581"
MAIN_NAME = "红利ETF"

# V7.3样本外验证中最常出现的参数
ADD_STEP = 0.03

# ============================================================
# 读取ETF数据
# ============================================================

def load_etf(code):

    filename = f"etf_{code}_signals.csv"

    path = os.path.join(
        DATA_DIR,
        filename
    )

    if not os.path.exists(path):

        print(f"❌ 找不到数据文件：{path}")

        return None

    try:

        df = pd.read_csv(path)

        if df.empty:

            print(f"❌ 数据为空：{path}")

            return None

        # 日期
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # 价格
        df["price"] = pd.to_numeric(
            df["price"],
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
            subset=["date"],
            keep="last"
        )

        if df.empty:

            return None

        return df

    except Exception as e:

        print(
            f"❌ 读取失败：{e}"
        )

        return None


# ============================================================
# 计算趋势
#
# 这里只作为风险过滤器。
#
# 不参与V7.3历史参数选择，
# 避免破坏原来的样本外验证逻辑。
# ============================================================

def calculate_trend(df):

    price = df["price"]

    ma20 = price.rolling(
        20
    ).mean()

    ma60 = price.rolling(
        60
    ).mean()

    current = price.iloc[-1]

    latest_ma20 = ma20.iloc[-1]
    latest_ma60 = ma60.iloc[-1]

    if pd.isna(latest_ma20):

        trend = "数据不足"

    elif pd.isna(latest_ma60):

        if current >= latest_ma20:

            trend = "偏强"

        else:

            trend = "偏弱"

    else:

        if (
            current > latest_ma20
            and latest_ma20 > latest_ma60
        ):

            trend = "强势"

        elif (
            current < latest_ma20
            and latest_ma20 < latest_ma60
        ):

            trend = "弱势"

        else:

            trend = "震荡"

    return {
        "current": current,
        "ma20": latest_ma20,
        "ma60": latest_ma60,
        "trend": trend
    }


# ============================================================
# 计算加仓价格
#
# 注意：
# 这里采用“首次建仓价格”为基准。
#
# 第1档：
# 首次建仓
#
# 第2档：
# 首次建仓价 × 97%
#
# 第3档：
# 首次建仓价 × 94%
#
# 第4档：
# 首次建仓价 × 91%
#
# 第5档：
# 首次建仓价 × 88%
# ============================================================

def calculate_levels(first_price):

    levels = []

    for i in range(5):

        level = (
            first_price
            * (
                1
                - ADD_STEP * i
            )
        )

        levels.append(level)

    return levels


# ============================================================
# 计算当前应该处于哪一档
# ============================================================

def determine_stage(
    current_price,
    first_price
):

    levels = calculate_levels(
        first_price
    )

    stage = 1

    for i in range(1, 5):

        if current_price <= levels[i]:

            stage = i + 1

    return stage, levels


# ============================================================
# 生成信号
# ============================================================

def generate_signal(
    df,
    invested_amount,
    first_price
):

    trend = calculate_trend(
        df
    )

    current_price = (
        trend["current"]
    )

    # --------------------------------------------------------
    # 尚未建仓
    # --------------------------------------------------------

    if invested_amount <= 0:

        # 如果趋势强或者震荡
        if trend["trend"] in [
            "强势",
            "震荡"
        ]:

            action = "首次建仓"

            amount = TRANCHE_AMOUNT

            status = "🟢 可以建仓"

        else:

            action = "等待"

            amount = 0

            status = "🟡 趋势偏弱，暂缓建仓"

        return {

            "status": status,

            "action": action,

            "amount": amount,

            "current_price":
                current_price,

            "trend":
                trend["trend"],

            "ma20":
                trend["ma20"],

            "ma60":
                trend["ma60"],

            "stage": 0,

            "next_price":
                current_price

        }

    # --------------------------------------------------------
    # 已经建仓
    # --------------------------------------------------------

    stage, levels = (
        determine_stage(
            current_price,
            first_price
        )
    )

    # 当前已经投入几档
    current_stage = int(
        round(
            invested_amount
            / TRANCHE_AMOUNT
        )
    )

    if current_stage < 1:

        current_stage = 1

    if current_stage > 5:

        current_stage = 5

    # --------------------------------------------------------
    # 满仓
    # --------------------------------------------------------

    if current_stage >= 5:

        return {

            "status":
                "🟢 已完成全部建仓",

            "action":
                "持有",

            "amount":
                0,

            "current_price":
                current_price,

            "trend":
                trend["trend"],

            "ma20":
                trend["ma20"],

            "ma60":
                trend["ma60"],

            "stage":
                5,

            "next_price":
                None

        }

    # --------------------------------------------------------
    # 下一档价格
    # --------------------------------------------------------

    next_stage = current_stage + 1

    next_price = levels[
        next_stage - 1
    ]

    # --------------------------------------------------------
    # 是否已经达到下一档价格
    # --------------------------------------------------------

    if current_price <= next_price:

        # 弱趋势过滤
        if trend["trend"] == "弱势":

            return {

                "status":
                    "🔴 风险过滤",

                "action":
                    "暂停加仓",

                "amount":
                    0,

                "current_price":
                    current_price,

                "trend":
                    trend["trend"],

                "ma20":
                    trend["ma20"],

                "ma60":
                    trend["ma60"],

                "stage":
                    current_stage,

                "next_price":
                    next_price

            }

        return {

            "status":
                "🟢 达到加仓条件",

            "action":
                "加仓",

            "amount":
                TRANCHE_AMOUNT,

            "current_price":
                current_price,

            "trend":
                trend["trend"],

            "ma20":
                trend["ma20"],

            "ma60":
                trend["ma60"],

            "stage":
                current_stage,

            "next_price":
                next_price

        }

    # --------------------------------------------------------
    # 尚未达到下一档
    # --------------------------------------------------------

    return {

        "status":
            "🟡 等待下一档",

        "action":
            "持有",

        "amount":
            0,

        "current_price":
            current_price,

        "trend":
            trend["trend"],

        "ma20":
            trend["ma20"],

        "ma60":
            trend["ma60"],

        "stage":
            current_stage,

        "next_price":
            next_price

    }


# ============================================================
# 打印单只ETF
# ============================================================

def print_result(
    code,
    name,
    result,
    invested_amount,
    first_price
):

    print()

    print("=" * 60)

    print(
        f"{code} {name}"
    )

    print("=" * 60)

    print(
        f"最新价格："
        f"{result['current_price']:.4f}"
    )

    print(
        f"20日均线："
        f"{result['ma20']:.4f}"
        if not pd.isna(result["ma20"])
        else "20日均线：数据不足"
    )

    print(
        f"60日均线："
        f"{result['ma60']:.4f}"
        if not pd.isna(result["ma60"])
        else "60日均线：数据不足"
    )

    print(
        f"趋势状态："
        f"{result['trend']}"
    )

    print()

    print(
        f"当前投入："
        f"{invested_amount:,.0f} / "
        f"{INITIAL_CAPITAL:,.0f} 元"
    )

    if first_price > 0:

        print(
            f"首次建仓价格："
            f"{first_price:.4f}"
        )

    print(
        f"当前建仓档位："
        f"{result['stage']} / 5"
    )

    print()

    print(
        f"模型状态："
        f"{result['status']}"
    )

    print(
        f"操作建议："
        f"{result['action']}"
    )

    if result["amount"] > 0:

        print(
            f"建议金额："
            f"{result['amount']:,.0f} 元"
        )

    if result["next_price"] is not None:

        print(
            f"下一档触发价格："
            f"{result['next_price']:.4f}"
        )

        distance = (
            result["current_price"]
            / result["next_price"]
            - 1
        ) * 100

        print(
            f"距离下一档："
            f"{distance:.2f}%"
        )


# ============================================================
# 主程序
# ============================================================

def main():

    print()

    print("=" * 60)

    print(
        "ETF V7.4 实盘执行信号"
    )

    print("=" * 60)

    print()

    print(
        "核心策略：159581 红利ETF"
    )

    print(
        "总资金：100,000 元"
    )

    print(
        "单次建仓/加仓：20,000 元"
    )

    print(
        "加仓间距：3%"
    )

    print()

    # ========================================================
    # 注意
    #
    # 第一次运行时：
    #
    # invested_amount = 0
    # first_price = 0
    #
    # 如果以后已经实际买入：
    #
    # 修改下面两个数字。
    # ========================================================

    invested_amount = 0

    first_price = 0


    # ========================================================
    # 读取159581
    # ========================================================

    df = load_etf(
        MAIN_CODE
    )

    if df is None:

        print()

        print(
            "❌ 无法读取159581数据"
        )

        return

    # ========================================================
    # 生成信号
    # ========================================================

    result = generate_signal(
        df,
        invested_amount,
        first_price
    )

    # ========================================================
    # 输出
    # ========================================================

    print_result(
        MAIN_CODE,
        MAIN_NAME,
        result,
        invested_amount,
        first_price
    )

    # ========================================================
    # 其他ETF只作为观察
    # ========================================================

    print()

    print("=" * 60)

    print(
        "其他ETF观察"
    )

    print("=" * 60)

    for code, name in [
        ("159209", "红利质量ETF"),
        ("159399", "现金流ETF")
    ]:

        other_df = load_etf(
            code
        )

        if other_df is None:

            continue

        other_trend = calculate_trend(
            other_df
        )

        print()

        print(
            f"{code} {name}"
        )

        print(
            f"最新价格："
            f"{other_trend['current']:.4f}"
        )

        print(
            f"趋势："
            f"{other_trend['trend']}"
        )

        print(
            "当前V7.3评级：暂缓"
        )

    # ========================================================
    # 完成
    # ========================================================

    print()

    print("=" * 60)

    print(
        "V7.4执行信号生成完成"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
