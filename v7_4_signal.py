import os
import pandas as pd
import numpy as np

# ============================================================
# ETF V7.4 实盘执行信号
# ============================================================

DATA_DIR = "data"

INITIAL_CAPITAL = 100000
TRANCHE_AMOUNT = 20000

# 当前主策略
MAIN_CODE = "159581"
MAIN_NAME = "红利ETF"

# V7.3样本外验证得到的主要加仓参数
ADD_STEP = 0.03


# ============================================================
# 读取本地ETF数据
# ============================================================

def load_etf(code):

    filename = f"etf_{code}_signals.csv"

    path = os.path.join(
        DATA_DIR,
        filename
    )

    print(f"读取本地数据：{path}")

    if not os.path.exists(path):

        print(f"❌ 找不到数据文件：{path}")

        return None

    try:

        df = pd.read_csv(path)

        if df.empty:

            print("❌ 数据为空")

            return None

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

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

        print(
            f"成功读取 {len(df)} 条数据"
        )

        print(
            f"最新日期："
            f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}"
        )

        return df

    except Exception as e:

        print(
            f"❌ 读取数据失败：{e}"
        )

        return None


# ============================================================
# 趋势计算
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
# 计算5档价格
#
# 第1档 = 首次建仓价格
# 第2档 = 首次价格 × 97%
# 第3档 = 首次价格 × 94%
# 第4档 = 首次价格 × 91%
# 第5档 = 首次价格 × 88%
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
# 生成首次建仓后的价格计划
# ============================================================

def build_price_plan(first_price):

    levels = calculate_levels(
        first_price
    )

    return {

        "level1": levels[0],

        "level2": levels[1],

        "level3": levels[2],

        "level4": levels[3],

        "level5": levels[4]

    }


# ============================================================
# 判断当前档位
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
# 生成交易信号
# ============================================================

def generate_signal(
    df,
    invested_amount,
    first_price
):

    trend = calculate_trend(
        df
    )

    current_price = trend[
        "current"
    ]

    # ========================================================
    # 情况一：完全没有建仓
    # ========================================================

    if invested_amount <= 0:

        if trend["trend"] in [
            "强势",
            "震荡"
        ]:

            return {

                "status":
                    "🟢 可以建仓",

                "action":
                    "首次建仓",

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
                    0,

                "next_price":
                    None,

                "price_plan":
                    None

            }

        else:

            return {

                "status":
                    "🟡 暂缓建仓",

                "action":
                    "等待",

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
                    0,

                "next_price":
                    None,

                "price_plan":
                    None

            }

    # ========================================================
    # 情况二：已经建仓
    # ========================================================

    if first_price <= 0:

        return {

            "status":
                "⚠️ 缺少首次建仓价格",

            "action":
                "无法计算加仓",

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
                0,

            "next_price":
                None,

            "price_plan":
                None

        }

    # ========================================================
    # 计算价格档位
    # ========================================================

    stage, levels = determine_stage(
        current_price,
        first_price
    )

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

    # ========================================================
    # 已经满仓
    # ========================================================

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
                None,

            "price_plan":
                levels

        }

    # ========================================================
    # 下一档
    # ========================================================

    next_stage = current_stage + 1

    next_price = levels[
        next_stage - 1
    ]

    # ========================================================
    # 已经跌到下一档
    # ========================================================

    if current_price <= next_price:

        # 强趋势 / 震荡：允许加仓
        if trend["trend"] in [
            "强势",
            "震荡"
        ]:

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
                    next_price,

                "price_plan":
                    levels

            }

        # 弱势：暂停加仓
        else:

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
                    next_price,

                "price_plan":
                    levels

            }

    # ========================================================
    # 尚未达到下一档
    # ========================================================

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
            next_price,

        "price_plan":
            levels

    }


# ============================================================
# 输出首次建仓计划
# ============================================================

def print_first_entry_plan(
    current_price
):

    levels = calculate_levels(
        current_price
    )

    print()

    print(
        "【首次建仓后的完整价格计划】"
    )

    print()

    print(
        f"第1档："
        f"{levels[0]:.4f}"
        f" → 20,000元"
    )

    print(
        f"第2档："
        f"{levels[1]:.4f}"
        f" → 20,000元"
    )

    print(
        f"第3档："
        f"{levels[2]:.4f}"
        f" → 20,000元"
    )

    print(
        f"第4档："
        f"{levels[3]:.4f}"
        f" → 20,000元"
    )

    print(
        f"第5档："
        f"{levels[4]:.4f}"
        f" → 20,000元"
    )


# ============================================================
# 输出结果
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

    if not pd.isna(
        result["ma20"]
    ):

        print(
            f"20日均线："
            f"{result['ma20']:.4f}"
        )

    else:

        print(
            "20日均线：数据不足"
        )

    if not pd.isna(
        result["ma60"]
    ):

        print(
            f"60日均线："
            f"{result['ma60']:.4f}"
        )

    else:

        print(
            "60日均线：数据不足"
        )

    print(
        f"趋势状态："
        f"{result['trend']}"
    )

    print()

    print(
        f"当前投入："
        f"{invested_amount:,.0f}"
        f" / "
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

    # ========================================================
    # 尚未建仓
    # ========================================================

    if (
        invested_amount <= 0
        and result["action"] == "首次建仓"
    ):

        print()

        print_first_entry_plan(
            result["current_price"]
        )

        print()

        print(
            "⚠️ 上述第1档价格只是当前参考价。"
        )

        print(
            "实际首次成交价格确定后，"
        )

        print(
            "后续4档价格应以实际成交价重新计算。"
        )

        return

    # ========================================================
    # 已建仓
    # ========================================================

    if result["next_price"] is not None:

        print()

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

    # ========================================================
    # 输出完整价格计划
    # ========================================================

    if result["price_plan"] is not None:

        print()

        print(
            "【当前价格计划】"
        )

        for i, price in enumerate(
            result["price_plan"],
            start=1
        ):

            print(
                f"第{i}档："
                f"{price:.4f}"
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
        "主策略：159581 红利ETF"
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
    # 当前真实账户状态
    #
    # 第一次运行：
    #
    # invested_amount = 0
    # first_price = 0
    #
    # 当你实际买入以后：
    #
    # invested_amount
    # 改成实际投入金额
    #
    # first_price
    # 改成第一次实际成交价格
    #
    # ========================================================

    invested_amount = 0

    first_price = 0

    # ========================================================
    # 主ETF
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

    result = generate_signal(
        df,
        invested_amount,
        first_price
    )

    print_result(
        MAIN_CODE,
        MAIN_NAME,
        result,
        invested_amount,
        first_price
    )

    # ========================================================
    # 其他ETF观察
    # ========================================================

    print()

    print("=" * 60)

    print(
        "其他ETF观察"
    )

    print("=" * 60)

    for code, name in [

        (
            "159209",
            "红利质量ETF"
        ),

        (
            "159399",
            "现金流ETF"
        )

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

    print()

    print("=" * 60)

    print(
        "V7.4执行信号生成完成"
    )

    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()
