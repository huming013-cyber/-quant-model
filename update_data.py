import os
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

LIST_FILE = "stock_list.csv"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# A股代码转换
# ============================================================

def convert_code(asset_type, code):
    code = str(code).strip()

    if asset_type.lower() == "etf":
        if code.startswith(("15", "16")):
            return code + ".SZ"
        elif code.startswith(("51", "56", "58")):
            return code + ".SS"
        return code

    if code.startswith(("60", "68", "51", "58")):
        return code + ".SS"

    if code.startswith(("00", "30", "12", "15", "16")):
        return code + ".SZ"

    return code


# ============================================================
# 计算技术指标
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]

    # 均线
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # 收益率
    df["RET5"] = close.pct_change(5)
    df["RET20"] = close.pct_change(20)
    df["RET60"] = close.pct_change(60)
    df["RET120"] = close.pct_change(120)

    # 20日高点
    df["HIGH20"] = close.rolling(20).max()

    # 120日高点
    df["HIGH120"] = close.rolling(120).max()

    # 回撤
    df["DRAWDOWN20"] = close / df["HIGH20"] - 1
    df["DRAWDOWN120"] = close / df["HIGH120"] - 1

    # ========================================================
    # RSI
    # ========================================================

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # ========================================================
    # MACD
    # ========================================================

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26

    df["SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = df["MACD"] - df["SIGNAL"]

    # ========================================================
    # 成交量
    # ========================================================

    if "Volume" in df.columns:

        df["VOL20"] = df["Volume"].rolling(20).mean()

        df["VOLUME_RATIO"] = (
            df["Volume"] / df["VOL20"]
        )

    else:

        df["VOLUME_RATIO"] = np.nan

    # ========================================================
    # 波动率
    # ========================================================

    daily_return = close.pct_change()

    df["VOLATILITY20"] = (
        daily_return.rolling(20).std()
        * np.sqrt(252)
    )

    return df


# ============================================================
# 趋势评分
# ============================================================

def trend_score(x):

    score = 0

    price = x["Close"]

    if price > x["MA20"]:
        score += 8

    if price > x["MA60"]:
        score += 7

    if x["MA20"] > x["MA60"]:
        score += 5

    if x["MA60"] > x["MA120"]:
        score += 5

    return score


# ============================================================
# 动量评分
# ============================================================

def momentum_score(x):

    score = 0

    r20 = x["RET20"]
    r60 = x["RET60"]
    r120 = x["RET120"]

    if r20 > 0:
        score += 8

    if r20 > 0.05:
        score += 2

    if r60 > 0:
        score += 5

    if r120 > 0:
        score += 5

    return score


# ============================================================
# 回撤评分
# ============================================================

def drawdown_score(x):

    score = 0

    dd20 = x["DRAWDOWN20"]
    dd120 = x["DRAWDOWN120"]

    if dd20 >= -0.03:
        score += 8

    elif dd20 >= -0.08:
        score += 5

    elif dd20 >= -0.15:
        score += 2

    if dd120 >= -0.10:
        score += 7

    elif dd120 >= -0.20:
        score += 4

    return score


# ============================================================
# RSI评分
# ============================================================

def rsi_score(rsi):

    if 50 <= rsi <= 65:
        return 10

    if 45 <= rsi < 50:
        return 8

    if 65 < rsi <= 70:
        return 8

    if 35 <= rsi < 45:
        return 5

    if rsi < 35:
        return 3

    if rsi > 75:
        return 2

    return 5


# ============================================================
# MACD评分
# ============================================================

def macd_score(x):

    score = 0

    if x["MACD"] > x["SIGNAL"]:
        score += 7

    if x["MACD_HIST"] > 0:
        score += 3

    return score


# ============================================================
# 成交量评分
# ============================================================

def volume_score(x):

    ratio = x["VOLUME_RATIO"]

    if pd.isna(ratio):
        return 5

    if 0.8 <= ratio <= 1.8:
        return 8

    if ratio > 1.8:
        return 6

    return 4


# ============================================================
# 稳定性评分
# ============================================================

def stability_score(x):

    volatility = x["VOLATILITY20"]

    if pd.isna(volatility):
        return 5

    if volatility < 0.20:
        return 10

    if volatility < 0.30:
        return 8

    if volatility < 0.40:
        return 6

    if volatility < 0.50:
        return 4

    return 2


# ============================================================
# 综合评分
# ============================================================

def calculate_score(df):

    x = df.iloc[-1]

    trend = trend_score(x)
    momentum = momentum_score(x)
    drawdown = drawdown_score(x)
    rsi = rsi_score(x["RSI14"])
    macd = macd_score(x)
    volume = volume_score(x)
    stability = stability_score(x)

    total = (
        trend
        + momentum
        + drawdown
        + rsi
        + macd
        + volume
        + stability
    )

    return {
        "trend_score": trend,
        "momentum_score": momentum,
        "drawdown_score": drawdown,
        "rsi_score": rsi,
        "macd_score": macd,
        "volume_score": volume,
        "stability_score": stability,
        "score": min(total, 100)
    }


# ============================================================
# 信号判断
# ============================================================

def get_signal(score):

    if score >= 85:
        return "强势", "买入"

    if score >= 75:
        return "偏强", "持有/逢低关注"

    if score >= 60:
        return "中性", "观察"

    if score >= 45:
        return "偏弱", "谨慎"

    return "弱势", "减仓风险"


# ============================================================
# 分析单个标的
# ============================================================

def analyze_asset(asset_type, code, name):

    yahoo_code = convert_code(
        asset_type,
        code
    )

    print()
    print("=" * 60)
    print(f"正在分析：{name}")
    print(f"代码：{code}")
    print(f"Yahoo：{yahoo_code}")

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

            print("❌ 没有获取到数据")

            return None

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:

            print("❌ 缺少Close")

            return None

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 130:

            print("❌ 历史数据不足130个交易日")

            return None

        df = calculate_indicators(df)

        df = df.dropna(
            subset=[
                "MA120",
                "RET120",
                "RSI14",
                "VOLATILITY20"
            ]
        )

        if df.empty:

            print("❌ 指标计算失败")

            return None

        latest = df.iloc[-1]

        scores = calculate_score(df)

        market_state, signal = get_signal(
            scores["score"]
        )

        result = {

            "type": asset_type,

            "code": code,

            "name": name,

            "yahoo_code": yahoo_code,

            "date": str(
                df.index[-1].date()
            ),

            "price": round(
                float(latest["Close"]),
                2
            ),

            "MA20": round(
                float(latest["MA20"]),
                2
            ),

            "MA60": round(
                float(latest["MA60"]),
                2
            ),

            "MA120": round(
                float(latest["MA120"]),
                2
            ),

            "RSI14": round(
                float(latest["RSI14"]),
                2
            ),

            "MACD": round(
                float(latest["MACD"]),
                4
            ),

            "return5": round(
                float(latest["RET5"]) * 100,
                2
            ),

            "return20": round(
                float(latest["RET20"]) * 100,
                2
            ),

            "return60": round(
                float(latest["RET60"]) * 100,
                2
            ),

            "return120": round(
                float(latest["RET120"]) * 100,
                2
            ),

            "drawdown20": round(
                float(latest["DRAWDOWN20"]) * 100,
                2
            ),

            "drawdown120": round(
                float(latest["DRAWDOWN120"]) * 100,
                2
            ),

            "volatility20": round(
                float(latest["VOLATILITY20"]) * 100,
                2
            ),

            "trend_score": scores["trend_score"],

            "momentum_score": scores["momentum_score"],

            "drawdown_score": scores["drawdown_score"],

            "rsi_score": scores["rsi_score"],

            "macd_score": scores["macd_score"],

            "volume_score": scores["volume_score"],

            "stability_score": scores["stability_score"],

            "score": scores["score"],

            "state": market_state,

            "signal": signal
        }

        print(
            f"评分：{scores['score']}"
        )

        print(
            f"状态：{market_state}"
        )

        print(
            f"信号：{signal}"
        )

        return result

    except Exception as e:

        print(
            f"❌ {name} 获取失败：{e}"
        )

        return None


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 60)
    print("A股 + ETF Quant Model V2")
    print("=" * 60)

    if not os.path.exists(LIST_FILE):

        print(
            f"❌ 找不到 {LIST_FILE}"
        )

        return

    stocks = pd.read_csv(
        LIST_FILE
    )

    required = [
        "type",
        "code",
        "name"
    ]

    for column in required:

        if column not in stocks.columns:

            print(
                f"❌ stock_list.csv缺少：{column}"
            )

            return

    results = []

    print(
        f"共发现 {len(stocks)} 个标的"
    )

    for _, row in stocks.iterrows():

        result = analyze_asset(

            str(row["type"]).strip(),

            str(row["code"]).strip(),

            str(row["name"]).strip()
        )

        if result is not None:

            results.append(result)

        time.sleep(1)

    if not results:

        print(
            "❌ 所有数据获取失败"
        )

        return

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 股票排名
    # ========================================================

    stocks_df = result_df[
        result_df["type"].str.lower()
        == "stock"
    ].copy()

    if not stocks_df.empty:

        stocks_df = stocks_df.sort_values(
            "score",
            ascending=False
        )

        stocks_df.insert(
            0,
            "rank",
            range(
                1,
                len(stocks_df) + 1
            )
        )

        stocks_file = os.path.join(
            DATA_DIR,
            "stock_result.csv"
        )

        stocks_df.to_csv(
            stocks_file,
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # ETF排名
    # ========================================================

    etf_df = result_df[
        result_df["type"].str.lower()
        == "etf"
    ].copy()

    if not etf_df.empty:

        etf_df = etf_df.sort_values(
            "score",
            ascending=False
        )

        etf_df.insert(
            0,
            "rank",
            range(
                1,
                len(etf_df) + 1
            )
        )

        etf_file = os.path.join(
            DATA_DIR,
            "etf_result.csv"
        )

        etf_df.to_csv(
            etf_file,
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 总排名
    # ========================================================

    all_df = result_df.sort_values(
        "score",
        ascending=False
    )

    all_df.insert(
        0,
        "rank",
        range(
            1,
            len(all_df) + 1
        )
    )

    all_file = os.path.join(
        DATA_DIR,
        "quant_result.csv"
    )

    all_df.to_csv(
        all_file,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("V2量化分析完成")
    print("=" * 60)

    print(
        all_df[
            [
                "rank",
                "type",
                "name",
                "score",
                "state",
                "signal"
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"成功分析：{len(results)}"
    )

    print(
        f"结果已保存到：{DATA_DIR}/"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
