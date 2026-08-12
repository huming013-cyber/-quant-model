import os
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# 基本设置
# ============================================================

LIST_FILE = "stock_list.csv"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 把 A 股代码转换成 Yahoo Finance 代码
# ============================================================

def convert_code(asset_type, code):
    code = str(code).strip()

    # ETF
    if asset_type.lower() == "etf":
        if code.startswith("15") or code.startswith("16"):
            return code + ".SZ"
        elif code.startswith("51") or code.startswith("56") or code.startswith("58"):
            return code + ".SS"
        else:
            return code

    # 股票
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

    if len(df) < 60:
        return None

    close = df["Close"]

    # 均线
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    # 20日最高价
    df["HIGH20"] = close.rolling(20).max()

    # 当前回撤
    df["DRAWDOWN"] = close / df["HIGH20"] - 1

    # 20日收益率
    df["RETURN20"] = close.pct_change(20)

    # 60日收益率
    df["RETURN60"] = close.pct_change(60)

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 成交量变化
    if "Volume" in df.columns:
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["VOLUME_RATIO"] = df["Volume"] / df["VOL20"]

    return df


# ============================================================
# 计算量化评分
# ============================================================

def calculate_score(df):

    latest = df.iloc[-1]

    score = 0

    # -------------------------
    # 趋势
    # -------------------------

    if latest["Close"] > latest["MA20"]:
        score += 15

    if latest["Close"] > latest["MA60"]:
        score += 15

    if latest["MA20"] > latest["MA60"]:
        score += 10

    # -------------------------
    # 中短期动量
    # -------------------------

    if latest["RETURN20"] > 0:
        score += 10

    if latest["RETURN60"] > 0:
        score += 10

    # -------------------------
    # MACD
    # -------------------------

    if latest["MACD"] > latest["SIGNAL"]:
        score += 10

    # -------------------------
    # RSI
    # -------------------------

    rsi = latest["RSI14"]

    if 45 <= rsi <= 70:
        score += 10

    elif 30 <= rsi < 45:
        score += 5

    # -------------------------
    # 回撤
    # -------------------------

    drawdown = latest["DRAWDOWN"]

    if drawdown >= -0.05:
        score += 10

    elif drawdown >= -0.10:
        score += 5

    # -------------------------
    # 成交量
    # -------------------------

    if "VOLUME_RATIO" in latest.index:

        volume_ratio = latest["VOLUME_RATIO"]

        if volume_ratio > 1:
            score += 5

    return min(score, 100)


# ============================================================
# 分析单个股票 / ETF
# ============================================================

def analyze_asset(asset_type, code, name):

    yahoo_code = convert_code(asset_type, code)

    print(f"\n正在获取：{name} ({code})")
    print(f"Yahoo代码：{yahoo_code}")

    try:

        df = yf.download(
            yahoo_code,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            print(f"❌ {name}：没有获取到数据")
            return None

        # 某些版本的 yfinance 会产生多层列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Close"]

        for col in required:
            if col not in df.columns:
                print(f"❌ {name}：缺少 {col} 数据")
                return None

        df = df.dropna(subset=["Close"])

        if len(df) < 60:
            print(f"❌ {name}：历史数据不足")
            return None

        df = calculate_indicators(df)

        if df is None:
            print(f"❌ {name}：无法计算指标")
            return None

        latest = df.iloc[-1]

        score = calculate_score(df)

        close = float(latest["Close"])

        ma20 = float(latest["MA20"])
        ma60 = float(latest["MA60"])

        return {
            "type": asset_type,
            "code": code,
            "name": name,
            "yahoo_code": yahoo_code,
            "date": str(df.index[-1].date()),
            "price": round(close, 2),
            "MA20": round(ma20, 2),
            "MA60": round(ma60, 2),
            "RSI14": round(float(latest["RSI14"]), 2),
            "MACD": round(float(latest["MACD"]), 4),
            "return20": round(float(latest["RETURN20"]) * 100, 2),
            "return60": round(float(latest["RETURN60"]) * 100, 2),
            "drawdown": round(float(latest["DRAWDOWN"]) * 100, 2),
            "score": score
        }

    except Exception as e:

        print(f"❌ {name} 获取失败：{e}")

        return None


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 60)
    print("A股 + ETF 量化分析系统")
    print("=" * 60)

    if not os.path.exists(LIST_FILE):

        print(f"❌ 找不到 {LIST_FILE}")

        return

    stocks = pd.read_csv(LIST_FILE)

    required_columns = ["type", "code", "name"]

    for col in required_columns:

        if col not in stocks.columns:

            print(f"❌ stock_list.csv 缺少字段：{col}")

            return

    results = []

    total = len(stocks)

    print(f"\n共发现 {total} 个分析对象")

    for index, row in stocks.iterrows():

        asset_type = str(row["type"]).strip()
        code = str(row["code"]).strip()
        name = str(row["name"]).strip()

        result = analyze_asset(
            asset_type,
            code,
            name
        )

        if result is not None:

            results.append(result)

        time.sleep(1)

    # ========================================================
    # 生成结果
    # ========================================================

    if len(results) == 0:

        print("\n❌ 所有数据都获取失败")
        print("程序不会生成错误的结果文件")

        return

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="score",
        ascending=False
    )

    result_df.insert(
        0,
        "rank",
        range(1, len(result_df) + 1)
    )

    output_file = os.path.join(
        DATA_DIR,
        "quant_result.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("量化分析完成")
    print("=" * 60)

    print(result_df.to_string(index=False))

    print("\n结果文件：")
    print(output_file)

    print("\n成功分析：", len(results))
    print("名单总数：", total)
    print("=" * 60)


if __name__ == "__main__":
    main()
