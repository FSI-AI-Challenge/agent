# pip install yfinance pandas numpy python-dateutil
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

import re, json
from typing import Dict, Any


def extract_json(text: str) -> Dict[str, Any]:
    # ```json ... ``` 블록 우선 추출 → 실패 시 중괄호 첫/끝 매칭
    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if codeblock:
        text = codeblock.group(1)
    else:
        # 가장 바깥 {} 추정
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
    return json.loads(text)

# 1) 한글 종목명 → 야후 파이낸스 티커 매핑 (예시, 필요 시 추가)
NAME_TO_TICKER = {
    "삼성전자": "005930.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "현대차": "005380.KS",
    "LG에너지솔루션": "373220.KS",
    # ... 원하시는 종목 계속 추가
}

def get_month_window(today=None):
    """야후 휴장일을 감안해 최근 거래일 기반 윈도우를 확보합니다."""
    if today is None:
        today = datetime.today()
    start = today - relativedelta(months=1) - timedelta(days=3)  # 버퍼 3일
    end = today + timedelta(days=1)                              # 오늘 포함
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def compute_rate_and_risk(names, name_to_ticker=NAME_TO_TICKER):
    """
    입력: ['삼성전자', 'NAVER', ...]
    출력: {
      "삼성전자": {"rate": float(수익률), "risk": float(최고-최저), "risk_pct": float(%)},
      ...
    }
    """
    start, end = get_month_window()
    result = {}

    # 이름→티커 변환
    missing = [n for n in names if n not in name_to_ticker]
    if missing:
        # 매핑 누락 종목은 스킵 또는 직접 티커 넣을 수 있게 알림
        for n in missing:
            result[n] = {"error": "티커 매핑 없음 (NAME_TO_TICKER에 추가 필요)"}

    tickers = [name_to_ticker[n] for n in names if n in name_to_ticker]
    if not tickers:
        return result

    # 2) 가격 데이터 수집 (일봉)
    df = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False, threads=True, group_by='ticker', progress=False)

    # 단일/다중 티커 모두 처리
    def get_series(tk, col):
        if isinstance(df.columns, pd.MultiIndex):
            return df[(tk, col)].dropna()
        else:
            # 단일 티커인 경우
            return df[col].dropna()

    # 3) 종목별 계산
    for name in names:
        tk = name_to_ticker.get(name)
        if not tk:
            continue

        try:
            close = get_series(tk, "Close")
            high = get_series(tk, "High")
            low  = get_series(tk, "Low")

            if close.empty or high.empty or low.empty:
                result[name] = {"error": "가격 데이터 부족"}
                continue

            # 가장 이른 날(=한 달 전 근처)과 가장 최근 거래일의 종가
            first_close = close.iloc[0]
            last_close  = close.iloc[-1]

            # 수익률
            rate = float((last_close / first_close) - 1)

            # 위험도: 한 달 동안의 (최고 - 최저)
            month_high = float(high.max())
            month_low  = float(low.min())
            risk_abs = month_high - month_low

            # 퍼센트 스프레드(선택): 기준을 "한 달 전 종가"로
            risk_pct = float(risk_abs / first_close)

            result[name] = {
                "rate": round(rate, 6),        # 예: 0.0345 → 3.45%
                "risk": round(risk_abs, 4),    # 절대가격(원)
                "risk_pct": round(risk_pct, 6) # 예: 0.0812 → 8.12%
            }
        except Exception as e:
            result[name] = {"error": f"계산 실패: {e}"}

    return result

def monthly_contribution(target_fv: float,
                         annual_rate_percent: float,
                         months: int,
                         interest_type: str = "compound",   # "compound"(복리) or "simple"(단리)
                         payment_timing: str = "end"         # "end"(후불식) or "begin"(선불식, 선택)
                         ) -> float:
    """
    반환값: 매월 납입해야 할 금액 (원)
    - target_fv: 목표 최종 금액 (원)
    - annual_rate_percent: 연 수익률(%) 예: 6 -> 6%
    - months: 기간(개월)
    - interest_type: "compound"(복리) 또는 "simple"(단리)
    - payment_timing: "end"(후불식, 기본) 또는 "begin"(선불식)
    """
    if months <= 0:
        raise ValueError("months는 1 이상이어야 합니다.")
    if target_fv < 0:
        raise ValueError("목표 금액은 음수일 수 없습니다.")
    i_annual = annual_rate_percent / 100.0
    r = i_annual / 12.0  # 월 이율

    # 무이자 처리
    if abs(r) < 1e-12:
        return target_fv / months

    interest_type = interest_type.lower()
    payment_timing = payment_timing.lower()

    if interest_type == "compound":
        # 복리(월 복리) - 연금종가치(적립식) 공식
        # FV = A * [((1+r)^n - 1) / r]          (후불식, end)
        # 선불식(begin)은 FV = A * [((1+r)^n - 1) / r] * (1+r)
        factor = ((1 + r)**months - 1) / r
        if payment_timing == "begin":
            factor *= (1 + r)
        A = target_fv / factor

    elif interest_type == "simple":
        # 단리(월 단리) - 각 납입금은 남은 개월 수만큼 선형 이자
        # 후불식(end)에서 FV = A * sum_{k=0}^{n-1} (1 + r*k)
        #   = A * [ n + r * n(n-1)/2 ]
        # 선불식(begin)에서는 한 달 더 길게 붙이려면 k를 1..n로 두면 됨:
        #   sum_{k=1}^{n} (1 + r*k) = n + r * n(n+1)/2
        if payment_timing == "end":
            factor = months + r * months * (months - 1) / 2.0
        elif payment_timing == "begin":
            factor = months + r * months * (months + 1) / 2.0
        else:
            raise ValueError("payment_timing은 'end' 또는 'begin'이어야 합니다.")
        A = target_fv / factor
    else:
        raise ValueError("interest_type은 'compound' 또는 'simple'이어야 합니다.")

    return float(A)


# 사용 예시
if __name__ == "__main__":
    names = ["삼성전자", "NAVER", "카카오"]
    out = compute_rate_and_risk(names)
    print(out)


