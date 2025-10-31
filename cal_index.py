# %%
'''
* 전략 지역구 조사 데이터를 수집·정리·구조화하여 분석에 필요한 형태로 집계
전체 구조
1) 작업 경로 이동: 분석할 데이터가 위치한 폴더로 작업 디렉터리를 변경합니다.
2) 파일 목록 불러오기: 집계 대상 데이터파일의 목록과 경로를 불러옵니다.
3)정당 라벨 불러오기 및 매핑: 파일별로 원본 정당명과 표준 라벨 매핑을 불러오고, 일부 파일 쌍은 자동 보정합니다.
4) 집계 함수 실행: 파일·정당·지역구 정보를 토대로, 10개 전략 지역구만 추출해서 정당별 집계·비율 산출을 wide/long 형태로 반환합니다.
5) 결과 데이터 저장: 합쳐진 결과물을 wide/long 형태 csv 파일로 저장합니다.
6) 저장 완료 출력: 과정이 성공적으로 끝나면 경로를 안내합니다.

'''

import pandas as pd
from pathlib import Path
import os

# 원하는 폴더 경로
work_dir = r"D:\에스티아이\2025 프로젝트 - 전략지역구 조사\test"
os.chdir(work_dir)


# 1. 파일 목록 불러오기
def load_file_list(path="data2/file_list.csv") -> dict:
    df = pd.read_csv(path)
    files = dict(zip(df["file_name"], df["file_path"]))
    return files

# 2-1. 정당 분류 딕셔너리 불러오기 (예: CSV → dict)
def load_party_labels(path="party_labels.csv") -> dict:
    df = pd.read_csv(path)
    party_labels = (
        df.groupby("file_name")
          .apply(lambda x: dict(zip(x["party_name"], x["label"])))
          .to_dict()
    )

    # S → G 매핑
    party_labels["2016_G_na_pro"]   = party_labels["2016_S_na_pro"]
    party_labels["2017_G_president"] = party_labels["2017_S_president"]
    party_labels["2020_G_na_pro"]   = party_labels["2020_S_na_pro"]
    party_labels["2022_G_president"] = party_labels["2022_S_president"]
    party_labels["2024_G_na_pro"]   = party_labels["2024_S_na_pro"]
    party_labels["2025_G_president"] = party_labels["2025_S_president"]

    return party_labels


# 3. 집계 함수
import pandas as pd
import re

def make_vote_trend(files: dict, party_labels: dict, region_map: dict):
    vote_trend_raw = {}
    vote_trend = {}

    valid_codes = set(region_map.keys())   # 우리가 원하는 10개 코드만 (int)

    for fname, fpath in files.items():
        df = pd.read_csv(fpath)
        # 컬럼 정리
        df.columns = (df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False))

        # 라벨 사전
        labels = party_labels.get(fname, {})
        labels = {str(k).strip().replace("\ufeff", ""): v for k, v in labels.items()}

        # --- 코드 컬럼 결정 + 통일 ---
        if "지역구코드" in df.columns:
            code_col = "지역구코드"
        elif "선거구코드" in df.columns:
            code_col = "선거구코드"
        else:
            raise ValueError(f"{fname} 파일에 지역구코드/선거구코드 컬럼이 없음")

        # 무조건 '지역구코드'로 이름 통일
        df = df.rename(columns={code_col: "지역구코드"})
        code_col = "지역구코드"

        # ── (중요) 코드 정규화: NaN/'-' 제거 → 숫자 → int ─────────────────
        code_raw = df[code_col].astype(str).str.strip()
        code_raw = code_raw.where(code_raw != "-", None)
        code_num = pd.to_numeric(code_raw, errors="coerce")
        df = df.loc[code_num.notna()].copy()          # NaN 코드 행 제거
        df[code_col] = code_num.loc[code_num.notna()].astype(int)
        # ────────────────────────────────────────────────────────────────

        # 10개 지역만 필터
        df = df[df[code_col].isin(valid_codes)].copy()

        # # (선택) 필터 후 행수 체크
        # print(f"[DEBUG] {fname} after code filter: {len(df)} rows")

        # 집계: 정당 컬럼만 숫자 변환(쉼표/공백 제거) 후 라벨별 누적
        agg_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if col in labels:
                series_num = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                    errors="coerce"
                ).fillna(0)

                label = labels[col] if pd.notna(labels[col]) else "기타"
                if label not in agg_df.columns:
                    agg_df[label] = series_num
                else:
                    agg_df[label] = agg_df[label] + series_num

        # 코드 붙여서 지역구 단위 집계
        agg_df[code_col] = df[code_col].values
        agg_df = agg_df.groupby(code_col, as_index=False).sum(min_count=1)

        # 빠진 컬럼 보정
        for c in ["민주","보수","진보","기타","진보당"]:
            if c not in agg_df.columns:
                agg_df[c] = 0

        # 지역/선거명
        agg_df["region"]   = agg_df[code_col].map(region_map)
        agg_df["election"] = re.sub(r"_[SG]_", "_", fname)

        # wide 저장
        vote_trend_raw[fname] = agg_df.copy()

        # long 변환
        long_df = agg_df.melt(
            id_vars=["region", code_col, "election"],
            value_vars=["민주","보수","진보","기타","진보당"],
            var_name="label", value_name="votes"
        )

        # ✅ (필수1) 진보 + 진보당 합치기
        if (long_df["label"] == "진보당").any():
            long_df.loc[long_df["label"]=="진보", "votes"] += \
                long_df.loc[long_df["label"]=="진보당", "votes"].values
            long_df = long_df[long_df["label"]!="진보당"]

        # ✅ (필수2) prop 계산
        long_df["계"] = long_df.groupby([code_col, "election"])["votes"].transform("sum")
        long_df["prop"] = (long_df["votes"] / long_df["계"].where(long_df["계"]>0, other=1)) * 100

        vote_trend[fname] = long_df[["region", code_col, "election", "label", "votes", "prop"]]

    return vote_trend_raw, vote_trend



# %%
# 4. 실행 시 데이터 준비

# --- 실행 코드 ---
files = load_file_list("data2/file_list.csv")
party_labels = load_party_labels("data2/party_labels.csv")

# 지역구 코드 매핑 (예시: 실제 딕셔너리 전달)
region_map = {
    2411: "서울 강서구병",
    2412: "서울 관악구을",
    2413: "서울 구로구갑",
    2414: "서울 서대문구갑",
    2415: "서울 은평구갑",
    2421: "경기 고양시을",
    2422: "경기 부천시을",
    2423: "경기 수원시을",
    2424: "경기 평택시을",
    2425: "경기 화성시을"
}

vote_trend_raw, vote_trend = make_vote_trend(files, party_labels, region_map)

# --- 결과물 저장 ---
os.makedirs("output", exist_ok=True)

# wide 결과 합치기
vote_trend_raw_all = pd.concat(vote_trend_raw.values(), ignore_index=True)
vote_trend_raw_all.to_csv("output/vote_trend_raw.csv", index=False, encoding="utf-8-sig")

# long 결과 합치기
vote_trend_all = pd.concat(vote_trend.values(), ignore_index=True)
vote_trend_all.to_csv("output/vote_trend.csv", index=False, encoding="utf-8-sig")

print("저장 완료 ✅")
print("output/vote_trend_raw.csv (wide)")
print("output/vote_trend.csv (long)")

# %%
# 1) 2024, 2025 선거 제외
df = vote_trend_all[
    ~vote_trend_all["election"].str.contains("2024|2025")
]

# 2) label == '진보' 만 선택
df_jinbo = df[df["label"] == "진보"]

# 3) 지역구별 평균 prop 계산
result = (
    df_jinbo
    .groupby(["region", "지역구코드"])["prop"]
    .mean()
    .reset_index()
    .rename(columns={"prop": "진보_prop_mean"})
)

# 4) 출력 확인
print(result.head())

# 저장 원하면:
# result.to_csv("output/jinbo_prop_mean.csv", index=False, encoding="utf-8-sig")


# %%
df_filtered = vote_trend_all[
    ~vote_trend_all["election"].str.contains("2024|2025")   #2024,2025 제외
]

# # 2024, 2025도 포함
# df_filtered = vote_trend_all.copy()

# 진보만 추출
jinbo_df = df_filtered[df_filtered["label"] == "진보"]

# 지역구-선거 기준 평균
jinbo_avg = (
    jinbo_df.groupby(["region", "지역구코드"])["prop"]
    .mean()
    .reset_index()
)

# 📌 빠진 지역이 있으면 region_map 기준으로 채워 넣기
all_regions = pd.DataFrame(list(region_map.items()), columns=["지역구코드", "region"])
jinbo_avg = all_regions.merge(jinbo_avg, on=["지역구코드","region"], how="left")

print(jinbo_avg)


# %%
# 🔹 선거 순서 (원하시는 순서 지정)
election_order = [
    "2016_na_pro", "2017_president", "2018_loc_gov", "2018_loc_pro",
    "2020_na_pro", "2022_president","2022_loc_gov", "2022_loc_pro", 
    "2024_na_pro", "2025_president"
]

# 1위 계열 찾기
first_rank = (
    vote_trend_all
    .sort_values(["지역구코드", "election", "votes"], ascending=[True, True, False])
    .groupby(["지역구코드", "election"])
    .first()
    .reset_index()
)

# 선거 순서를 categorical 로 지정
first_rank["election"] = pd.Categorical(first_rank["election"], categories=election_order, ordered=True)

# 지역구별 pivot: 각 선거에서 1위 label
pivot_df = first_rank.pivot(index=["지역구코드","region"], columns="election", values="label")

# 문자열로 합치기
pivot_df["trend"] = pivot_df[election_order].agg("-".join, axis=1)

# 🔹 1위 변경 횟수 계산
def count_changes(seq):
    vals = [v for v in seq if pd.notna(v)]
    return sum(x != y for x, y in zip(vals, vals[1:]))

pivot_df["changes"] = pivot_df[election_order].apply(count_changes, axis=1)

# 정리
result = pivot_df.reset_index()[["지역구코드","region","trend","changes"]]

print(result)


# %%


# 1-2위 격차 계산
def top2_gap(df):
    sorted_df = df.sort_values("prop", ascending=False)
    if len(sorted_df) < 2:
        return None
    return sorted_df.iloc[0]["prop"] - sorted_df.iloc[1]["prop"]

gaps = (
    vote_trend_all
    .groupby(["지역구코드","region","election"])
    .apply(top2_gap)
    .reset_index(name="gap")
)

# 지역별 평균 격차(%p)
avg_gap = (
    gaps.groupby(["지역구코드","region"])["gap"]
    .mean()
    .reset_index(name="avg_gap")
)

print(avg_gap)


# %%
gaps


