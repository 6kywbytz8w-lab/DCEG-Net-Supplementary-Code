from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
import pandas as pd
DEFAULT_POLLUTANTS = ('PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3-8h')

def audit_missingness(df: pd.DataFrame, city_col: str, numeric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for city, group in df.groupby(city_col, sort=False, dropna=False):
        for col in numeric_cols:
            missing_count = int(group[col].isna().sum())
            total_count = int(len(group))
            rows.append({city_col: city, 'Feature': col, 'Missing_Count': missing_count, 'Total_Count': total_count, 'Missing_Rate': missing_count / total_count if total_count else 0.0})
    return pd.DataFrame(rows)

def audit_and_impute(df: pd.DataFrame, *, covariate_cols: Sequence[str], date_col: str='Date', city_col: str='City', target_col: str='AQI', pollutant_cols: Sequence[str]=DEFAULT_POLLUTANTS, output_dir: str | Path='imputation_audit', max_fill_gap: int | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    required = [date_col, city_col, target_col, *pollutant_cols, *covariate_cols]
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f'数据缺少必要列: {absent}')
    if not covariate_cols:
        raise ValueError('必须通过 covariate_cols 明确指定允许前向填充的外生协变量')
    if len(set(covariate_cols)) != len(covariate_cols):
        raise ValueError('covariate_cols 中存在重复列名')
    x = df.copy()
    if x[city_col].isna().any():
        raise ValueError('城市字段存在空值')
    if x[date_col].isna().any():
        raise ValueError('日期字段存在空值')
    x[city_col] = x[city_col].astype(str).str.strip()
    if x[city_col].eq('').any():
        raise ValueError('城市字段存在空字符串')
    x[date_col] = pd.to_datetime(x[date_col], errors='coerce')
    if x[date_col].isna().any():
        raise ValueError('日期字段存在无法解析的值')
    x = x.sort_values([city_col, date_col], kind='stable').reset_index(drop=True)
    if x.duplicated([city_col, date_col]).any():
        raise ValueError('存在重复的城市-日期记录，必须先处理后再插补')
    numeric_cols = [col for col in x.columns if col not in (date_col, city_col) and pd.api.types.is_numeric_dtype(x[col])]
    protected = {target_col, *pollutant_cols}
    overlap = protected.intersection(covariate_cols)
    if overlap:
        raise ValueError(f'受保护的 AQI/污染物列不能作为插补协变量: {sorted(overlap)}')
    non_numeric = [col for col in covariate_cols if not pd.api.types.is_numeric_dtype(x[col])]
    if non_numeric:
        raise TypeError(f'指定的协变量必须为数值型: {non_numeric}')
    before = audit_missingness(x, city_col, numeric_cols)
    before.to_excel(output_path / 'missingness_before_fill.xlsx', index=False)
    original_protected = x[list(protected)].copy()
    if max_fill_gap is None:
        x.loc[:, list(covariate_cols)] = x.groupby(city_col, sort=False)[list(covariate_cols)].ffill()
    else:
        if max_fill_gap < 0:
            raise ValueError('max_fill_gap 必须为非负整数或 None')
        x.loc[:, list(covariate_cols)] = x.groupby(city_col, sort=False)[list(covariate_cols)].ffill(limit=max_fill_gap)
    for col in protected:
        if not x[col].equals(original_protected[col]):
            raise AssertionError(f'受保护的真实观测列被修改: {col}')
    after = audit_missingness(x, city_col, numeric_cols)
    after.to_excel(output_path / 'missingness_after_fill.xlsx', index=False)
    summary_rows = []
    for stage, audit in (('Before imputation', before), ('After covariate forward-fill', after)):
        missing_cells = int(audit['Missing_Count'].sum())
        audited_cells = int(audit['Total_Count'].sum())
        summary_rows.append({'Stage': stage, 'Missing_Cells': missing_cells, 'Audited_Cells': audited_cells, 'Missing_Rate_Percent': 100.0 * missing_cells / audited_cells if audited_cells else 0.0})
    pd.DataFrame(summary_rows).to_excel(output_path / 'missingness_imputation_summary.xlsx', index=False)
    return (x, before, after)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 Excel 文件')
    parser.add_argument('--sheet', default='0', help='工作表名称或序号')
    parser.add_argument('--date-col', default='Date')
    parser.add_argument('--city-col', default='City')
    parser.add_argument('--target-col', default='AQI')
    parser.add_argument('--pollutant-cols', default='PM2.5,PM10,SO2,NO2,CO,O3-8h')
    parser.add_argument('--covariate-cols', required=True, help='允许前向填充的外生数值列，逗号分隔；必须显式指定')
    parser.add_argument('--max-fill-gap', type=int, default=None, help='最多连续前向填充行数；默认不限制，建议按数据采样间隔设定')
    parser.add_argument('--output-dir', default='imputation_audit')
    parser.add_argument('--output-file', default='processed_data.xlsx')
    args = parser.parse_args()
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    data = pd.read_excel(args.input, sheet_name=sheet)
    pollutants = tuple((c.strip() for c in args.pollutant_cols.split(',') if c.strip()))
    covariates = tuple((c.strip() for c in args.covariate_cols.split(',') if c.strip()))
    cleaned, _, _ = audit_and_impute(data, covariate_cols=covariates, date_col=args.date_col, city_col=args.city_col, target_col=args.target_col, pollutant_cols=pollutants, output_dir=args.output_dir, max_fill_gap=args.max_fill_gap)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cleaned.to_excel(output_path / args.output_file, index=False)
    print(f'处理完成：{output_path / args.output_file}')
    print('注意：插补后仍缺失的必要输入/标签须在监督样本构造阶段按规则剔除并审计。')
if __name__ == '__main__':
    main()
