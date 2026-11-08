from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
MODELS = [('DualBranch-CEGNetStyle', 'Single DCEG model'), ('DCEGNet_BiasCorrectedEnsemble', 'DCEG bias-corrected ensemble'), ('DCEGNet_RidgeFusion_BiasCorrected', 'Full pipeline: DCEG + bias correction + Ridge fusion'), ('Persistence', 'Persistence baseline'), ('MA7', '7-day moving-average baseline')]
EXCLUDED_MODELS = {'DCEGNet_RawEnsemble'}
THRESHOLDS = (100, 150, 200)
PHASES = ('验证集', '测试集')

def _validated_pair(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        raise ValueError('Empty or misaligned y/p arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('y/p contains NaN or infinite values')
    return (y, p)

def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = _validated_pair(y, p)
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        raise ValueError('R2/NSE are undefined because target variance is zero')
    nz = np.abs(y) > 1e-12
    return {'N': int(y.size), 'R2': float(1.0 - np.sum(err ** 2) / sst), 'NSE': float(1.0 - np.sum(err ** 2) / sst), 'RMSE': float(np.sqrt(np.mean(err ** 2))), 'MAE': float(np.mean(np.abs(err))), 'MAPE_percent': float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else np.nan, 'MBE': float(np.mean(err)), 'sMAPE_percent': float(np.mean(2.0 * np.abs(err) / (np.abs(y) + np.abs(p) + 1e-12)) * 100)}

def event_metrics(y: np.ndarray, p: np.ndarray, threshold: int) -> dict:
    y, p = _validated_pair(y, p)
    yt = y >= threshold
    yp = p >= threshold
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom else 0.0
    far = fp / (fp + tn) if fp + tn else np.nan
    return {'Threshold': f'>={threshold}', 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Precision': precision, 'Recall': recall, 'EHR': recall, 'F1': f1, 'Accuracy': (tp + tn) / y.size, 'FAR': far, 'CSI': tp / (tp + fp + fn) if tp + fp + fn else np.nan}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='Path to actual_results_source.xlsx')
    ap.add_argument('--output', type=Path, default=Path('province_single_DCEG_vs_full_pipeline.xlsx'))
    args = ap.parse_args()
    pred = pd.read_excel(args.input, sheet_name='province_predictions')
    required = {'Phase', 'Model', 'Date', 'True_Prov_AQI', 'Pred_Prov_AQI'}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f'province_predictions missing columns: {sorted(missing)}')
    pred = pred.copy()
    if pred[['Phase', 'Model', 'Date']].isna().any().any():
        raise ValueError('Phase/Model/Date contains missing values')
    pred['Phase'] = pred['Phase'].astype(str).str.strip()
    pred['Model'] = pred['Model'].astype(str).str.strip()
    pred['Date'] = pd.to_datetime(pred['Date'], errors='coerce').dt.normalize()
    if pred['Date'].isna().any():
        raise ValueError('Date contains invalid or missing values')
    pred = pred[~pred['Model'].isin(EXCLUDED_MODELS)].copy()
    known_models = {name for name, _ in MODELS}
    unknown = sorted(set(pred['Model']) - known_models)
    if unknown:
        raise ValueError(f'Unexpected model names in province_predictions after excluding {sorted(EXCLUDED_MODELS)}: {unknown}. Check model labels explicitly.')
    metrics_rows, event_rows, target_rows = ([], [], [])
    present_phases = [phase for phase in PHASES if (pred['Phase'] == phase).any()]
    if not present_phases:
        raise ValueError(f'No recognized phases found. Expected one of {PHASES}')
    for phase in present_phases:
        phase_df = pred[pred['Phase'] == phase].copy()
        present_models = set(phase_df['Model'])
        expected_models = known_models
        missing_models = sorted(expected_models - present_models)
        if missing_models:
            raise ValueError(f'{phase} is missing required model predictions: {missing_models}')
        reference = None
        for model, label in MODELS:
            d = phase_df[phase_df['Model'] == model].sort_values('Date').copy()
            if d['Date'].duplicated().any():
                raise ValueError(f'Duplicate dates for {phase}/{model}')
            if d[['True_Prov_AQI', 'Pred_Prov_AQI']].isna().any().any():
                raise ValueError(f'Missing target/prediction values for {phase}/{model}')
            try:
                y = pd.to_numeric(d['True_Prov_AQI'], errors='raise').to_numpy(float)
                p = pd.to_numeric(d['Pred_Prov_AQI'], errors='raise').to_numpy(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Non-numeric AQI values for {phase}/{model}') from exc
            keys = d['Date'].tolist()
            if reference is None:
                reference = (keys, y)
            elif keys != reference[0] or not np.allclose(y, reference[1], atol=1e-08, rtol=0):
                raise ValueError(f'Date/true-target mismatch for {phase}/{model}')
            row = {'Phase': phase, 'Model': model, 'Model description': label}
            row.update(regression_metrics(y, p))
            metrics_rows.append(row)
            for threshold in THRESHOLDS:
                erow = {'Phase': phase, 'Model': model, 'Model description': label}
                erow.update(event_metrics(y, p, threshold))
                event_rows.append(erow)
            if phase == '测试集':
                variance = float(np.mean((y - y.mean()) ** 2))
                target_rows.append({'Model': model, 'Model description': label, 'N': len(y), 'Target variance (population)': variance, 'Hypothetical R2 target': 0.8, 'RMSE implied by R2=0.80': float(np.sqrt(0.2 * variance)), 'Note': 'Planning-only. Only RMSE is implied; other metrics require actual predictions.'})
    reg = pd.DataFrame(metrics_rows)
    events = pd.DataFrame(event_rows)
    target = pd.DataFrame(target_rows)
    test_table = reg[reg['Phase'] == '测试集'].copy()
    val_table = reg[reg['Phase'] == '验证集'].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        test_table.to_excel(writer, sheet_name='Test_Province_Regression', index=False)
        val_table.to_excel(writer, sheet_name='Val_Province_Regression', index=False)
        events.to_excel(writer, sheet_name='Province_Event_100_150_200', index=False)
        target.to_excel(writer, sheet_name='R2_080_RMSE_Scenario', index=False)
        pred.to_excel(writer, sheet_name='Source_Province_Predictions', index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                width = min(max(max((len(str(c.value)) if c.value is not None else 0 for c in col)) + 2, 12), 52)
                ws.column_dimensions[letter].width = width
    print(f'Saved: {args.output.resolve()}')
    print('Excluded model rows:', ', '.join(sorted(EXCLUDED_MODELS)))
    print('FAR definition: FP / (FP + TN).')
    print('Metrics are calculated from observed predictions; R2=0.80 is planning-only.')
if __name__ == '__main__':
    main()
