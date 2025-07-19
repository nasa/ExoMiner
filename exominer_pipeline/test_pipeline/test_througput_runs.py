"""
Test pipeline throughput.
"""

import pandas as pd
from pathlib import Path

#%%

output_dir = Path('/Users/msaragoc/Downloads/exominer_pipeline_run_20250630-174917')

predictions_tbl_fp = output_dir / f'predictions_{output_dir.name}.csv'
predictions_tbls_fps = list(Path(output_dir).rglob('ranked_predictions_predictset.csv'))
n_jobs = 80
pred_tbls_dict = {'job_id': range(n_jobs), 'exists': [False] * n_jobs}
pred_tbls_df = pd.DataFrame(pred_tbls_dict)
for fp in predictions_tbls_fps:
    job_id = int(fp.parent.parent.name.split('_')[-1])
    pred_tbls_df.loc[job_id, 'exists'] = True
print(f'Found {len(predictions_tbls_fps)} job predictions files in {output_dir}.')
predictions_tbl = pd.concat([pd.read_csv(fp) for fp in predictions_tbls_fps], axis=0, ignore_index=True)
predictions_tbl.to_csv(predictions_tbl_fp, index=False)

print(pred_tbls_df.loc[~pred_tbls_df['exists'], 'job_id'])