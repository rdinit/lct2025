import pandas as pd
from pathlib import Path

def merge_physiological_data(base_path):
    all_data = []
    base_path = Path(base_path)
    
    for condition in ['hypoxia', 'regular']:
        condition_path = base_path / condition
        
        if not condition_path.exists():
            continue
            
        for group_dir in condition_path.iterdir():
            if not group_dir.is_dir():
                continue
                
            group_id = group_dir.name
            bpm_path = group_dir / 'bpm'
            uterus_path = group_dir / 'uterus'
            
            if not (bpm_path.exists() and uterus_path.exists()):
                continue
            
            bpm_files = {f.stem: f for f in bpm_path.glob('*.csv')}
            uterus_files = {f.stem: f for f in uterus_path.glob('*.csv')}
            
            for base_name in bpm_files.keys(): # [_1, _3]
                uterus_base = base_name.replace('_1', '_2').replace('_3', '_4') # [_1, _3] -> [_2, _4]
                
                if uterus_base in uterus_files:
                    bpm_df = pd.read_csv(bpm_files[base_name])
                    bpm_df.columns = ['time_sec', 'bpm']
                    uterus_df = pd.read_csv(uterus_files[uterus_base])
                    uterus_df.columns = ['time_sec', 'uterus']
                    merged = pd.merge(bpm_df, uterus_df, on='time_sec', how='inner')
                    merged['group_id'] = group_id
                    merged['sequence_id'] = base_name
                    merged['target'] = condition
                    merged['timestamp'] = merged['time_sec']
                    merged = merged[['timestamp', 'group_id', 'sequence_id', 'bpm', 'uterus', 'target']]
                    all_data.append(merged)
                    # print(f"Processed: {condition}/{group_id}/{base_name}")
                        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()