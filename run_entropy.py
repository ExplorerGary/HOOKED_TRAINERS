# run
import os
import torch
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from helper_modules import scan_pt,entropy
except:
    from .helper_modules import scan_pt,entropy

def main(pure_data_enable=True, abs_enabled = False, scale = int(1e6)):
    info = f'''
pure data enabled? -- {pure_data_enable}
abs enabled? -- {abs_enabled}
scale -- {scale}
'''
    # 初始化路径
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, "data_obtained")
    os.makedirs(output_path, exist_ok=True)
    BASE_DIR = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/" if torch.cuda.is_available() else "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files" 
    Logger = ErrorLogger()

    # 系统信息
    total_cpus = multiprocessing.cpu_count()
    print(f"Total CPU cores available: {total_cpus}")
    max_workers = max(4, total_cpus - 2)
    print(f"Using {max_workers} worker processes")
    print("Running in local test mode" if not torch.cuda.is_available() else "Running on HPC")

    # 文件配置
    
    if pure_data_enable:
        suffix = "a1" if not abs_enabled else "a2"
    elif int(scale) == int(1e6):
        suffix = "b1" if not abs_enabled else "b2"
    elif int(scale) == int(1e5):
        suffix = "c1" if not abs_enabled else "c2"
    else:
        suffix = "d1" if not abs_enabled else "d2"
    cache_file = os.path.join(output_path, f"zzz_avail_pt_{suffix}.csv")
    results_file = os.path.join(output_path,f"001_ENTROPY_RESULTS_PROCESSPOOL_{suffix}.csv")
    fieldnames = ["name", "entropy"]
    chunk_size = 250

    processed_files = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                processed_files = {row['name'] for row in reader}
            print(f"Found {len(processed_files)} previously processed files for mode {suffix}")
        except Exception as e:
            print(f"Error reading results file: {e}")
            Logger.record(e)

    try:
        all_files = scan_pt(BASE_DIR)
        avail_pt = [f for f in all_files if f not in processed_files]
        print(f"Total files found: {len(all_files)}, Remaining to process: {len(avail_pt)}")

        if not avail_pt:
            print("All files already processed!")
            return results_file

        pd.DataFrame({"name": avail_pt}).to_csv(cache_file, index=False)
        print(f"Cached file list to: {cache_file}")

    except Exception as e:
        print(f"Error loading files: {e}")
        Logger.record(e)
        return None

    try:
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not processed_files:
                writer.writeheader()

            for i in tqdm(range(0, len(avail_pt), chunk_size), desc=f"Processing batches ({suffix})"):
                batch = avail_pt[i:i + chunk_size]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            cal_entropy,
                            os.path.join(BASE_DIR, path),
                            fp64_enable=True,
                            pure_data_enable=pure_data_enable,
                            scaling=scale,
                            abs_enabled = abs_enabled,
                        ): path
                        for path in batch
                    }

                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                        path = futures[future]
                        try:
                            result = future.result()
                            writer.writerow(result)
                            csvfile.flush()
                        except Exception as e:
                            print(f"Error processing {path}: {e}")
                            Logger.record(f"Failed {path}: {str(e)}")

    except Exception as e:
        print(f"Fatal processing error: {e}")
        Logger.record(f"Fatal error: {str(e)}")
        return None

    print(f"\nProcessing complete! Results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    result_file_a1 = main(pure_data_enable=True, abs_enabled=False)
    if result_file_a1:
        print(f"\n[001a1] PURE_DATA ENABLED:\nCheck results in {result_file_a}")
    else:
        print("[001a1] Failed — check logs.")

    result_file_b1 = main(pure_data_enable=False,abs_enabled=False)
    if result_file_b1:
        print(f"\n[001b1] PURE_DATA DISABLED:\nCheck results in {result_file_b}")
    else:
        print("[001b1] Failed — check logs.")

    result_file_c1 = main(pure_data_enable=False,abs_enabled=False, scale = int(1e5))
    if result_file_c1:
        print(f"\n[001c1] PURE_DATA DISABLED:\nCheck results in {result_file_d}")
    else:
        print("[001c1] Failed — check logs.")
    
    result_file_d1 = main(pure_data_enable=False,abs_enabled=False, scale = int(1e4))
    if result_file_d1:
        print(f"\n[001d1] PURE_DATA DISABLED:\nCheck results in {result_file_c}")
    else:
        print("[001d1] Failed — check logs.")
        
    ### abs enabled : ##############
        
    # result_file_a2 = main(pure_data_enable=True, abs_enabled=True)
    # if result_file_a2:
    #     print(f"\n[001a2] PURE_DATA ENABLED:\nCheck results in {result_file_a}")
    # else:
    #     print("[001a2] Failed — check logs.")

    # result_file_b2 = main(pure_data_enable=False,abs_enabled=True)
    # if result_file_b2:
    #     print(f"\n[001b2] PURE_DATA DISABLED:\nCheck results in {result_file_b}")
    # else:
    #     print("[001b2] Failed — check logs.")
        
    # result_file_c2 = main(pure_data_enable=False,abs_enabled=True, scale = int(1e5))
    # if result_file_c2:
    #     print(f"\n[00c2] PURE_DATA DISABLED:\nCheck results in {result_file_d}")
    # else:
    #     print("[001c2] Failed — check logs.")
        
    # result_file_d2 = main(pure_data_enable=False,abs_enabled=True, scale = int(1e4))
    # if result_file_d2:
    #     print(f"\n[001d2] PURE_DATA DISABLED:\nCheck results in {result_file_c}")
    # else:
    #     print("[001d2] Failed — check logs.")
        



