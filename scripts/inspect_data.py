"""
Inspect the FIRST row of a verl RL training parquet: field names, types, and
a short sample of each value. Usage:

    python scripts/inspect_data.py <path/to/train.parquet> [row_index]

The default row is 0. Also prints the pyarrow schema (authoritative types),
since pandas may flatten nested struct/list fields.
"""
import json
import sys

import pandas as pd
import pyarrow.parquet as pq


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"=== Schema of {path} ===")
    print(pq.read_schema(path))

    df = pd.read_parquet(path)
    x = df.iloc[idx]
    print(f"\n=== Row {idx} (of {len(df)}) ===")
    for k, v in x.items():
        print(f"\n[{k}] type={type(v).__name__}")
        if isinstance(v, (dict, list)):
            print(json.dumps(v, ensure_ascii=False, indent=2)[:1000])
        else:
            print(str(v)[:1000])


if __name__ == "__main__":
    main()
