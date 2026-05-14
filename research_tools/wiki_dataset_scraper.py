from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--out', type=Path, default=Path('scraped_table.csv'))
    args = parser.parse_args()
    tables = pd.read_html(args.url)
    if not tables:
        raise ValueError('No tables found')
    tables[0].to_csv(args.out, index=False)
    print('saved', args.out)

if __name__ == '__main__':
    main()
