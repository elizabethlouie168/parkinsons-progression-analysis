from __future__ import annotations
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'https://clinicaltrials.gov/api/query/study_fields'


def fetch(condition: str, max_results: int = 20):
    params = urllib.parse.urlencode({
        'expr': condition,
        'fields': 'NCTId,BriefTitle,OverallStatus,ConditionName,LocationCity,LocationCountry,InterventionName',
        'min_rnk': 1,
        'max_rnk': max_results,
        'fmt': 'json'
    })
    with urllib.request.urlopen(f'{BASE}?{params}') as r:
        return json.loads(r.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('condition')
    parser.add_argument('--out', type=Path, default=Path('clinical_trials.json'))
    args = parser.parse_args()
    data = fetch(args.condition)
    args.out.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('saved', args.out)

if __name__ == '__main__':
    main()
