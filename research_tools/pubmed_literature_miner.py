"""
PubMed Literature Miner

A lightweight bibliometric tool for biomedical literature exploration.
It queries PubMed, retrieves article metadata, and creates a simple trend report.

Example:
    python research_tools/pubmed_literature_miner.py "Parkinson's disease biomarkers" --max-results 100 --out pubmed_report.md

Notes:
    This tool uses NCBI E-utilities and does not require an API key for small requests.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def search_pubmed(query: str, max_results: int) -> list[str]:
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    data = json.loads(fetch_url(f"{BASE}/esearch.fcgi?{params}"))
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_details(pmids: list[str]) -> list[dict[str, str]]:
    if not pmids:
        return []
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })
    root = ET.fromstring(fetch_url(f"{BASE}/efetch.fcgi?{params}"))
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="").strip()
        year = article.findtext(".//PubDate/Year", default="Unknown")
        journal = article.findtext(".//Journal/Title", default="Unknown")
        abstract_parts = [node.text or "" for node in article.findall(".//AbstractText")]
        abstract = " ".join(part.strip() for part in abstract_parts if part.strip())
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", default="")
            fore = author.findtext("ForeName", default="")
            name = " ".join(x for x in [fore, last] if x).strip()
            if name:
                authors.append(name)
        records.append({
            "pmid": pmid,
            "title": title,
            "year": year,
            "journal": journal,
            "authors": "; ".join(authors),
            "abstract": abstract,
        })
    return records


def keyword_counts(records: list[dict[str, str]], top_n: int = 20) -> list[tuple[str, int]]:
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "into", "using", "study", "disease", "patients"}
    counter: collections.Counter[str] = collections.Counter()
    for record in records:
        text = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
        words = re.findall(r"[a-z][a-z\-]{3,}", text)
        counter.update(word for word in words if word not in stopwords)
    return counter.most_common(top_n)


def write_csv(records: list[dict[str, str]], path: Path) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def build_report(query: str, records: list[dict[str, str]]) -> str:
    year_counts = collections.Counter(record.get("year", "Unknown") for record in records)
    top_journals = collections.Counter(record.get("journal", "Unknown") for record in records).most_common(10)
    top_keywords = keyword_counts(records)
    lines = [
        f"# PubMed Literature Mining Report: {query}",
        "",
        f"Articles analyzed: {len(records)}",
        "",
        "## Publications by Year",
    ]
    for year, count in sorted(year_counts.items()):
        lines.append(f"- {year}: {count}")
    lines.extend(["", "## Top Journals"])
    for journal, count in top_journals:
        lines.append(f"- {journal}: {count}")
    lines.extend(["", "## Top Keywords"])
    for word, count in top_keywords:
        lines.append(f"- {word}: {count}")
    lines.extend(["", "## Sample Articles"])
    for record in records[:10]:
        lines.append(f"- {record.get('year')} — {record.get('title')} (PMID: {record.get('pmid')})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine PubMed literature metadata for quick bibliometric analysis.")
    parser.add_argument("query", help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("pubmed_report.md"))
    parser.add_argument("--csv", type=Path, default=Path("pubmed_records.csv"))
    args = parser.parse_args()

    pmids = search_pubmed(args.query, args.max_results)
    time.sleep(0.35)
    records = fetch_details(pmids)
    args.out.write_text(build_report(args.query, records), encoding="utf-8")
    write_csv(records, args.csv)
    print(f"Wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
