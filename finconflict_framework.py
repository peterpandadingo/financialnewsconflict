"""
FinConflict Synthetic Data Generation Framework
================================================
This script defines the pipeline for generating synthetic news entries
(rumor, denial, counterfactual rewrite) and inserting them into real
FNSPID article timelines.

Usage:
    1. Load FNSPID data for a specific ticker and date range
    2. Select anchor events from the predefined list
    3. Generate synthetic entries using the prompting pipeline
    4. Insert entries into the real timeline with metadata tags
    5. Export the combined dataset as JSONL

Requirements:
    pip install openai pandas tqdm

Set your API key:
    export 
"""

import openai
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

CLIENT = openai.OpenAI(api_key="")
MODEL = "gpt-4o"

# Anchor events: (event_name, date, affected_tickers, event_description)
ANCHOR_EVENTS = [
    {
        "event_id": "tariff_2018_march",
        "event_name": "US-China Section 301 Tariff Announcement",
        "anchor_date": "2018-03-22",
        "affected_tickers": ["AAPL", "AMZN", "BA", "CAT", "WMT"],
        "event_description": (
            "The United States announced Section 301 tariffs on approximately "
            "$60 billion worth of Chinese goods, escalating trade tensions "
            "between the US and China."
        ),
        "window_days": 7,  # Extract real articles from ±7 days around anchor date
    },
    {
        "event_id": "tariff_2018_china_retaliation",
        "event_name": "China Retaliatory Tariff Announcement",
        "anchor_date": "2018-04-04",
        "affected_tickers": ["AAPL", "BA", "CAT", "AMZN"],
        "event_description": (
            "China announced retaliatory tariffs on 106 US products worth "
            "approximately $50 billion, responding to US Section 301 tariffs."
        ),
        "window_days": 7,
    },
    {
        "event_id": "uber_ipo_2019",
        "event_name": "Uber IPO",
        "anchor_date": "2019-05-10",
        "affected_tickers": ["UBER", "LYFT"],
        "event_description": (
            "Uber Technologies went public on the New York Stock Exchange "
            "at $45 per share, in one of the most anticipated IPOs of 2019."
        ),
        "window_days": 10,
    },
    {
        "event_id": "airbnb_ipo_2020",
        "event_name": "Airbnb IPO",
        "anchor_date": "2020-12-10",
        "affected_tickers": ["ABNB"],
        "event_description": (
            "Airbnb went public on NASDAQ at $68 per share, surging over 100% "
            "on its first day of trading despite pandemic-related challenges."
        ),
        "window_days": 10,
    },
    {
        "event_id": "iran_soleimani_2020",
        "event_name": "Qasem Soleimani Killing and Iran Escalation",
        "anchor_date": "2020-01-03",
        "affected_tickers": ["LMT", "RTX", "NOC", "GD", "XOM"],
        "event_description": (
            "The United States killed Iranian General Qasem Soleimani in a drone "
            "strike, triggering fears of military escalation and impacting "
            "defense and energy stocks significantly."
        ),
        "window_days": 7,
    },
    {
        "event_id": "arm_ipo_2023",
        "event_name": "Arm Holdings IPO",
        "anchor_date": "2023-09-14",
        "affected_tickers": ["ARM", "NVDA", "QCOM", "INTC"],
        "event_description": (
            "Arm Holdings went public on NASDAQ at $51 per share in the largest "
            "semiconductor IPO in over a decade, backed by SoftBank."
        ),
        "window_days": 10,
    },
]


# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial news writer generating realistic synthetic 
news articles for academic research on AI and misinformation. Your articles must:
- Match the tone, style, and register of real financial newswire reporting
- Be factually plausible but clearly based on the scenario provided
- Be approximately 100-150 words in length
- Not include any real named journalists or real specific quotes that could 
  be misattributed
- Be written in past or present tense consistent with the scenario date"""


def build_rumor_prompt(real_articles: list[dict], event: dict) -> str:
    """
    Type A: Generate a synthetic rumor article.
    The rumor introduces a new, specific market-relevant claim not present
    in the real articles.
    """
    articles_text = "\n\n".join([
        f"DATE: {a['date']}\nHEADLINE: {a['headline']}\nBODY: {a['body']}"
        for a in real_articles
    ])

    return f"""You are writing a synthetic RUMOR article for academic research.

REAL NEWS ARTICLES FROM THIS PERIOD:
{articles_text}

EVENT CONTEXT: {event['event_description']}

TASK: Write a single synthetic rumor article that:
1. Is set on {event['anchor_date']}
2. Introduces a NEW, SPECIFIC claim not mentioned in any of the real articles above
   (e.g., a specific merger valuation, an unnamed source claim, a price target, 
   a regulatory outcome, an executive action)
3. Is written as breaking news from an unverified source
4. Uses phrases like "according to people familiar with the matter" or 
   "sources say" or "is reportedly considering"
5. Is plausible given the event context but NOT verifiable from the real articles

Output format — respond with ONLY this JSON, nothing else:
{{
  "headline": "...",
  "body": "...",
  "claim": "one-sentence summary of the specific new claim introduced"
}}"""


def build_denial_prompt(rumor_article: dict, event: dict, denial_date: str) -> str:
    """
    Type B: Generate a denial article that directly contradicts the rumor.
    """
    return f"""You are writing a synthetic DENIAL article for academic research.

RUMOR ARTICLE TO DENY:
HEADLINE: {rumor_article['headline']}
BODY: {rumor_article['body']}
CLAIM BEING DENIED: {rumor_article['claim']}

TASK: Write a denial article that:
1. Is set on {denial_date} (after the rumor)
2. Directly denies the specific claim in the rumor above
3. Is written as an official company statement or named executive denial
4. Uses phrases like "denied", "has no plans to", "is not in discussions",
   "categorically rejects", "a spokesperson said"
5. Preserves all named entities and numerical values from the rumor 
   but reverses the claim's direction
6. Is approximately 100 words

Output format — respond with ONLY this JSON, nothing else:
{{
  "headline": "...",
  "body": "..."
}}"""


def build_counterfactual_prompt(real_article: dict) -> str:
    """
    Type C: Generate a counterfactual rewrite of a real article.
    Reverses the directional claim while preserving all other content.
    """
    return f"""You are creating a COUNTERFACTUAL REWRITE of a real article for 
academic research on AI misinformation detection.

ORIGINAL REAL ARTICLE:
HEADLINE: {real_article['headline']}
BODY: {real_article['body']}

TASK: Rewrite BOTH the headline and body so that:
1. The key directional claim is REVERSED:
   - "rose" becomes "fell" (and vice versa)
   - "increased" becomes "decreased" (and vice versa)
   - "at the low end" becomes "at the high end" (and vice versa)
   - "priced below expectations" becomes "priced above expectations"
   - If the original outcome was POSITIVE, make it NEGATIVE
   - If the original outcome was NEGATIVE, make it POSITIVE
2. The headline MUST also be rewritten to reflect the reversed claim
3. All named entities (companies, people, places) are PRESERVED exactly
4. All numerical values are PRESERVED exactly
5. The article length and structure are PRESERVED
6. Every change must be logically consistent — do not introduce contradictions

Output format — respond with ONLY this JSON, nothing else:
{{
  "headline": "rewritten headline reflecting the reversed claim",
  "body": "rewritten body with reversed directional claim",
  "original_claim": "one-sentence summary of the original directional claim",
  "counterfactual_claim": "one-sentence summary of the reversed claim"
}}"""


# ── Generation Functions ─────────────────────────────────────────────────────

def generate_synthetic_entry(prompt: str, entry_type: str) -> dict | None:
    """Call OpenAI API and return parsed JSON response."""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"Error generating {entry_type} entry: {e}")
        return None


def generate_rumor(real_articles: list[dict], event: dict) -> dict | None:
    prompt = build_rumor_prompt(real_articles, event)
    result = generate_synthetic_entry(prompt, "rumor")
    if result:
        result["entry_type"] = "TYPE_A_RUMOR"
        result["event_id"] = event["event_id"]
        result["is_synthetic"] = True
        result["insertion_date"] = event["anchor_date"]
        result["source_label"] = "PLACEHOLDER_ASSIGN_IN_NEXT_STEP"
    return result


def generate_denial(rumor: dict, event: dict, days_after: int = 2) -> dict | None:
    anchor = datetime.strptime(event["anchor_date"], "%Y-%m-%d")
    denial_date = (anchor + timedelta(days=days_after)).strftime("%Y-%m-%d")
    prompt = build_denial_prompt(rumor, event, denial_date)
    result = generate_synthetic_entry(prompt, "denial")
    if result:
        result["entry_type"] = "TYPE_B_DENIAL"
        result["event_id"] = event["event_id"]
        result["is_synthetic"] = True
        result["insertion_date"] = denial_date
        result["denies_claim"] = rumor.get("claim", "")
        result["source_label"] = "PLACEHOLDER_ASSIGN_IN_NEXT_STEP"
    return result


def generate_counterfactual(real_article: dict, event: dict) -> dict | None:
    prompt = build_counterfactual_prompt(real_article)
    result = generate_synthetic_entry(prompt, "counterfactual")
    if result:
        result["entry_type"] = "TYPE_C_COUNTERFACTUAL"
        result["event_id"] = event["event_id"]
        result["is_synthetic"] = True
        result["insertion_date"] = real_article["date"]
        result["source_label"] = real_article.get("source", "UNKNOWN")
        result["original_real_article_date"] = real_article["date"]
    return result


# ── Source Label Assignment ───────────────────────────────────────────────────

SOURCE_LABELS = {
    "high_credibility": ["Reuters", "Bloomberg", "Wall Street Journal",
                         "Financial Times", "CNBC"],
    "low_credibility":  ["MarketInsider Blog", "TradingNow",
                         "FinanceUpdate24", "StockAlert Daily"],
    "unknown":          ["Anonymous Source", "Unverified Report"],
}


def assign_source_labels(entries: list[dict]) -> list[dict]:
    """
    Systematically assign source label conditions across entries
    so each event has entries under each label condition.
    Cycles through high/low/unknown across entries.
    """
    label_cycle = (
        SOURCE_LABELS["high_credibility"] +
        SOURCE_LABELS["low_credibility"] +
        SOURCE_LABELS["unknown"]
    )
    for i, entry in enumerate(entries):
        if entry["source_label"] == "PLACEHOLDER_ASSIGN_IN_NEXT_STEP":
            entry["source_label"] = label_cycle[i % len(label_cycle)]
        entry["source_credibility_condition"] = (
            "high" if entry["source_label"] in SOURCE_LABELS["high_credibility"]
            else "low" if entry["source_label"] in SOURCE_LABELS["low_credibility"]
            else "unknown"
        )
    return entries


# ── Timeline Construction ─────────────────────────────────────────────────────

def build_timeline(real_articles: list[dict],
                   synthetic_entries: list[dict],
                   position: str = "late") -> list[dict]:
    """
    Insert synthetic entries into real article timeline.
    position: 'early' | 'middle' | 'late'
    Returns chronologically sorted combined timeline.
    """
    timeline = []

    # Tag real articles
    for a in real_articles:
        a["is_synthetic"] = False
        timeline.append(a)

    # Assign insertion position
    n = len(real_articles)
    for entry in synthetic_entries:
        if position == "early":
            # Insert before first real article
            entry["timeline_position"] = "early"
            # Adjust date to be before first real article if needed
        elif position == "middle":
            entry["timeline_position"] = "middle"
        else:
            entry["timeline_position"] = "late"
        timeline.append(entry)

    # Sort by date
    timeline.sort(key=lambda x: x.get("insertion_date") or x.get("date", ""))
    return timeline


# ── Export ────────────────────────────────────────────────────────────────────

def export_dataset(timelines: list[dict], output_path: str):
    """Export combined real+synthetic timelines as JSONL."""
    with open(output_path, "w") as f:
        for timeline in timelines:
            f.write(json.dumps(timeline) + "\n")
    print(f"Exported {len(timelines)} timelines to {output_path}")


# ── Validation Rubric ─────────────────────────────────────────────────────────

VALIDATION_RUBRIC = """
MANUAL VALIDATION CHECKLIST (for 10% random sample)

For each synthetic entry, score each criterion Y/N:

TYPE A (Rumor):
[ ] Factual plausibility — reads as credible financial news reporting
[ ] Claim distinctness — introduces a claim NOT present in any real article
[ ] Appropriate uncertainty language ("sources say", "reportedly", etc.)
[ ] Length 100-150 words
[ ] No real journalist names or specific attributable quotes

TYPE B (Denial):
[ ] Directly denies the specific claim in the paired rumor entry
[ ] Named entity preservation — same companies/people as the rumor
[ ] Appropriate denial language ("denied", "no plans to", etc.)
[ ] Source is plausible (company spokesperson, executive, regulator)
[ ] Length 100-150 words

TYPE C (Counterfactual):
[ ] Directional claim correctly reversed (rose→fell, approved→rejected, etc.)
[ ] All named entities preserved exactly
[ ] All numerical values preserved exactly
[ ] Length matches original article (±20%)
[ ] Writing style preserved

FAIL CONDITIONS (discard and regenerate if any):
- Entry repeats content from a real article word-for-word
- Claim is too vague to be testable (e.g., "things changed")  
- Named real journalists appear with fabricated quotes
- Numerical values were changed (Type C only)
"""

# ── Demo Run ─────────────────────────────────────────────────────────────────

def demo_run():
    import csv
    event = ANCHOR_EVENTS[0]

    # Load real articles spread across different dates
    real_articles = []
    seen_dates = set()
    
    with open('./aa_tariff_articles.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row['Date'][:10]
            body = row.get('Article', '').strip()
            if not body:
                body = row.get('Article_title', '').strip()
            # One article per date to get spread
            if body and date not in seen_dates:
                real_articles.append({
                    "date": date,
                    "headline": row['Article_title'],
                    "body": body[:500],
                    "source": row.get('Publisher') or "Unknown",
                    "is_synthetic": False,
                })
                seen_dates.add(date)
            if len(real_articles) >= 6:
                break

    real_articles.sort(key=lambda x: x['date'])
    print(f"Loaded {len(real_articles)} real articles across {len(seen_dates)} dates:")
    for a in real_articles:
        print(f"  {a['date']}: {a['headline'][:60]}...")

    # Generate three synthetic entries at different positions
    print("\nGenerating Type A (Rumor) — inserted EARLY...")
    rumor_early = generate_rumor(real_articles[:2], event)
    if rumor_early:
        rumor_early['insertion_date'] = '2018-03-04'
        rumor_early['timeline_position'] = 'early'
        print(f"Rumor: {rumor_early['headline']}")

    print("\nGenerating Type B (Denial) — inserted MIDDLE...")
    denial_mid = generate_denial(rumor_early, event, days_after=0)
    if denial_mid:
        denial_mid['insertion_date'] = '2018-03-10'
        denial_mid['timeline_position'] = 'middle'
        print(f"Denial: {denial_mid['headline']}")

    print("\nGenerating Type C (Counterfactual) — inserted LATE...")
    cf_late = generate_counterfactual(real_articles[-1], event)
    if cf_late:
        cf_late['insertion_date'] = '2018-04-01'
        cf_late['timeline_position'] = 'late'
        print(f"Counterfactual: {cf_late['headline']}")

    synthetic = [e for e in [rumor_early, denial_mid, cf_late] if e]
    synthetic = assign_source_labels(synthetic)

    # Combine and sort chronologically
    all_entries = real_articles + synthetic
    all_entries.sort(key=lambda x: x.get('insertion_date') or x.get('date', ''))

    with open('./finconflict_sample_dataset.jsonl', 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')

    print("\n--- FINAL TIMELINE (chronological) ---")
    for e in all_entries:
        date = e.get('insertion_date') or e.get('date')
        tag = '[SYNTHETIC]' if e['is_synthetic'] else '[REAL]'
        etype = e.get('entry_type', '')
        print(f"  {date} {tag} {etype}: {e['headline'][:60]}...")

    print(f"\nTotal: {len(all_entries)} entries")
    print(f"Real: {sum(1 for e in all_entries if not e['is_synthetic'])}")
    print(f"Synthetic: {sum(1 for e in all_entries if e['is_synthetic'])}")
    print("Saved to finconflict_sample_dataset.jsonl")
    return all_entries

if __name__ == "__main__":
    print("FinConflict Synthetic Data Generation Framework")
    print("=" * 50)
    print("\nRunning demo with real FNSPID data...\n")
    demo_run()