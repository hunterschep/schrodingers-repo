#!/usr/bin/env python3
"""
Create a big combined corpus from all language CSV files.

Output format: lang_code | formosan_sentence | english_sentence | chinese_sentence
Where english_sentence is populated from _en.csv files and chinese_sentence from _zh.csv files.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv_files(directory: Path) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Read all CSV files and organize by language code.
    
    Returns:
        Dict mapping lang_code -> list of (formosan, english, chinese, source) tuples
    """
    data = {}
    
    for csv_file in directory.glob("*.csv"):
        if csv_file.name in ["big_corpus_combined.csv", "summary_stats.csv"]:
            continue  # Skip output files
            
        # Extract language code and type from filename (e.g., "ami_en.csv" -> "ami", "en")
        name_parts = csv_file.stem.split('_')
        if name_parts[1] not in ['en', 'zh']:
            continue
            
        lang_code = name_parts[0]
        target_lang = name_parts[1]
        
        print(f"📖 Reading {csv_file.name}...")
        
        if lang_code not in data:
            data[lang_code] = []
            
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) < 4:
                        continue
                        
                    formosan_text = row[0]
                    target_text = row[1]  # english or chinese
                    source = row[2]
                    
                    # Create tuple: (formosan, english, chinese, source)
                    if target_lang == 'en':
                        entry = (formosan_text, target_text, "", source)
                    else:  # target_lang == 'zh'
                        entry = (formosan_text, "", target_text, source)
                        
                    data[lang_code].append(entry)
                    
        except Exception as e:
            print(f"⚠️  Error reading {csv_file.name}: {e}")
            continue
    
    return data


def combine_language_data(lang_data: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """
    Combine English and Chinese entries for the same language.
    Try to match formosan sentences and merge translations.
    """
    # Group by formosan sentence
    formosan_to_entry = {}
    
    for formosan, english, chinese, source in lang_data:
        key = formosan.strip()
        
        if key in formosan_to_entry:
            # Merge translations
            existing = formosan_to_entry[key]
            merged_english = existing[1] or english
            merged_chinese = existing[2] or chinese
            # Keep the first source, or prefer non-empty source
            merged_source = existing[3] if existing[3] else source
            formosan_to_entry[key] = (formosan, merged_english, merged_chinese, merged_source)
        else:
            formosan_to_entry[key] = (formosan, english, chinese, source)
    
    return list(formosan_to_entry.values())


def write_combined_corpus(data: Dict[str, List[Tuple[str, str, str, str]]], output_file: Path):
    """Write the combined corpus to a CSV file."""
    total_sentences = 0
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lang_code', 'formosan_sentence', 'english_sentence', 'chinese_sentence', 'source'])
        
        for lang_code in sorted(data.keys()):
            print(f"🔄 Processing {lang_code}...")
            
            # Combine English and Chinese data for this language
            combined_entries = combine_language_data(data[lang_code])
            
            # Write all entries for this language
            for formosan, english, chinese, source in combined_entries:
                writer.writerow([lang_code, formosan, english, chinese, source])
                total_sentences += 1
                
            print(f"✅ {lang_code}: {len(combined_entries):,} sentences")
    
    print(f"\n🎉 Combined corpus written to {output_file}")
    print(f"📊 Total sentences: {total_sentences:,}")


def main():
    script_dir = Path(__file__).parent
    output_file = script_dir / "big_corpus_combined.csv"
    
    print("🚀 Creating combined corpus from all language CSV files...\n")
    
    # Read all CSV files
    data = read_csv_files(script_dir)
    
    if not data:
        print("❌ No valid CSV files found!")
        return
    
    print(f"\n📋 Found {len(data)} languages: {', '.join(sorted(data.keys()))}")
    
    # Write combined corpus
    write_combined_corpus(data, output_file)


if __name__ == "__main__":
    main()