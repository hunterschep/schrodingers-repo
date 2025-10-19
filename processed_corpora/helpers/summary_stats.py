#!/usr/bin/env python3
"""
Processed Formosan Corpora Summary Statistics

Analyze the filtered and split corpus files, showing:
- Number of sentences per corpus
- Train/validation/test split statistics
- Language and direction totals
- Formatted summary tables
"""

import csv
import os
from pathlib import Path
from collections import defaultdict


# Language code to name mapping
LANGUAGE_NAMES = {
    'ami': 'Amis',
    'bnn': 'Bunun', 
    'ckv': 'Kavalan',
    'dru': 'Rukai',
    'pwn': 'Paiwan',
    'pyu': 'Puyuma',
    'ssf': 'Thao',
    'sxr': 'Saaroa',
    'szy': 'Sakizaya',
    'tao': 'Yami/Tao',
    'tay': 'Atayal',
    'trv': 'Seediq/Truku',
    'tsu': 'Tsou',
    'xnb': 'Kanakanavu',
    'xsy': 'Saisiyat'
}

# Target language mapping
TARGET_LANGUAGES = {
    'en': 'English',
    'zh': 'Chinese'
}


def count_csv_rows_and_splits(file_path: Path) -> tuple[int, dict]:
    """
    Count total rows and split breakdown in a CSV file.
    
    Returns:
        (total_rows, split_counts) where split_counts is {'train': N, 'validate': N, 'test': N}
    """
    if not file_path.exists():
        return 0, {}
    
    split_counts = defaultdict(int)
    total_rows = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                split_type = row.get('split', 'unknown')
                split_counts[split_type] += 1
                
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0, {}
    
    return total_rows, dict(split_counts)


def count_csv_rows_by_dialect(file_path: Path) -> tuple[int, dict, dict]:
    """
    Count total rows, split breakdown, and dialect breakdown in a CSV file.
    
    Returns:
        (total_rows, split_counts, dialect_counts) where:
        - split_counts is {'train': N, 'validate': N, 'test': N}
        - dialect_counts is {'dialect_name': N, ...}
    """
    if not file_path.exists():
        return 0, {}, {}
    
    split_counts = defaultdict(int)
    dialect_counts = defaultdict(int)
    total_rows = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                split_type = row.get('split', 'unknown')
                split_counts[split_type] += 1
                
                dialect = row.get('dialect', 'unknown')
                if dialect and dialect.strip():
                    dialect_counts[dialect.strip()] += 1
                else:
                    dialect_counts['unknown'] += 1
                
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0, {}, {}
    
    return total_rows, dict(split_counts), dict(dialect_counts)


def parse_filename(filename: str) -> tuple[str, str, str]:
    """
    Parse filename like 'ami_zh.csv' into (lang_code, target_lang, full_name).
    
    Returns:
        ('ami', 'zh', 'Amis ↔ Chinese')
    """
    # Remove .csv extension first
    base = filename.replace('.csv', '').replace('.CSV', '')
    
    # Split into language code and target
    parts = base.split('_')
    if len(parts) >= 2:
        lang_code = parts[0]
        target_code = parts[1]
        
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        target_name = TARGET_LANGUAGES.get(target_code, target_code.upper())
        
        full_name = f"{lang_name} ↔ {target_name}"
        return lang_code, target_code, full_name
    
    return filename, 'unknown', filename


def get_processed_files() -> list[Path]:
    """Get all processed CSV files."""
    current_dir = Path(__file__).parent
    return sorted(current_dir.glob("*.csv"))


def print_individual_corpus_stats(files: list[Path]):
    """Print detailed statistics for each corpus file."""
    print("=" * 80)
    print("INDIVIDUAL CORPUS STATISTICS")
    print("=" * 80)
    print(f"{'Corpus':<20} | {'Total':<8} | {'Train':<8} | {'Valid':<8} | {'Test':<8}")
    print("-" * 80)
    
    for file_path in files:
        lang_code, target_code, full_name = parse_filename(file_path.name)
        total_rows, split_counts = count_csv_rows_and_splits(file_path)
        
        train_count = split_counts.get('train', 0)
        val_count = split_counts.get('validate', 0) + split_counts.get('val', 0)
        test_count = split_counts.get('test', 0)
        
        print(f"{full_name:<20} | {total_rows:<8,} | {train_count:<8,} | {val_count:<8,} | {test_count:<8,}")


def print_language_summary(files: list[Path]):
    """Print summary by language (combining English and Chinese)."""
    lang_totals = defaultdict(int)
    lang_details = defaultdict(lambda: {'en': 0, 'zh': 0})
    
    for file_path in files:
        lang_code, target_code, full_name = parse_filename(file_path.name)
        total_rows, _ = count_csv_rows_and_splits(file_path)
        
        lang_totals[lang_code] += total_rows
        lang_details[lang_code][target_code] += total_rows
    
    print("\n" + "=" * 80)
    print("LANGUAGE SUMMARY")
    print("=" * 80)
    print(f"{'Language':<15} | {'English':<10} | {'Chinese':<10} | {'Total':<10}")
    print("-" * 80)
    
    total_en = total_zh = total_all = 0
    
    for lang_code in sorted(lang_totals.keys()):
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        en_count = lang_details[lang_code]['en']
        zh_count = lang_details[lang_code]['zh']
        total_count = lang_totals[lang_code]
        
        print(f"{lang_name:<15} | {en_count:<10,} | {zh_count:<10,} | {total_count:<10,}")
        
        total_en += en_count
        total_zh += zh_count
        total_all += total_count
    
    print("-" * 80)
    print(f"{'TOTAL':<15} | {total_en:<10,} | {total_zh:<10,} | {total_all:<10,}")
    print("=" * 80)


def print_direction_summary(files: list[Path]):
    """Print summary by translation direction."""
    direction_totals = defaultdict(int)
    direction_splits = defaultdict(lambda: defaultdict(int))
    
    for file_path in files:
        lang_code, target_code, full_name = parse_filename(file_path.name)
        total_rows, split_counts = count_csv_rows_and_splits(file_path)
        
        direction = TARGET_LANGUAGES.get(target_code, target_code.upper())
        direction_totals[direction] += total_rows
        
        for split_type, count in split_counts.items():
            direction_splits[direction][split_type] += count
    
    print("\n" + "=" * 70)
    print("TRANSLATION DIRECTION SUMMARY")
    print("=" * 70)
    print(f"{'Direction':<10} | {'Total':<8} | {'Train':<8} | {'Valid':<8} | {'Test':<8}")
    print("-" * 70)
    
    for direction in sorted(direction_totals.keys()):
        total = direction_totals[direction]
        train = direction_splits[direction]['train']
        val = direction_splits[direction]['validate'] + direction_splits[direction]['val']
        test = direction_splits[direction]['test']
        
        print(f"{direction:<10} | {total:<8,} | {train:<8,} | {val:<8,} | {test:<8,}")


def print_dialect_summary(files: list[Path]):
    """Print summary by dialect for each language."""
    # Collect dialect data by language
    lang_dialect_counts = defaultdict(lambda: defaultdict(int))
    
    for file_path in files:
        lang_code, target_code, full_name = parse_filename(file_path.name)
        total_rows, split_counts, dialect_counts = count_csv_rows_by_dialect(file_path)
        
        # Aggregate dialect counts for this language
        for dialect, count in dialect_counts.items():
            lang_dialect_counts[lang_code][dialect] += count
    
    print("\n" + "=" * 80)
    print("DIALECT SUMMARY BY LANGUAGE")
    print("=" * 80)
    
    for lang_code in sorted(lang_dialect_counts.keys()):
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        dialect_data = lang_dialect_counts[lang_code]
        
        # Calculate total for this language
        lang_total = sum(dialect_data.values())
        
        print(f"\n{lang_name} ({lang_code.upper()}) - Total: {lang_total:,} sentences")
        print("-" * 50)
        
        # Sort dialects by count (descending)
        sorted_dialects = sorted(dialect_data.items(), key=lambda x: x[1], reverse=True)
        
        for dialect, count in sorted_dialects:
            percentage = (count / lang_total * 100) if lang_total > 0 else 0
            print(f"  {dialect:<30} | {count:>8,} ({percentage:>5.1f}%)")


def print_overall_summary(files: list[Path]):
    """Print overall corpus statistics."""
    total_sentences = 0
    total_splits = defaultdict(int)
    total_files = len(files)
    
    for file_path in files:
        total_rows, split_counts = count_csv_rows_and_splits(file_path)
        total_sentences += total_rows
        
        for split_type, count in split_counts.items():
            total_splits[split_type] += count
    
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"Total corpus files:     {total_files:>8}")
    print(f"Total languages:        {len(LANGUAGE_NAMES):>8}")
    print(f"Total sentence pairs:   {total_sentences:>8,}")
    print()
    print("Split breakdown:")
    print(f"  Training:             {total_splits['train']:>8,}")
    print(f"  Validation:           {total_splits['validate'] + total_splits['val']:>8,}")
    print(f"  Test:                 {total_splits['test']:>8,}")
    print("=" * 50)


def main():
    """Main function to generate all statistics."""
    print("🌏 PROCESSED FORMOSAN CORPORA STATISTICS")
    print(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all processed files
    files = get_processed_files()
    
    if not files:
        print("❌ No processed CSV files found in current directory!")
        print("Looking for files matching pattern: *.csv")
        return
    
    print(f"\n📊 Found {len(files)} processed corpus files")
    
    # Generate all statistics
    print_individual_corpus_stats(files)
    print_language_summary(files)
    print_direction_summary(files)
    print_dialect_summary(files)
    print_overall_summary(files)
    
    print("\n✅ Statistics generation complete!")


if __name__ == "__main__":
    main()