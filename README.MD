# FormosanBank • Machine Translation Corpus 🌏

Ready-to-use parallel corpora and training pipelines for **15 Indigenous Taiwanese languages**.

## 📊 What's Inside

**30 processed CSV files** containing sentence-aligned parallel data:
- **15 Formosan languages** ↔ Chinese (Traditional)  
- **15 Formosan languages** ↔ English (where available)
- Pre-cleaned, deduplicated, and split into train/validate/test
- **~500K total sentence pairs** across all languages

### Languages Included

| Code | Language | 
|------|----------|
| `ami` | Amis |
| `tay` | Atayal |
| `bnn` | Bunun | 
| `ckv` | Kavalan | 
| `pwn` | Paiwan |
| `pyu` | Puyuma | 
| `dru` | Rukai | 
| `sxr` | Saaroa | 
| `xsy` | Saisiyat | 
| `szy` | Sakizaya | 
| `trv` | Seediq/Truku | 
| `ssf` | Thao | 
| `tsu` | Tsou | 
| `xnb` | Kanakanavu | 
| `tao` | Yami/Tao |  

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/Make-Corpus.git
cd Make-Corpus

# Install dependencies
pip install -r requirements.txt

# Set up GitHub token for data fetching (optional, only needed for rebuilding)
echo "GITHUB_TOKEN=ghp_your_token_here" > .env
```

### Using Pre-processed Corpora

Each CSV file contains columns: `{formosan_lang}`, `{target_lang}`, `source`, `kindOf`, `dialect`, `row_type`, `split`

```python
import pandas as pd

# Load Amis-English corpus
df = pd.read_csv('processed_corpora/ami_en_processed.csv')

# Filter by split
train_data = df[df['split'] == 'train']
test_data = df[df['split'] == 'test']

print(f"Training pairs: {len(train_data):,}")
print(f"Test pairs: {len(test_data):,}")
```

### Generate Corpus Statistics

```bash
cd processed_corpora/helpers
python summary_stats.py
```

This shows detailed statistics including:
- Sentence counts per corpus
- Train/validation/test splits
- Language and direction totals
- Dialect distributions

---

## 🤖 Training Machine Translation Models

### Option 1: NLLB-200 (Recommended)

**Best for:** State-of-the-art multilingual models, transfer learning

```bash
# 1. Set up custom NLLB model with Formosan language tokens
cd scripts/mt/nllb/prelims
python setup_nllb200_formosan.py

# 2. Train bidirectional model
cd ../training
python train_formosan_nllb200.py \
  --src-lang amis --tgt-lang english \
  --tokenizer ../prelims/formosan_multilingual_nllb_tokenizer \
  --model ../prelims/formosan_multilingual_nllb_model \
  --input ../../../processed_corpora/ami_en_processed.csv \
  --normalize \
  --steps 20000 --batch-size 8 \
  --save-interval 5000 --eval-interval 5000

# 3. Evaluate
cd ../eval
python eval_formosan_nllb200.py \
  --tokenizer ../prelims/formosan_multilingual_nllb_tokenizer \
  --model runs/amis_english/[timestamp]/final \
  --input ../../../processed_corpora/ami_en_processed.csv \
  --batch-size 16 --beam 5 \
  --save-json results.json --csv-out predictions.csv
```

**Features:**
- Bidirectional training (Formosan↔English/Chinese)
- Transfer learning from NLLB-200's 200+ languages
- FP16/AMP support for efficiency
- Adafactor optimizer with warmup

### Option 2: OpenNMT-py (Classic RNN Seq2Seq)

**Best for:** Lightweight models, CPU inference, baseline comparisons

```bash
# 1. Prepare data for OpenNMT format
cd scripts/mt/opennmt/setup
python prep_opennmt.py \
  --csv ../../../processed_corpora/ami_en_processed.csv \
  --outdir ../../../processed_corpora/opennmt/data

# 2. Train SentencePiece tokenizer
python prep_spm.py \
  --data-dir ../../../processed_corpora/opennmt/data/ami_en

# 3. Train model (requires GPU, typically run on HPC)
cd ../../../../bash
./train_opennmt_pair.sh ami_en
```

**Features:**
- BiLSTM encoder + LSTM decoder with attention
- Early stopping on validation perplexity
- Target-language tags for bidirectional training (`<2en>`, `<2ami>`)
- Automatic BLEU/chrF++ scoring on test set

---

## 🏗️ Rebuilding Pipeline (Advanced)

To rebuild corpora from scratch:

```bash
# Run full pipeline for all 15 languages
./build_corpora.sh

# Or manually for a specific language:
python scripts/local/fetch_xml.py --src-lang ami --public
python scripts/local/clean_xml.py --src-lang ami
python scripts/local/make_corpus.py \
  --xml-dir downloaded_ami --target chinese \
  --out raw_corpora/ami_zh.csv
python scripts/local/filter_split_corpus.py \
  --input raw_corpora/ami_zh.csv \
  --output processed_corpora/ami_zh_processed.csv \
  --workers 32
```

### Pipeline Steps

1. **`fetch_xml.py`** - Download XML files from FormosanBank GitHub
   - Filters by source language
   - Two modes: full org scan or public release only (`--public`)
   - Parallel downloads with retry logic

2. **`clean_xml.py`** - Clean XML with official QC scripts
   - Auto-downloads latest QC scripts from FormosanBank/FormosanBank
   - Uses checksums to avoid redundant downloads
   - Runs standardization and validation

3. **`make_corpus.py`** - Extract parallel sentence pairs
   - Converts XML → CSV format
   - Handles "standard" vs "original" orthography
   - Preserves source metadata and dialect information

4. **`filter_split_corpus.py`** - Advanced preprocessing
   - **Text normalization**: Moses punctuation + NFKC + whitespace cleanup
   - **Safety filters**: Removes Chinese chars in Formosan text, punctuation-only rows, dialog spam
   - **Lexeme detection**: Identifies single-word pairs (→ train only, prevents vocab leakage)
   - **Deduplication**: Exact duplicate removal
   - **Fertility filtering**: Token-ratio outlier removal (0.2-8.0 for sentences)
   - **80/10/10 split**: Random train/validate/test split

---

## 📁 Repository Structure

```
processed_corpora/           # ✅ Ready-to-use corpora
├── ami_en_processed.csv     # Amis ↔ English  
├── ami_zh_processed.csv     # Amis ↔ Chinese
├── ...                      # (30 files total)
├── helpers/
│   ├── summary_stats.py            # Corpus analysis tool
│   └── big_corpus_for_tokenizer.py # Combine all languages
├── opennmt/
│   └── data/                # OpenNMT-formatted data (train/valid/test splits)
└── untrained/               # English pairs for languages without training data

scripts/
├── local/                   # Data pipeline scripts
│   ├── fetch_xml.py         # Step 1: Download FormosanBank XML
│   ├── clean_xml.py         # Step 2: Clean & standardize XML
│   ├── make_corpus.py       # Step 3: Extract parallel pairs → CSV
│   └── filter_split_corpus.py  # Step 4: Filter, clean, deduplicate, split
├── mt/                      # Machine translation training
│   ├── nllb/                # NLLB-200 (Transformer, recommended)
│   │   ├── prelims/setup_nllb200_formosan.py
│   │   ├── training/train_formosan_nllb200.py
│   │   └── eval/eval_formosan_nllb200.py
│   └── opennmt/             # OpenNMT-py (RNN seq2seq)
│       ├── setup/prep_opennmt.py
│       ├── setup/prep_spm.py
│       └── training/config.yaml
└── deprecated/
    └── mbart/               # mBART-50 (older experiments)

raw_corpora/                 # Intermediate CSV (before filtering)
bash/                        # HPC training scripts
build_corpora.sh            # Master pipeline orchestration
requirements.txt            # Python dependencies
```

---

## 📚 Academic Use

This corpus is designed for computational linguistics research on endangered languages:

- **Language Preservation**: Digital documentation of Indigenous Taiwanese languages
- **Low-Resource MT**: Machine translation model development for under-resourced languages
- **Multilingual NLP**: Cross-lingual representation learning and transfer learning
- **Corpus Linguistics**: Comparative analysis of Austronesian languages

### Citation

```bibtex
TBD!
```

### Data Source

Original XML data from [FormosanBank/FormosanBank](https://github.com/FormosanBank/FormosanBank) - Taiwan's comprehensive digital archive of Indigenous languages.

---

## 🔧 Requirements

**Core dependencies:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+ (for NLLB/mBART)
- OpenNMT-py 1.2.0 (for RNN models)
- SacreBLEU, sacremoses (for evaluation)
- pandas, tqdm, lxml

**Optional:**
- CUDA 11+ (for GPU training)
- SentencePiece (for OpenNMT tokenization)

See `requirements.txt` for full dependency list.

---

## 🆘 Troubleshooting

### Common Issues

**"No XML files found"**: Make sure you've set `GITHUB_TOKEN` in `.env` for fetching data

**"QC scripts download failed"**: Check internet connection or use `--force-update` flag

**"Out of memory" during training**: Reduce `--batch-size` or enable `--fp16` for NLLB

**Column detection errors**: Specify `--src-col` and `--tgt-col` explicitly

### Getting Help

- Check corpus statistics: `python processed_corpora/helpers/summary_stats.py`
- Review pipeline logs in generated directories
- Open GitHub issue for bugs or questions

**Ready to preserve and revitalize Indigenous languages through AI! 🚀**
