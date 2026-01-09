# Task Instructions for VSCode Codex Assistant

## Mission

You need to run a **knowledge extraction pipeline** to process 6,108 medical research articles (articles #18325-24432) and extract structured knowledge graphs from them.

## Context

This is a **parallel extraction task**:
- **Main machine**: Already processing articles 1-12216 (PID 15083, running)
- **Your task**: Process articles 18325-24432 (second half)
- **Total**: 24,432 articles about heart transplantation

## What You Have

A complete extraction package at: `/home/user/SHU/assistant_package_v3_english.tar.gz` (21MB)

This package contains:
- ✅ Python script: `assistant_extract.py` (processes articles 18325-24432)
- ✅ Data file: `heart_tx_all_merged_v8.json` (78MB, all 24,432 articles)
- ✅ Schema: `english_medical_kg_schema.json` (entity and relation types)
- ✅ Config: `config.yaml` (API settings - needs API key)
- ✅ README: Detailed instructions

## Your Task: 5 Steps

### Step 1: Extract Package

```bash
cd /home/user/SHU
tar -xzf assistant_package_v3_english.tar.gz
cd assistant_package
```

### Step 2: Install Dependencies (3 packages only)

```bash
pip install openai pyyaml httpx
```

**IMPORTANT**: Do NOT run `pip install -r requirements.txt` (file doesn't exist and is not needed)

### Step 3: Configure API Key

Edit `automated_kg_pipeline/config.yaml` and replace the API key:

```yaml
llm:
  deepseek:
    api_key: "sk-YOUR-API-KEY-HERE"  # ← CHANGE THIS
    base_url: "https://yinli.one/v1"
    model: "deepseek-chat"
    max_tokens: 2048
    temperature: 0.1
```

**CRITICAL**: Ask the user for their DeepSeek API key. The main machine is using one key, so ideally use a different key to avoid rate limiting conflicts.

### Step 4: Create Required Directories

```bash
mkdir -p logs
mkdir -p cache/llm_raw_outputs
mkdir -p cache/parsed_triples
```

### Step 5: Run Extraction

```bash
# Option 1: Foreground (see output in real-time)
python3 automated_kg_pipeline/assistant_extract.py

# Option 2: Background (recommended for long runs)
nohup python3 -u automated_kg_pipeline/assistant_extract.py > logs/assistant_extraction.log 2>&1 &

# To monitor progress (if running in background)
tail -f logs/assistant_extraction.log
```

## What the Script Does

1. **Loads data**: Reads `heart_tx_all_merged_v8.json` (24,432 articles)
2. **Filters range**: Processes only articles 18325-24432 (6,108 articles)
3. **Extracts knowledge**: Calls DeepSeek API to extract entities and relations
4. **Outputs English**: Entities like "heart transplantation", "acute rejection", "immunosuppression"
5. **Saves results**:
   - Raw LLM output → `cache/llm_raw_outputs/{article_id}_raw.json`
   - Parsed triples → `cache/parsed_triples/{article_id}_triples.json`
6. **Checkpoint**: Saves progress to `cache/assistant_checkpoint.json` (independent from main machine)

## Key Features

### ✅ Intelligent Retry
- If API returns 503/429 (rate limit), waits and retries (1s→3s→5s→10s→30s→60s→2min→5min→10min)
- If balance is insufficient, saves checkpoint and stops gracefully

### ✅ Resume from Checkpoint
- If interrupted, rerun the same command - it will resume from last processed article
- Checkpoint file: `cache/assistant_checkpoint.json`

### ✅ No Conflicts with Main Machine
- Different checkpoint file (assistant_checkpoint.json vs extraction_checkpoint.json)
- Different article ranges (no overlap)
- Shared cache directories (named by article_id, so no collisions)

## Expected Output

### When Running Correctly

```
============================================================
Assistant Knowledge Extraction - Processing 18325-24432
============================================================
Total articles: 6108 (second half, ~1/4)
Processed: 0
Remaining: 6108
Start time: 2026-01-09T14:20:00
============================================================

[18325/24432] Processing article: 40960031 [Indexed for MEDLINE]
  Text length: 2453 characters
HTTP Request: POST https://yinli.one/v1/chat/completions "HTTP/1.1 200 OK"
  ✓ Extraction successful: 11 entities, 10 relations

[18326/24432] Processing article: ...
```

### Example English Output (Correct)

```json
{
  "entities": [
    {"name": "heart transplantation", "type": "surgical_procedure"},
    {"name": "acute rejection", "type": "complication"},
    {"name": "tacrolimus", "type": "medication"}
  ],
  "relations": [
    {"head": "tacrolimus", "relation": "treats", "tail": "acute rejection"}
  ]
}
```

## Progress Monitoring

### Check Files Processed
```bash
ls cache/parsed_triples/ | wc -l
# Should grow from 0 to 6108
```

### Check Current Progress
```bash
cat cache/assistant_checkpoint.json
# Shows: processed_ids, last_index, start_time
```

### Check for Errors
```bash
tail -50 logs/assistant_extraction.log
# Look for "❌" or "ERROR"
```

### Monitor Real-time
```bash
tail -f logs/assistant_extraction.log
# See live extraction progress
```

## Expected Performance

- **Speed**: ~0.007 articles/second (1 article every ~14-20 seconds)
- **Time**: 6,108 articles ÷ 0.007/sec ≈ **12-17 hours**
- **Cost**: ~4.9M tokens ≈ **5-10 RMB** (DeepSeek pricing)
- **API calls**: 6,108 calls (stays under 60 req/min rate limit with 1s sleep)

## Troubleshooting

### Problem 1: SSL Certificate Error
```
TLS_error:CERTIFICATE_VERIFY_FAILED
```
**Solution**: Script already disables SSL verification (`verify=False`), this is a warning, not an error. Ignore it.

### Problem 2: API Key Invalid
```
Error: Incorrect API key provided
```
**Solution**: Double-check the API key in `config.yaml`. Make sure it starts with `sk-`.

### Problem 3: Module Not Found
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: Run `pip install openai pyyaml httpx`

### Problem 4: Rate Limiting
```
⚠️  API rate limit/503 error, waiting 30 seconds and retrying
```
**Solution**: This is normal. Script handles it automatically. Just wait.

### Problem 5: Balance Insufficient
```
❌ Insufficient balance! Progress saved
```
**Solution**:
1. Recharge DeepSeek account
2. Run the same command again
3. Script will resume from checkpoint

## Success Indicators

### You'll know it's working if:
1. ✅ You see articles being processed sequentially (18325, 18326, 18327...)
2. ✅ Files appear in `cache/parsed_triples/` (use `ls | wc -l` to count)
3. ✅ Entities and relations are in **English** (not Chinese)
4. ✅ Checkpoint file updates regularly: `cache/assistant_checkpoint.json`
5. ✅ Log shows "✓ Extraction successful: X entities, Y relations"

### Final completion message:
```
============================================================
Current Progress
============================================================
Processed articles: 6108/6108
Completion: 100.00%
Total entities: ~52,000
Total relations: ~44,000
Errors: <50
============================================================
```

## Critical Notes

### 🔴 Must Use English Output
The script is configured to extract in **English**. If you see Chinese entities (like "心脏移植"), something is wrong. The output should look like "heart transplantation", "acute rejection", etc.

### 🔴 Independent Checkpoint
Your checkpoint file is `assistant_checkpoint.json`. The main machine uses `extraction_checkpoint.json`. **Do not delete or modify the main machine's checkpoint.**

### 🔴 Shared Cache is OK
Both machines write to `cache/parsed_triples/` and `cache/llm_raw_outputs/`, but files are named by article_id, so there's no conflict since you're processing different articles.

### 🔴 Process Persistence
Once started with `nohup ... &`, the process runs independently. Even if VSCode crashes or you disconnect, the extraction continues. Check progress using:
```bash
tail -f logs/assistant_extraction.log
ps aux | grep assistant_extract
```

## Questions to Ask User

Before starting, confirm:
1. **Do you have a DeepSeek API key?** (Need to add it to config.yaml)
2. **Is the key different from the main machine's key?** (To avoid rate limit conflicts)
3. **Do you want to run in foreground or background?** (Background recommended for 12-17 hour run)

## Summary

**Goal**: Extract knowledge from 6,108 medical articles (18325-24432)
**Time**: 12-17 hours
**Cost**: ~5-10 RMB
**Output**: English entities and relations in `cache/parsed_triples/`
**Resume**: Automatic via checkpoint if interrupted

**Your role**: Set up the environment, configure API key, start the script, and monitor progress. The script handles all extraction logic, retries, and checkpointing automatically.
