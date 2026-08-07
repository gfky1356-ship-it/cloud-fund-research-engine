# Cloud Fund Research Engine

这是一个 Colab / Google Drive friendly 的退休基金研究引擎。

目标流程：

`Python / Colab` -> `Google Drive persistent cache + output` -> `ChatGPT Google Drive connector 读取 latest CSV/JSON`

## Google Drive 输出路径

Colab 挂载 Google Drive 后，默认输出会写到：

- `My Drive/AI_Fund_Research/output/fund_daily_summary.csv`
- `My Drive/AI_Fund_Research/output/fund_daily_summary.json`
- `My Drive/AI_Fund_Research/cache/fund_research.sqlite`

本地 smoke test 没有 Google Drive 时，会写到项目内：

- `CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex/AI_Fund_Research/output/fund_daily_summary.csv`
- `CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex/AI_Fund_Research/output/fund_daily_summary.json`

## Colab 从零运行

1. 把整个 `CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex` folder 上传到 Google Drive，建议位置：

   `My Drive/AI_Fund_Research/code/CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex/`

2. 在 Colab 新建 notebook，或上传本项目的 `colab_run_fund_research.ipynb`。

3. 运行 notebook。它会：

   - mount Google Drive
   - install requirements
   - run `quick-daily`
   - 写出 latest CSV + JSON

## Colab 命令

```bash
cd /content/drive/MyDrive/AI_Fund_Research/code/CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex
pip install -r requirements_colab.txt
python fund_research_engine.py --mode quick-daily --storage-root /content/drive/MyDrive/AI_Fund_Research
```

Weekend 深度刷新：

```bash
python fund_research_engine.py --mode deep-weekend --storage-root /content/drive/MyDrive/AI_Fund_Research
```

## 本地验证命令

```bash
cd "/Users/ky/Documents/AI-on iCloud/PROJECT/DAILY NEWS READ/CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_colab.txt pytest
python fund_research_engine.py --mode quick-daily
pytest
```

## Ranking 规则

硬条件：

`5Y Max Drawdown <= 5%`

如果历史不足 5 年，使用 inception-to-date MaxDD，并在 latest output 标记 `History_LT_5Y=true`。

`MaxDD < -5%` 直接 fail，不参与 score。SPY 永远只是 benchmark，不进入退休 shortlist。

Score 权重：

- MaxDD: 45%
- CAGR: 25%
- Yield: 15%
- Fee: 10%
- Volatility: 5%

## 数据源与 fallback

第一版真实 NAV/price 来源：

- Yahoo Finance chart endpoint, no API key

如果 source 拒绝访问或某个 symbol 无数据：

- 不 silent fail
- `output/YYYY-MM-DD_fund_run_status.json` 会记录错误
- `logs/fund_research_run.jsonl` 会记录每个 symbol 的状态

Slow-changing metadata 目前来自 `config/fund_universe.csv` seed：

- fee
- yield
- fund type
- currency
- SGD hedged status
- duration

后续可以在 `deep-weekend` 增加 issuer factsheet refresh。
