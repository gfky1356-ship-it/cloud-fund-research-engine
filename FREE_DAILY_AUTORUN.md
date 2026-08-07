# Free Daily Autorun Setup

免费自动运行推荐路线：

`GitHub Actions schedule` -> `Python fund engine` -> `Google Drive latest CSV/JSON`

这个方案不需要 Mac 开机，也不依赖 Colab runtime 常驻。

## 1. 上传项目到 GitHub

把 `CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex` 放进一个 GitHub repo。

然后把本项目里的 workflow template 复制到 repo root：

```bash
mkdir -p .github/workflows
cp CLOUD_FUND_RESEARCH_ENGINE_20260807_by_codex/.github_workflow_fund_daily.yml .github/workflows/fund_daily.yml
```

## 2. 建立 Google Drive output folder

在 Google Drive 建议建立：

`My Drive/AI_Fund_Research/output`

打开这个 folder，复制 URL 里的 folder id：

`https://drive.google.com/drive/folders/<THIS_IS_FOLDER_ID>`

## 3. 建立 Google service account

在 Google Cloud Console:

1. Create project
2. Enable Google Drive API
3. Create service account
4. Create JSON key
5. 复制 service account email，类似：

   `fund-drive-writer@your-project.iam.gserviceaccount.com`

6. 回到 Google Drive，把 `AI_Fund_Research/output` folder 分享给这个 email，权限给 `Editor`

## 4. GitHub Secrets

在 GitHub repo:

`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

新增三个 secrets：

`GOOGLE_SERVICE_ACCOUNT_JSON`

- 内容是完整 service account JSON key

`GOOGLE_DRIVE_CSV_FILE_ID`

- 内容是 `fund_daily_summary.csv` 的 Drive file id

`GOOGLE_DRIVE_JSON_FILE_ID`

- 内容是 `fund_daily_summary.json` 的 Drive file id

## 5. 启用 daily run

workflow 已设置：

`30 22 * * *` UTC

等于 Singapore time 每天 06:30。

你也可以到 GitHub:

`Actions` -> `Daily Fund Research` -> `Run workflow`

手动试跑一次。

## 6. ChatGPT 最终读取路径

GitHub Actions 每天会覆盖更新 Google Drive folder 里的固定文件：

- `My Drive/AI_Fund_Research/output/fund_daily_summary.csv`
- `My Drive/AI_Fund_Research/output/fund_daily_summary.json`

ChatGPT 后续只读这两个 latest files。

## Notes

- GitHub Actions cron 有时不会精确到分钟，但通常足够做每日任务。
- GitHub Actions cache 用来保存 SQLite NAV cache，避免每天重新下载全历史。
- 如果 Drive upload 失败，GitHub artifact 里仍会保留当天 CSV/JSON/status backup。
