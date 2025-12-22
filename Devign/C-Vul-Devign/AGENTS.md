# Agents Configuration

## Git Commit Rules

When making git commits, do NOT include:
- `Amp-Thread-ID:` lines
- `Co-authored-by: Amp <amp@ampcode.com>` lines

Keep commit messages clean and simple.

## Project Info

- **Project**: BiGRU Vulnerability Detection for C code (Devign dataset)
- **Python**: Use `python3` command (not `python`)
- **Dataset**: `/media/hdi/Hdii/Work/C Vul Devign/Dataset/devign slice/`
- **Output**: `/media/hdi/Hdii/Work/C Vul Devign/output /`

## Commands

```bash
# Check Python syntax
python3 -m py_compile <file.py>

# Zip pipeline for Kaggle
zip -r devign_pipeline.zip devign_pipeline/
```
