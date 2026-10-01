basedir = './Data/data/'

mathqa_path = basedir + "mathqa.json"
xcopa_path = basedir + "xcopa/data-gmt/zh/test.zh.jsonl"
commensenseqa_path = basedir + "commenseqa.json"
mgsm_en_path = basedir + "mgsm_en.json"
cmb_path = basedir + "CMB/CMB-Exam/CMB-val/CMB-val-merge.json"

translatedBaseDir = './Data/v2_translated'

# English paraphrases produced by run_rewrite.py, one file per dataset and version:
#   version 1: {rewrittenBaseDir}/{datasetType}_english.json
#   version n: {rewrittenBaseDir}/{datasetType}_english_v{n}.json  (n >= 2, written after seeing versions 1..n-1)
rewrittenBaseDir = './Data/rewritten'

def rewriteFileName(datasetType: str, version: int = 1) -> str:
    return f"{datasetType}_english.json" if version == 1 else f"{datasetType}_english_v{version}.json"