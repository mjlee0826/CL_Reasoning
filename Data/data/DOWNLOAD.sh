#!/usr/bin/env bash

# Download XCOPA dataset
if [ ! -d "xcopa" ]; then
    git clone https://github.com/cambridgeltl/xcopa.git xcopa
fi

echo "Dataset download completed!"
echo "Available datasets:"
echo "- MathQA: mathqa.json"
echo "- MMLU: mmlu/test/ (extracted from mmlu_data.tar)"
echo "- TruthfulQA: truthfulqa/truthful_qa"
echo "- XCOPA: xcopa/"
echo "- MGSM: mgsm_en.json"
echo "-CommenseQA: commenseqa.json"