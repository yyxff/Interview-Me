"""
快速测试 explore_concept_bfs 工具逻辑。
运行：/opt/homebrew/Caskroom/miniconda/base/envs/interview-me/bin/python3 test_explore_concept.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rag.retrieval import explore_concept_bfs

cases = [
    "CPU Cache",       # 精确匹配
    "缓存",            # 模糊匹配（向量搜索）
    "线程",            # 精确匹配
    "不存在的概念XYZ", # 找不到的情况
]

for entity in cases:
    print(f"\n{'='*60}")
    print(f"输入：{entity}")
    result = explore_concept_bfs(entity)
    print(f"matched: {result['matched']}")
    print(f"邻居数: {len(result['neighbors'])}")
    print(result["summary"])
