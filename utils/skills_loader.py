import re
from pathlib import Path
from typing import Dict, List, Tuple


DOMAIN_ALIASES = {
    "概率论": [
        "依概率", "依分布", "几乎必然", "随机变量", "分布函数", "概率收敛",
        "中心极限定理", "大数定律", "Slutsky", "贝叶斯", "全概率",
        "二项分布", "泊松分布", "指数分布", "正态分布", "矩母函数", "特征函数",
    ],
    "随机过程": [
        "Brownian", "布朗", "Wiener", "首达时", "停时", "鞅", "Markov链",
        "马尔可夫", "Poisson过程", "泊松过程", "生灭过程", "更新过程", "随机游走",
    ],
    "数学分析": [
        "一致收敛", "函数列", "函数项级数", "逐点收敛", "极限函数", "可导",
        "导数列", "Taylor", "泰勒", "幂级数", "含参积分", "广义积分", "上确界",
    ],
    "统计推断": [
        "最大似然", "MLE", "置信区间", "假设检验", "拒绝域", "显著性水平",
        "p值", "p-value", "样本", "估计量", "无偏", "方差估计",
    ],
    "复分析": ["留数", "解析", "全纯", "Cauchy", "柯西", "Laurent", "洛朗", "极点"],
    "抽象代数": ["有限域", "群", "环", "理想", "同态", "子群", "正规子群", "域扩张"],
    "高等代数": ["矩阵", "特征值", "特征向量", "线性空间", "秩", "行列式", "二次型"],
    "常微分方程": ["常微分", "ODE", "初值问题", "通解", "特解", "Wronskian"],
    "偏微分方程": ["偏微分", "PDE", "热方程", "波动方程", "Laplace方程", "边值问题"],
    "泛函分析": ["Banach", "Hilbert", "有界线性算子", "泛函", "弱收敛", "紧算子"],
    "拓扑学": ["拓扑", "开集", "闭集", "紧致", "连通", "同胚", "基本群"],
    "微分几何": ["流形", "曲率", "测地线", "联络", "切空间", "第一基本形式"],
    "数值分析": ["插值", "Newton", "迭代", "误差", "数值积分", "Runge-Kutta"],
    "测度积分": ["测度", "可测", "Lebesgue", "勒贝格", "几乎处处", "支配收敛", "Fatou"],
    "运筹学": ["线性规划", "单纯形", "对偶", "运输问题", "排队", "决策树"],
    "离散数学": ["图", "树", "组合", "递推", "生成函数", "布尔", "命题逻辑"],
    "线性回归": ["回归", "最小二乘", "OLS", "残差", "t检验", "F检验", "R方"],
}


class SkillsLoader:
    DEFAULT_BASE = Path(__file__).resolve().parent.parent / "skills_pythonscripts"

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else self.DEFAULT_BASE
        self.categories = self._scan_category_names()
        self.keywords_index = self._build_keywords_index()
        self._doc_cache: Dict[str, str] = {}
        self._script_cache: Dict[str, str] = {}
        self._embedding_index = None

    def _scan_category_names(self) -> List[str]:
        return sorted(p.name for p in self.base_path.iterdir() if p.is_dir())

    def _build_keywords_index(self) -> Dict[str, List[str]]:
        idx = {}
        for cat in self.categories:
            kw = set()
            kw.add(cat)
            kw.update(DOMAIN_ALIASES.get(cat, []))
            md = self.base_path / cat / f"{cat}skill.md"
            if md.exists():
                text = md.read_text(encoding="utf-8")
                for line in text.splitlines():
                    s = line.strip().lstrip("#").strip()
                    s = re.sub(r"^\d+\.\s*", "", s)
                    if 1 < len(s) <= 12 and not s.startswith("|"):
                        kw.add(s)
            idx[cat] = list(kw)
        return idx

    def get_skill_document(self, category: str) -> str:
        if category not in self._doc_cache:
            f = self.base_path / category / f"{category}skill.md"
            self._doc_cache[category] = f.read_text(encoding="utf-8")
        return self._doc_cache[category]

    def get_validation_script(self, category: str) -> str:
        if category not in self._script_cache:
            f = self.base_path / category / f"{category}验证示例.py"
            self._script_cache[category] = f.read_text(encoding="utf-8")
        return self._script_cache[category]

    def find_candidate_categories(self, problem: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = {}
        for cat, kws in self.keywords_index.items():
            hits = [kw for kw in kws if kw and kw in problem]
            alias_hits = [kw for kw in DOMAIN_ALIASES.get(cat, []) if kw and kw in problem]
            category_hit = 1.0 if cat in problem else 0.0
            scores[cat] = len(hits) + len(alias_hits) + category_hit
        return sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:top_k]

    def get_embedding_index(self):
        if self._embedding_index is None:
            from utils.category_embedding_index import CategoryEmbeddingIndex
            self._embedding_index = CategoryEmbeddingIndex(self)
        return self._embedding_index
