import re
import jieba

USELESS_PATTERNS = [
    r"微博内容[:：]?",
    r"转发微博",
    r"全文",
    r"网页链接"
]

EMOJI_MAP = {
    "允悲": "悲伤",
    "泪": "悲伤",
    "😭": "悲伤",
    "😢": "悲伤",
    "二哈": "中性",
    "哈哈": "中性"
}

STOPWORDS = set([
    "的", "了", "是", "在", "和", "也", "就", "都",
    "而", "及", "与", "着", "呢", "吧", "啊"
])

def clean_weibo_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    for p in USELESS_PATTERNS:
        text = re.sub(p, "", text)

    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#([^#]+)#", r"\1", text)
    text = re.sub(r"http[s]?://\S+", "", text)

    for emo, rep in EMOJI_MAP.items():
        text = text.replace(emo, f" {rep} ")

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str):
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in STOPWORDS]

def load_txt_dataset(dep_path, non_dep_path):
    texts = []
    labels = []

    # 抑郁样本 → label = 1
    with open(dep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned = clean_weibo_text(line)
            tokens = tokenize(cleaned)
            texts.append(" ".join(tokens))
            labels.append(1)

    # 非抑郁样本 → label = 0
    with open(non_dep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned = clean_weibo_text(line)
            tokens = tokenize(cleaned)
            texts.append(" ".join(tokens))
            labels.append(0)

    return texts, labels
