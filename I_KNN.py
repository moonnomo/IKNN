import csv
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
import math


def chi_feature_aggregation(X_tfidf, y, feature_names, top_k=1000):
    """
    X_tfidf: sparse matrix (n_samples, n_features)
    y: labels (0/1)
    feature_names: list[str]
    """

    chi_vals, _ = chi2(X_tfidf, y)

    # 选取高 CHI 特征
    top_idx = np.argsort(chi_vals)[-top_k:]
    selected_features = [feature_names[i] for i in top_idx]

    return top_idx, selected_features

def chi_pattern_aggregation(X, chi_idx, group_size=5):
    """
    真正安全的 CHI Pattern 聚合
    输出 shape: (n_samples, n_patterns)
    """
    chi_idx = np.array(chi_idx)
    n_samples = X.shape[0]
    patterns = []

    for i in range(0, len(chi_idx), group_size):
        idx_group = chi_idx[i:i + group_size]

        # 提取子矩阵
        sub = X[:, idx_group]

        # 关键：转为 array，再 mean
        if hasattr(sub, "toarray"):
            sub = sub.toarray()

        pattern = np.mean(sub, axis=1)   # (n_samples,)
        patterns.append(pattern.reshape(-1, 1))

    # (n_samples, n_patterns)
    return np.hstack(patterns)



from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np

def compress_majority_class(X, y, minority_label=1, ratio=1.0):
    """
    ratio: 多数类压缩后与少数类的比例 (1.0 表示 1:1, 2.0 表示 2:1)
    """
    y = np.asarray(y).ravel()
    X_min = X[y == minority_label]
    X_maj = X[y != minority_label]

    n_min = X_min.shape[0]
    n_maj = X_maj.shape[0]

    # 目标聚类数量
    n_clusters = int(n_min * ratio)
    if n_clusters > n_maj:
        n_clusters = n_maj

    # ===== 直接对原始特征进行 KMeans =====
    # 因为特征只有 200 维，直接聚类效果更精准
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_maj)

    # 聚类中心即为“代表样本”
    centers = kmeans.cluster_centers_
    # 确保没有因浮点运算产生的负值
    centers = np.maximum(centers, 0)

    # ===== 改进的权重 W =====
    # cluster_sizes[i] 代表该中心点代表了多少个原始多数类样本
    cluster_sizes = np.bincount(kmeans.labels_)
    
    # 逻辑：每个中心的权重 = 它代表的样本数 / 理论平均样本数
    # 这样 W 会在 1.0 附近波动，不会过小
    avg_represent = n_maj / n_clusters
    weights = cluster_sizes / avg_represent

    # ===== 合并训练集 =====
    X_new = np.vstack([X_min, centers])
    y_new = np.array([1] * n_min + [0] * len(centers))
    W = np.concatenate([np.ones(n_min), weights])

    return X_new, y_new, W


def load_depression_set_from_csv(filepath):
    """从CSV文件加载抑郁词汇集合"""
    depression_set = set()
    
    try:
        # 尝试UTF-8编码
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            # ... 其他代码
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GBK编码
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 2:
                        word = row[0].strip()
                        is_depression = row[1].strip()
                        
                        if is_depression == '1':  # 只添加抑郁词汇
                            depression_set.add(word)
        except UnicodeDecodeError:
            # 如果GBK也失败，尝试其他编码
            with open(filepath, 'r', encoding='gb2312') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 2:
                        word = row[0].strip()
                        is_depression = row[1].strip()
                        
                        if is_depression == '1':  # 只添加抑郁词汇
                            depression_set.add(word)
    
    return depression_set

PSY_DICT = load_depression_set_from_csv("Dataset/psy_dict.csv")

def psychological_weight(vector, feature_names):
    w = 1.0
    indices = np.nonzero(vector)[0]  # 非零索引
    for idx in indices:
        if feature_names[idx] in PSY_DICT:
            w *= 1.5
    return min(w, 3.0)

def asknn_similarity(x, y, W_i, sigma):
    cos = cosine_similarity([x], [y])[0][0]
    d = 1 - cos
    return cos * W_i * math.exp(-(d**2) / (2 * sigma**2))

def adaptive_sigma(distances):
    return np.std(distances) + 1e-6

def I_KNN_predict(X_subtrain_text, X_val_text, X_test_text, y_subtrain, y_test, y_val, 
                  iknn_k = 10,
                  iknn_alpha = 0.8,
                  iknn_beta = 1.2):
    # 1) TF-IDF 在 subtrain 上拟合
    vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.95)
    X_subtrain_tfidf = vectorizer.fit_transform(X_subtrain_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    feature_names = vectorizer.get_feature_names_out()

    # 2) CHI 选特征并做 Pattern 聚合（基于 subtrain）
    idx, selected_feats = chi_feature_aggregation(X_subtrain_tfidf, y_subtrain, feature_names, top_k=1000)
    X_subtrain_pattern = chi_pattern_aggregation(X_subtrain_tfidf, idx, group_size=5)
    X_val_pattern = chi_pattern_aggregation(X_val_tfidf, idx, group_size=5)
    X_test_pattern = chi_pattern_aggregation(X_test_tfidf, idx, group_size=5)

    # flatten 若有多余维度
    if X_subtrain_pattern.ndim > 2:
        X_subtrain_pattern = X_subtrain_pattern.reshape(X_subtrain_pattern.shape[0], -1)
    if X_val_pattern.ndim > 2:
        X_val_pattern = X_val_pattern.reshape(X_val_pattern.shape[0], -1)
    if X_test_pattern.ndim > 2:
        X_test_pattern = X_test_pattern.reshape(X_test_pattern.shape[0], -1)

    X_subtrain_pattern = np.asarray(X_subtrain_pattern)
    X_val_pattern = np.asarray(X_val_pattern)
    X_test_pattern = np.asarray(X_test_pattern)

    print("Pattern shapes:", X_subtrain_pattern.shape, X_val_pattern.shape, X_test_pattern.shape)

    # 3) 降维（TruncatedSVD）以去噪，保持 KNN 更稳定
    svd_dim = 100
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_subtrain_low = svd.fit_transform(X_subtrain_pattern)
    X_val_low = svd.transform(X_val_pattern)
    X_test_low = svd.transform(X_test_pattern)

    # 4) 计算样本级心理词典权重（基于 TF-IDF 原始空间）
    psy_weights_subtrain = compute_sample_psy_weights_from_tfidf(X_subtrain_tfidf, feature_names, PSY_DICT, factor=1.5, max_w=3.0)

    print("Psy weights (subtrain) stats:", np.min(psy_weights_subtrain), np.max(psy_weights_subtrain))

    # ============ KMKNN 压缩 + I-KNN ============
    # compress_majority_class expects pattern-space features; 我们传入低维后或直接传 higher-dim
    X_new, y_new, W_new = compress_majority_class(X_subtrain_low, y_subtrain, minority_label=1)
    # For psy weights for new set: minority samples keep psy from subtrain; cluster centers -> set = 1.0 (conservative)
    psy_new = np.concatenate([psy_weights_subtrain[y_subtrain==1], np.ones(len(X_new) - np.sum(y_subtrain==1))])

    # compute iknn val/test scores using compressed train
    iknn_val_scores_km = iknn_raw_scores(X_new, y_new, W_new, psy_new, X_val_low, k=iknn_k, alpha=iknn_alpha, beta=iknn_beta)
    # train SVM on original subtrain_low (or on compressed X_new; here use original to keep SVM stable)
    svm_cal2 = CalibratedClassifierCV(SVC(kernel='rbf', C=1.0), cv=3)
    svm_cal2.fit(X_subtrain_low, y_subtrain)
    svm_val_proba2 = svm_cal2.predict_proba(X_val_low)[:,1]

    # meta2
    X_stack_val2 = np.vstack([svm_val_proba2, iknn_val_scores_km]).T
    meta2 = LogisticRegression(max_iter=500)
    meta2.fit(X_stack_val2, y_val)

    # test
    iknn_test_scores_km = iknn_raw_scores(X_new, y_new, W_new, psy_new, X_test_low, k=iknn_k, alpha=iknn_alpha, beta=iknn_beta)
    svm_test_proba2 = svm_cal2.predict_proba(X_test_low)[:,1]
    X_stack_test2 = np.vstack([svm_test_proba2, iknn_test_scores_km]).T
    y_test_pred_stack2 = meta2.predict(X_stack_test2)
    result_stack2 = evaluate_model(y_test, y_test_pred_stack2)
    print("STACK (KMKNN path) result:", result_stack2)
    return result_stack2


# ---------- 辅助函数：返回 I-KNN raw score（0-1） ----------
def iknn_raw_scores(X_train, y_train, W, psy_train_weights, X_test, k=10, alpha=0.8, beta=1.2):
    """
    返回 length-n_test 的 raw scores (pos_sum / total_sum)
    仅依赖 numpy arrays; X_train, X_test 都是 2D numpy arrays
    """
    n_train = X_train.shape[0]
    scores = []
    for j in range(X_test.shape[0]):
        vec = X_test[j].reshape(1, -1)
        sims = cosine_similarity(X_train, vec).flatten()
        sims = np.maximum(0, sims)
        topk_idx = np.argsort(sims)[-k:]
        weighted = sims[topk_idx] * (1.0 + alpha * (W[topk_idx] - 1.0)) * (psy_train_weights[topk_idx] ** beta)
        pos = weighted[y_train[topk_idx] == 1].sum()
        total = weighted.sum() + 1e-8
        scores.append(pos / total)
    return np.array(scores)

# ---------- 计算样本级心理权重（在 TF-IDF 原始空间） ----------
def compute_sample_psy_weights_from_tfidf(X_tfidf, feature_names, psy_dict, factor=1.5, max_w=3.0):
    n = X_tfidf.shape[0]
    weights = np.ones(n, dtype=float)
    psy_mask = np.array([1 if fname in psy_dict else 0 for fname in feature_names])
    for i in range(n):
        row = X_tfidf[i]
        if hasattr(row, "indices"):
            nz = row.indices
        else:
            nz = np.nonzero(row.toarray())[1]
        count = int(np.sum(psy_mask[nz]))
        if count > 0:
            w = factor ** count
            weights[i] = min(w, max_w)
    return weights