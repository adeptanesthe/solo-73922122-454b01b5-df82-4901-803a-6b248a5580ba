"""Dawid–Skene 核心算法（NumPy 实现）。

约定
----
- 类别：0 = 无坑，1 = 有坑；类别先验固定为各 0.5。
- 混淆矩阵 conf[j][c][k] = P(标注员 j 投票 k | 真实类别 c)，M 步加一平滑。
  - 误标率（false positive rate）= conf[j][0][1]：无坑被投成有坑。
  - 漏标率（false negative rate）= conf[j][1][0]：有坑被投成无坑。
- 专家标签固定：E 步后把专家照片的后验重置为 one-hot，不再更新。
- 缺票不计：只对实际存在的投票做累加，不为缺失投票补默认值。
- 收敛：混淆矩阵相邻两次迭代的最大绝对变化 < TOL。未收敛的组不采用其
  混淆矩阵；可在同一请求的共享补算额度内继续迭代，收敛后才采用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASS_PRIOR = (0.5, 0.5)  # 类别先验各半
SMOOTHING = 1.0           # 混淆矩阵加一平滑
MAX_ITER = 200            # 每组首次拟合的原上限
EXTRA_ITER = 200          # 每个请求的补算额度（相关未收敛组共用）
TOL = 1e-9                # 原收敛精度：相邻两次混淆矩阵最大变化


@dataclass(frozen=True)
class Vote:
    photo_idx: int
    annotator_idx: int
    label: int  # 0 = 无坑，1 = 有坑


@dataclass
class ExtraBudget:
    """一次请求内相关未收敛组共用的补算额度（按迭代轮数计）。"""

    limit: int = EXTRA_ITER
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def spend(self, n: int = 1) -> None:
        if n < 0 or n > self.remaining:
            raise ValueError("补算额度不足")
        self.used += n


class _EMFit:
    """一次 Dawid–Skene EM 拟合的可续算状态。

    每调用一次 :meth:`step` 执行一轮 E 步 + M 步；因此既可用 ``run`` 一次
    跑完首轮额度，也可在补算额度内逐轮推进（多组轮转公平分享额度）。
    """

    def __init__(self, votes, n_photos, n_annotators, expert_labels):
        self.n_photos = n_photos
        self.n_annotators = n_annotators
        self.log_prior = np.log(np.asarray(CLASS_PRIOR))
        self.expert_labels = dict(expert_labels)

        self.vp = np.fromiter((v.photo_idx for v in votes), dtype=np.int64,
                              count=len(votes))
        self.va = np.fromiter((v.annotator_idx for v in votes), dtype=np.int64,
                              count=len(votes))
        self.vl = np.fromiter((v.label for v in votes), dtype=np.int64,
                              count=len(votes))

        self.confusion = np.array(
            [[[0.7, 0.3], [0.3, 0.7]]] * n_annotators, dtype=float)

        self.posteriors = np.broadcast_to(
            np.asarray(CLASS_PRIOR, dtype=float), (n_photos, 2)).copy()
        for i, c in self.expert_labels.items():
            self.posteriors[i] = (0.0, 1.0) if c == 1 else (1.0, 0.0)

        self.n_iter = 0
        self.converged = False
        self.delta = float("inf")

    def step(self, tol: float = TOL) -> bool:
        """执行一轮 E+M；返回本轮后是否达到收敛精度。"""
        # ---- E 步：更新未确认照片的后验（专家照片保持固定）----
        log_post = np.broadcast_to(self.log_prior, (self.n_photos, 2)).copy()
        if len(self.vp):
            np.add.at(log_post, self.vp,
                      np.log(self.confusion[self.va, :, self.vl]))
        log_post -= log_post.max(axis=1, keepdims=True)
        np.exp(log_post, out=log_post)
        log_post /= log_post.sum(axis=1, keepdims=True)
        self.posteriors[:] = log_post
        for i, c in self.expert_labels.items():
            self.posteriors[i] = (0.0, 1.0) if c == 1 else (1.0, 0.0)

        # ---- M 步：加一平滑更新混淆矩阵 ----
        new_conf = np.full((self.n_annotators, 2, 2), SMOOTHING)
        if len(self.vp):
            weight = self.posteriors[self.vp]  # (n_votes, 2)
            for c in (0, 1):
                np.add.at(new_conf[:, c, :], (self.va, self.vl), weight[:, c])
        new_conf /= new_conf.sum(axis=2, keepdims=True)

        self.delta = float(np.abs(new_conf - self.confusion).max())
        self.confusion = new_conf
        self.n_iter += 1
        self.converged = self.delta < tol
        return self.converged

    def run(self, max_iter: int, tol: float = TOL) -> bool:
        """至多再执行 ``max_iter`` 轮；已收敛时提前停止。"""
        for _ in range(max_iter):
            if self.converged:
                break
            self.step(tol)
        return self.converged


def fit_dawid_skene(votes, n_photos, n_annotators, expert_labels,
                    max_iter=MAX_ITER, tol=TOL):
    """EM 拟合 Dawid–Skene 模型。

    参数
    ----
    votes : list[Vote]               组内投票（photo/annotator 为局部下标）
    n_photos, n_annotators : int     组内照片数、标注员数
    expert_labels : dict[int, int]   局部照片下标 -> 0/1，拟合中固定

    返回
    ----
    (confusion, posteriors, n_iter, converged)
    confusion: (n_annotators, 2, 2)；posteriors: (n_photos, 2)
    ``n_iter`` 是本次调用实际执行的轮数；已收敛时提前停止。
    """
    fit = _EMFit(votes, n_photos, n_annotators, expert_labels)
    converged = fit.run(max_iter, tol)
    return fit.confusion, fit.posteriors, fit.n_iter, converged


@dataclass
class GroupFit:
    """一个连通组的拟合状态。

    ``eligible`` 表示组内正、负专家标签齐全、可以拟合；``converged`` 表示
    已在原 200 轮或补算额度内达到 TOL。只有 ``eligible and converged`` 的
    组，其混淆矩阵才会被采用。
    """

    photos: tuple[int, ...]
    annotators: tuple[int, ...]
    expert_labels: dict[int, int]
    eligible: bool
    fit: _EMFit | None = field(default=None, repr=False)
    extra_iter: int = 0

    @property
    def converged(self) -> bool:
        return bool(self.fit is not None and self.fit.converged)

    @property
    def n_iter(self) -> int:
        return self.fit.n_iter if self.fit is not None else 0

    @property
    def confusion(self):
        """达到收敛精度后才返回混淆矩阵，否则返回 None。"""
        return self.fit.confusion if self.converged else None


def connected_groups(votes, n_photos, n_annotators):
    """并查集：按投票边把相连的照片与标注员分组。

    返回 [{"photos": set, "annotators": set, "votes": list}]，只含至少有一条
    投票边的节点；没有任何投票的照片/标注员不属于任何组。
    """
    parent = list(range(n_photos + n_annotators))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in votes:
        ra, rb = find(v.photo_idx), find(n_photos + v.annotator_idx)
        if ra != rb:
            parent[rb] = ra

    groups = {}
    for v in votes:
        root = find(v.photo_idx)
        g = groups.setdefault(root, {"photos": set(), "annotators": set(), "votes": []})
        g["photos"].add(v.photo_idx)
        g["annotators"].add(v.annotator_idx)
        g["votes"].append(v)
    return list(groups.values())


def build_group_fits(votes, n_photos, n_annotators, expert_labels):
    """分组并让每组先按原上限（至多 200 轮）独立拟合。

    返回按（最小照片下标、最小标注员下标）排序的 :class:`GroupFit` 列表。
    缺少正或负专家标签的组标记为 ``eligible=False``，不拟合。
    """
    raw_groups = connected_groups(votes, n_photos, n_annotators)
    raw_groups.sort(key=lambda g: (min(g["photos"]), min(g["annotators"])))

    group_fits: list[GroupFit] = []
    for g in raw_groups:
        photos = tuple(sorted(g["photos"]))
        annotators = tuple(sorted(g["annotators"]))
        g_expert = {p: c for p, c in expert_labels.items() if p in g["photos"]}
        eligible = set(g_expert.values()) == {0, 1}
        if not eligible:
            group_fits.append(GroupFit(
                photos=photos, annotators=annotators, expert_labels=g_expert,
                eligible=False))
            continue

        photo_local = {p: i for i, p in enumerate(photos)}
        ann_local = {a: i for i, a in enumerate(annotators)}
        local_votes = [Vote(photo_local[v.photo_idx],
                            ann_local[v.annotator_idx], v.label)
                       for v in g["votes"]]
        local_expert = {photo_local[p]: c for p, c in g_expert.items()}
        fit = _EMFit(local_votes, len(photos), len(annotators), local_expert)
        fit.run(MAX_ITER, TOL)
        group_fits.append(GroupFit(
            photos=photos, annotators=annotators, expert_labels=g_expert,
            eligible=True, fit=fit))
    return group_fits


def supplement_fits(group_fits, budget: ExtraBudget, relevant=None):
    """用共享补算额度继续迭代未收敛的组。

    ``relevant`` 为 None 时处理所有未收敛的可拟合组；否则只处理给定集合
    （按对象身份）中的组。采用轮转方式逐轮推进，保证额度被相关组公平、
    确定地分享；某组收敛后立即停止并让出剩余额度。
    """
    if relevant is None:
        pending = [gf for gf in group_fits
                   if gf.eligible and gf.fit is not None and not gf.converged]
    else:
        allowed = {id(gf) for gf in relevant}
        pending = [gf for gf in group_fits
                   if id(gf) in allowed and gf.eligible
                   and gf.fit is not None and not gf.converged]

    while budget.remaining > 0 and pending:
        for gf in list(pending):
            if budget.remaining <= 0:
                break
            if gf.converged:
                pending.remove(gf)
                continue
            gf.fit.step(TOL)
            gf.extra_iter += 1
            budget.spend(1)
            if gf.converged:
                pending.remove(gf)
    return budget


def converged_confusions(group_fits):
    """返回 {全局标注员下标: (2,2) 混淆矩阵}，只含已收敛且可拟合的组。"""
    fits = {}
    for gf in group_fits:
        conf = gf.confusion
        if conf is None:
            continue
        for local, annotator in enumerate(gf.annotators):
            fits[annotator] = conf[local]
    return fits


def annotator_group_map(group_fits):
    """返回 {全局标注员下标: (GroupFit, 组内局部下标)}。"""
    mapping = {}
    for gf in group_fits:
        for local, annotator in enumerate(gf.annotators):
            mapping[annotator] = (gf, local)
    return mapping


def fit_groups(votes, n_photos, n_annotators, expert_labels,
               extra_budget=None, relevant_annotators=None):
    """分组拟合，只返回已收敛组中标注员的混淆矩阵。

    ``extra_budget`` 为 None/0 时不补算；可传 ``ExtraBudget`` 或整数额度。
    传 ``relevant_annotators`` 时只给与这些标注员同组的未收敛组补算。
    """
    group_fits = build_group_fits(votes, n_photos, n_annotators, expert_labels)
    if extra_budget:
        budget = (extra_budget if isinstance(extra_budget, ExtraBudget)
                  else ExtraBudget(limit=int(extra_budget)))
        relevant = None
        if relevant_annotators is not None:
            relevant_ann = set(relevant_annotators)
            relevant = [gf for gf in group_fits
                        if relevant_ann.intersection(gf.annotators)]
        supplement_fits(group_fits, budget, relevant)
    return converged_confusions(group_fits)


def prepare_photo_fits(target_idx, votes, n_photos, n_annotators, expert_labels):
    """留一法准备一张待判照片的分组拟合（尚未补算）。

    返回 ``(target_votes, group_fits, relevant_fits, annotator_map)``。
    相关组 = 留一后包含该照片投票人的组，即“投过当前待判照片的组”。
    """
    remaining = [v for v in votes if v.photo_idx != target_idx]
    group_fits = build_group_fits(remaining, n_photos, n_annotators,
                                  expert_labels)
    target_votes = [v for v in votes if v.photo_idx == target_idx]
    target_annotators = {v.annotator_idx for v in target_votes}
    relevant = [gf for gf in group_fits
                if target_annotators.intersection(gf.annotators)]
    return target_votes, group_fits, relevant, annotator_group_map(group_fits)


def evaluate_photo(target_idx, votes, n_photos, n_annotators, expert_labels,
                   extra_budget=None):
    """留一法评估一张照片：先从拟合数据中移除该照片及其投票，再分组拟合。

    返回 (该照片的全部投票, 可用投票, fits)。可用投票 = 由已收敛组中的
    标注员投出的票；无票可用时调用方应标为“待人工复核”。``extra_budget``
    可选，用于给投过该照片的未收敛组补算。
    """
    target_votes, group_fits, relevant, _ = prepare_photo_fits(
        target_idx, votes, n_photos, n_annotators, expert_labels)
    if extra_budget:
        budget = (extra_budget if isinstance(extra_budget, ExtraBudget)
                  else ExtraBudget(limit=int(extra_budget)))
        supplement_fits(group_fits, budget, relevant)
    fits = converged_confusions(group_fits)
    usable = [v for v in target_votes if v.annotator_idx in fits]
    return target_votes, usable, fits


def pothole_probability(usable_votes, fits):
    """用可用投票计算有坑概率（类别先验各半，混淆矩阵来自留一拟合）。"""
    log0 = float(np.log(CLASS_PRIOR[0]))
    log1 = float(np.log(CLASS_PRIOR[1]))
    for v in usable_votes:
        conf = fits[v.annotator_idx]
        log0 += float(np.log(conf[0, v.label]))
        log1 += float(np.log(conf[1, v.label]))
    m = max(log0, log1)
    e0, e1 = np.exp(log0 - m), np.exp(log1 - m)
    return float(e1 / (e0 + e1))
