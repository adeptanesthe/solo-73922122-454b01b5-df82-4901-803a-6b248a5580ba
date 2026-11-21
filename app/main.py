"""坑槽照片标注复核 API。

质检员汇总多人标注后调用：
- POST /review           留一法逐张复核未确认照片，给出有坑概率或“待人工复核”。
- POST /annotator-stats  在完整数据上分组拟合，汇总每人的误标率/漏标率。
- GET  /health           健康检查。

慢收敛处理：每组先按原上限独立拟合至多 200 轮；一个请求另有 200 轮补算
额度，供相关未收敛组共用。只有达到原收敛精度（混淆矩阵相邻两轮最大变化
< 1e-9）的组才被采用。复核只给“投过当前待判照片”的组补算，逐张按留一后
各组的收敛状态选票；仍未收敛组的票计入忽略数。/annotator-stats 对仍未
收敛的组标记 not_fitted 并说明原因。
"""
from __future__ import annotations

from fastapi import FastAPI

from . import ds
from .schemas import (AnnotatorStat, AnnotatorStatsResponse, ConfirmedPhoto,
                      PhotoResult, ReviewRequest, ReviewResponse, VoteUsed)

app = FastAPI(
    title="坑槽照片标注复核 API",
    version="1.0.0",
    description=(
        "Dawid–Skene 复核：类别先验各半、混淆矩阵加一平滑、专家标签固定、缺票不计。"
        "评估每张未确认照片时先移除该照片及其投票，按剩余投票把相连的照片和标注员分组，"
        "仅拟合同时含正、负专家标签的组，并只用这些组中标注员的投票计算有坑概率。"
        "每组先按原上限至多 200 轮拟合；每个请求另有 200 轮补算额度供相关未收敛组"
        "共用，达到原收敛精度后才采用该组。"
    ),
)


def _prepare(req: ReviewRequest):
    """把字符串编号映射为下标，构造投票与专家标签。"""
    photo_idx = {p: i for i, p in enumerate(req.photos)}
    ann_idx = {a: i for i, a in enumerate(req.annotators)}
    votes = [ds.Vote(photo_idx[v.photo], ann_idx[v.annotator], v.label)
             for v in req.votes]
    expert = {photo_idx[e.photo]: e.label for e in req.expert_labels}
    return photo_idx, ann_idx, votes, expert


def _pending_note(target_votes, group_map, relevant):
    """解释为何该照片无票可用（响应 note 字段）。"""
    if not target_votes:
        return "无票可用：该照片没有任何投票"
    if not any(v.annotator_idx in group_map for v in target_votes):
        return "无票可用：投票的标注员均不在已拟合的组中"
    reasons: list[str] = []
    if any(not gf.eligible for gf in relevant):
        reasons.append("投票的标注员所在组缺少正或负专家标签")
    if any(gf.eligible and not gf.converged for gf in relevant):
        reasons.append("投票的标注员所在组在原 200 轮与共享补算额度内均未收敛，"
                       "相关票已计入忽略数")
    if not reasons:  # 理论不可达：有组且有票时可用票不会为空
        reasons.append("投票的标注员所在组未产出可用混淆矩阵")
    return "无票可用：" + "；".join(reasons)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    photo_idx, ann_idx, votes, expert = _prepare(req)
    idx_photo = {i: p for p, i in photo_idx.items()}
    idx_ann = {i: a for a, i in ann_idx.items()}
    n_photos, n_ann = len(req.photos), len(req.annotators)

    # 每个请求共享一份补算额度；相关未收敛组在额度内轮转续算。
    budget = ds.ExtraBudget()

    results: list[PhotoResult] = []
    for photo in req.photos:
        i = photo_idx[photo]
        if i in expert:
            continue  # 专家已确认的照片无需复核
        target_votes, group_fits, relevant, group_map = ds.prepare_photo_fits(
            i, votes, n_photos, n_ann, expert)
        # 只给投过当前待判照片的组补算，共用本请求的剩余额度。
        ds.supplement_fits(group_fits, budget, relevant)
        fits = ds.converged_confusions(group_fits)
        usable = [v for v in target_votes if v.annotator_idx in fits]
        if not usable:
            results.append(PhotoResult(
                photo=photo, status="pending_manual_review",
                votes_ignored=len(target_votes),
                note=_pending_note(target_votes, group_map, relevant)))
            continue
        prob = ds.pothole_probability(usable, fits)
        used = [VoteUsed(
            annotator=idx_ann[v.annotator_idx],
            label=v.label,
            false_positive_rate=round(float(fits[v.annotator_idx][0, 1]), 6),
            false_negative_rate=round(float(fits[v.annotator_idx][1, 0]), 6),
        ) for v in usable]
        results.append(PhotoResult(
            photo=photo, status="scored",
            pothole_probability=round(prob, 6),
            votes_used=used,
            votes_ignored=len(target_votes) - len(usable)))

    return ReviewResponse(
        n_photos=n_photos,
        n_annotators=n_ann,
        confirmed=[ConfirmedPhoto(photo=idx_photo[i], expert_label=c)
                   for i, c in sorted(expert.items())],
        results=results,
    )


@app.post("/annotator-stats", response_model=AnnotatorStatsResponse)
def annotator_stats(req: ReviewRequest):
    """在完整数据上分组拟合，汇总每人的误标率/漏标率（供质检员汇总）。"""
    _, _, votes, expert = _prepare(req)
    group_fits = ds.build_group_fits(
        votes, len(req.photos), len(req.annotators), expert)
    # 本接口同样享有每请求 200 轮补算额度，由所有未收敛组共用。
    ds.supplement_fits(group_fits, ds.ExtraBudget())
    group_map = ds.annotator_group_map(group_fits)
    n_votes = [0] * len(req.annotators)
    for v in votes:
        n_votes[v.annotator_idx] += 1

    rows: list[AnnotatorStat] = []
    for a, name in enumerate(req.annotators):
        gf, local = group_map.get(a, (None, None))
        if gf is None:
            rows.append(AnnotatorStat(
                annotator=name, status="not_fitted", n_votes=n_votes[a],
                note="无任何投票（不在任何连通组中），未参与拟合"))
        elif not gf.eligible:
            rows.append(AnnotatorStat(
                annotator=name, status="not_fitted", n_votes=n_votes[a],
                note="所在连通组缺少正或负专家标签，未参与拟合"))
        elif not gf.converged:
            rows.append(AnnotatorStat(
                annotator=name, status="not_fitted", n_votes=n_votes[a],
                note=("所在连通组未收敛：原 200 轮及共享 200 轮补算额度内"
                      f"均未达到收敛精度（已跑 {gf.n_iter} 轮，"
                      f"最近两轮最大变化 {gf.fit.delta:.3e} ≥ 1e-9）")))
        else:
            conf = gf.confusion[local]
            rows.append(AnnotatorStat(
                annotator=name, status="fitted", n_votes=n_votes[a],
                false_positive_rate=round(float(conf[0, 1]), 6),
                false_negative_rate=round(float(conf[1, 0]), 6)))
    return AnnotatorStatsResponse(annotators=rows)
