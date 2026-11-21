"""慢收敛票表的回归测试：补算额度、收敛后才采用、忽略计数与可重复性。"""
from fastapi.testclient import TestClient

from app import ds
from app.main import app

client = TestClient(app)

# ---- 两组已知的慢收敛数据（同构小规模 EM，单组内三条连通边） ----
# SLOW：单组完整数据需 253 轮（200 轮内不收敛，补 53 轮收敛）。
SLOW_VOTES = [
    (0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 0),
    (5, 0, 0), (6, 0, 0), (7, 0, 1),
    (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 0),
    (5, 1, 0), (6, 1, 0), (7, 1, 1),
    (0, 2, 0), (1, 2, 0), (2, 2, 0), (3, 2, 1), (4, 2, 1),
    (5, 2, 1), (6, 2, 0), (7, 2, 1),
]
SLOW_EXPERT = {0: 1, 1: 0}

# VERY_SLOW：单组完整数据需 1327 轮，200+200 轮补算后仍不收敛。
VERY_SLOW_VOTES = [
    (0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 0),
    (5, 0, 0), (6, 0, 1), (7, 0, 0),
    (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 0), (4, 1, 1),
    (5, 1, 1), (6, 1, 0), (7, 1, 1),
    (0, 2, 0), (1, 2, 0), (2, 2, 1), (3, 2, 0), (4, 2, 1),
    (5, 2, 1), (6, 2, 1), (7, 2, 1),
]
VERY_SLOW_EXPERT = {0: 1, 1: 0}


# ---- 纯算法层：首轮上限、补算、收敛门槛 ----

def test_group_needs_supplement_beyond_original_limit():
    votes = [ds.Vote(*v) for v in SLOW_VOTES]
    fits = ds.build_group_fits(votes, 8, 3, SLOW_EXPERT)
    assert len(fits) == 1
    gf = fits[0]
    assert gf.eligible
    assert not gf.converged          # 200 轮未达 1e-9
    assert gf.n_iter == ds.MAX_ITER
    assert gf.confusion is None      # 未收敛不采用
    assert ds.converged_confusions(fits) == {}

    budget = ds.ExtraBudget()
    ds.supplement_fits(fits, budget)
    assert gf.converged
    assert gf.n_iter == 253
    assert gf.extra_iter == 53
    assert budget.used == 53 and budget.remaining == ds.EXTRA_ITER - 53
    assert set(ds.converged_confusions(fits)) == {0, 1, 2}


def test_very_slow_group_stays_unconverged_after_full_supplement():
    votes = [ds.Vote(*v) for v in VERY_SLOW_VOTES]
    fits = ds.build_group_fits(votes, 8, 3, VERY_SLOW_EXPERT)
    gf = fits[0]
    assert not gf.converged
    budget = ds.ExtraBudget()
    ds.supplement_fits(fits, budget)
    assert budget.used == ds.EXTRA_ITER
    assert not gf.converged
    assert gf.n_iter == ds.MAX_ITER + ds.EXTRA_ITER
    assert gf.confusion is None
    assert ds.converged_confusions(fits) == {}


def test_supplement_only_touches_relevant_groups():
    # 两组互不相连；只给 A 组补算，B 组必须停留在原 200 轮。
    n_photos, n_ann = 17, 6
    votes = [ds.Vote(p, a, l) for p, a, l in SLOW_VOTES]
    votes += [ds.Vote(8 + p, 3 + a, l) for p, a, l in VERY_SLOW_VOTES]
    # 待判照片 p16 由 A 组标注员投过 → 只与 A 组相关。
    votes.append(ds.Vote(16, 0, 1))
    expert = dict(SLOW_EXPERT)
    expert.update({8 + p: c for p, c in VERY_SLOW_EXPERT.items()})
    target_votes, fits, relevant, _ = ds.prepare_photo_fits(
        16, votes, n_photos, n_ann, expert)
    assert len(target_votes) == 1
    assert len(fits) == 2 and len(relevant) == 1
    slow = next(gf for gf in fits if min(gf.annotators) < 3)
    other = next(gf for gf in fits if min(gf.annotators) >= 3)

    budget = ds.ExtraBudget()
    ds.supplement_fits(fits, budget, relevant)
    assert slow.converged and budget.used == 53
    assert not other.converged and other.n_iter == ds.MAX_ITER


def test_budget_shared_round_robin_deterministic():
    # 两个都要补算的组共用 200 轮：轮转推进，结果与额度分配确定可复现。
    budget = ds.ExtraBudget(limit=40)
    votes = [ds.Vote(p, a, l) for p, a, l in SLOW_VOTES]
    fits = ds.build_group_fits(votes, 8, 3, SLOW_EXPERT)
    first = fits[0]
    ds.supplement_fits(fits, budget)
    assert budget.used == 40
    assert first.extra_iter == 40 and first.n_iter == 240
    assert not first.converged


# ---- API 层 ----

def _slow_payload(photo_order=None, extra_votes=(), extra_photos=()):
    """慢收敛组 S0..S7（专家 S1/S2），可指定复核顺序与附加照片。"""
    photos = [f"S{i+1}" for i in range(8)] + list(extra_photos)
    if photo_order is not None:
        photos = photo_order
    votes = [{"photo": f"S{p+1}", "annotator": f"s{a+1}", "label": l}
             for p, a, l in SLOW_VOTES]
    votes += list(extra_votes)
    experts = [{"photo": f"S{p+1}", "label": c} for p, c in SLOW_EXPERT.items()]
    return {"photos": photos,
            "annotators": [f"s{i+1}" for i in range(3)],
            "votes": votes, "expert_labels": experts}


def test_review_scores_slow_group_within_shared_budget():
    # S4（下标 3）留一后只需 +130 轮；S3（下标 2）需 +399 轮。
    # 请求顺序先 S4 后 S3：S4 用掉 130 轮后评分，S3 只剩 70 轮 → 待复核。
    payload = _slow_payload(photo_order=[
        "S1", "S2", "S4", "S3", "S5", "S6", "S7", "S8"])
    body = client.post("/review", json=payload).json()
    results = {r["photo"]: r for r in body["results"]}
    assert results["S4"]["status"] == "scored"
    assert results["S4"]["pothole_probability"] is not None
    assert results["S4"]["votes_ignored"] == 0
    assert results["S3"]["status"] == "pending_manual_review"
    assert results["S3"]["pothole_probability"] is None
    assert results["S3"]["votes_ignored"] == 3
    assert "未收敛" in results["S3"]["note"]


def test_review_same_request_is_deterministic():
    payload = _slow_payload(photo_order=[
        "S1", "S2", "S4", "S3", "S5", "S6", "S7", "S8"])
    first = client.post("/review", json=payload).json()
    second = client.post("/review", json=payload).json()
    assert first == second


def test_review_uses_converged_group_and_ignores_unconverged_photo_votes():
    # 快组 A（K1..K6）+ 极慢组 B（B1..B8），待判照片 T 由两边各投一票。
    # B 组在额度内不可能收敛 → T 只用 A 的票评分，B 的票计入忽略。
    photos = [f"K{i+1}" for i in range(6)] + [f"B{i+1}" for i in range(8)] + ["T"]
    votes = []
    for a in range(3):
        for i, t in enumerate([1, 0, 1, 0, 1, 0]):
            votes.append({"photo": f"K{i+1}", "annotator": f"k{a+1}",
                          "label": 1 - t if (a == 2 and i == 0) else t})
    for p, a, l in VERY_SLOW_VOTES:
        votes.append({"photo": f"B{p+1}", "annotator": f"b{a+1}", "label": l})
    votes.append({"photo": "T", "annotator": "k1", "label": 1})
    votes.append({"photo": "T", "annotator": "b1", "label": 1})
    experts = [{"photo": "K1", "label": 1}, {"photo": "K2", "label": 0},
               {"photo": "B1", "label": 1}, {"photo": "B2", "label": 0}]
    payload = {"photos": photos,
               "annotators": [f"k{i+1}" for i in range(3)]
                            + [f"b{i+1}" for i in range(3)],
               "votes": votes, "expert_labels": experts}
    body = client.post("/review", json=payload).json()
    assert body["results"][0]["photo"] == "K3"  # 快组照片先复核
    t_result = next(r for r in body["results"] if r["photo"] == "T")
    assert t_result["status"] == "scored"
    assert [v["annotator"] for v in t_result["votes_used"]] == ["k1"]
    assert t_result["votes_ignored"] == 1


def test_annotator_stats_marks_unconverged_group_not_fitted():
    payload = _slow_payload()
    body = client.post("/annotator-stats", json=payload).json()
    stats = {s["annotator"]: s for s in body["annotators"]}
    # 完整慢组补 53 轮收敛 → 均 fitted。
    assert {s["annotator"] for s in body["annotators"]} == {"s1", "s2", "s3"}
    assert all(s["status"] == "fitted" for s in body["annotators"])
    assert stats["s1"]["false_positive_rate"] is not None

    # 极慢组 200+200 轮仍未收敛 → 全部 not_fitted 且说明原因。
    payload = _slow_payload()
    payload["votes"] = [
        {"photo": f"S{p+1}", "annotator": f"s{a+1}", "label": l}
        for p, a, l in VERY_SLOW_VOTES]
    body = client.post("/annotator-stats", json=payload).json()
    for row in body["annotators"]:
        assert row["status"] == "not_fitted"
        assert row["false_positive_rate"] is None
        assert "未收敛" in row["note"] and "400" in row["note"]


def test_annotator_stats_deterministic():
    payload = _slow_payload()
    assert client.post("/annotator-stats", json=payload).json() == \
        client.post("/annotator-stats", json=payload).json()
