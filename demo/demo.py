#!/usr/bin/env python3
"""演示请求：读取同目录 payload.json，依次调用 /review 与 /annotator-stats。

演示数据包含三张慢收敛票表：
- A 组：原 200 轮内收敛，正常评分；
- B 组：原 200 轮未收敛，靠本请求共享补算额度（共 200 轮）收敛后评分；
- C 组：原 200 轮 + 全部补算额度仍未收敛，其票计入忽略数，
  照片转人工复核，标注员在汇总里标为未拟合。

用法：
    python3 demo/demo.py [--base-url http://127.0.0.1:8000]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path


def post(base_url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    payload = json.loads(
        Path(__file__).with_name("payload.json").read_text(encoding="utf-8"))

    review = post(args.base_url, "/review", payload)
    print("== 复核结果（/review）==")
    confirmed = {c["photo"]: c["expert_label"] for c in review["confirmed"]}
    print("专家已确认：" + ", ".join(
        f"{p}={'有坑' if l else '无坑'}" for p, l in confirmed.items()))
    for r in review["results"]:
        if r["status"] == "scored":
            used = ", ".join(
                f"{v['annotator']} 投{'有坑' if v['label'] else '无坑'}"
                f"(误标 {v['false_positive_rate']:.3f}/漏标 {v['false_negative_rate']:.3f})"
                for v in r["votes_used"])
            ignored = f"，忽略 {r['votes_ignored']} 票" if r["votes_ignored"] else ""
            print(f"{r['photo']}: 有坑概率 {r['pothole_probability']:.3f}"
                  f"  [{used}]{ignored}")
        else:
            print(f"{r['photo']}: 待人工复核（忽略 {r['votes_ignored']} 票；"
                  f"{r['note']}）")

    # 同一请求再调一次，结果应完全一致（确定性）。
    again = post(args.base_url, "/review", payload)
    print("\n重复调用 /review 结果一致：" +
          ("是" if again == review else "否"))

    stats = post(args.base_url, "/annotator-stats", payload)
    print("\n== 标注员汇总（/annotator-stats）==")
    for s in stats["annotators"]:
        if s["status"] == "fitted":
            print(f"{s['annotator']}: 误标率 {s['false_positive_rate']:.3f}，"
                  f"漏标率 {s['false_negative_rate']:.3f}（{s['n_votes']} 票）")
        else:
            print(f"{s['annotator']}: 未拟合（{s['note']}；{s['n_votes']} 票）")

    stats_again = post(args.base_url, "/annotator-stats", payload)
    print("\n重复调用 /annotator-stats 结果一致：" +
          ("是" if stats_again == stats else "否"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
