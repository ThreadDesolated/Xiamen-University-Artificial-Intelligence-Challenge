"""跨任务配方迁移：同任务类型、画像相似的历史方案优先复用。

与"同任务命中"的区别：`db.find_recipe` 要求画像指纹完全一致（同一任务、同一份画像），
这里放宽到结构相似（特征类型构成、特征数、缺失与失衡程度、数据规模），
并返回相似度与来源任务，便于在 trace / 面板 / 报告里作为迁移证据展示。
相似度门槛见 `profile.TRANSFER_MIN_SIMILARITY`；低于门槛宁可不迁移，也不乱套历史方案。
"""
from __future__ import annotations

from .profile import TRANSFER_MIN_SIMILARITY, profile_similarity
from .store import db


def find_transfer_recipe(
    task_type: str,
    profile: dict,
    *,
    exclude_profile_hash: str = "",
    threshold: float = TRANSFER_MIN_SIMILARITY,
    db_path=None,
) -> dict | None:
    """在配方库里找与当前画像最相似的一条历史配方（找不到返回 None）。

    - 只比较同 `task_type` 的配方（回归方案不能套到分类任务）；
    - 跳过 `exclude_profile_hash`（同任务精确命中的情况走 `db.find_recipe`）；
    - 没有画像的旧配方不参与迁移（无法判断是否适用）；
    - 相似度同分时保留配方库排序靠前（指标更高）的一条。
    """
    best: dict | None = None
    for recipe in db.list_recipes(task_type=task_type, db_path=db_path):
        if exclude_profile_hash and recipe["profile_hash"] == exclude_profile_hash:
            continue
        other = recipe.get("profile") or {}
        if not other:
            continue
        similarity = profile_similarity(profile, other)
        if similarity < threshold:
            continue
        if best is None or similarity > best["similarity"]:
            best = {**recipe, "similarity": similarity}
    return best
