"""
Построение признаков по событиям внутри суточного окна наблюдения.

Все признаки считаются только по событиям из окна [window_start_ts, window_end_ts).
В events.csv есть события после конца окна, брать их нельзя: это утечка из будущего.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Коды событий из описания задачи.
EVENT_CODES = {
    100: "search_results_view",
    200: "item_view",
    210: "photo_swipe",
    220: "seller_page_view",
    300: "contact_phone_show",
    301: "contact_chat_open",
    303: "contact_message_sent",
    400: "favorite_add",
    500: "login",
    900: "captcha_shown",
}
# Контактные действия (за ними и приходят парсеры) и действия с вовлечённостью.
CONTACT_CODES = [300, 301, 303]
ENGAGEMENT_CODES = [210, 400, 500]


def normalize_platform(s: pd.Series) -> pd.Series:
    """platform записана в разном регистре (WEB/Web/web) и с синонимами (iphone/IOS/ios)."""
    p = s.astype(str).str.strip().str.lower()
    return p.replace({"iphone": "ios"})


def clip_to_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только события внутри окна наблюдения своей куки."""
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner"
    )
    in_window = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    return ev.loc[in_window].copy()


def _entropy(counts: np.ndarray) -> float:
    """Нормированная энтропия Шеннона: 0 если всё в одной категории, 1 если равномерно."""
    total = counts.sum()
    if total == 0 or len(counts) <= 1:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def _group_entropy(df: pd.DataFrame, col: str) -> pd.Series:
    """Энтропия распределения куки по значениям col: насколько разбросан обход."""
    counts = df.groupby(["cookie_id", col], observed=True).size()
    return counts.groupby(level=0).agg(lambda s: _entropy(s.to_numpy()))


def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    events: сырые события (весь events.csv)
    meta:   train.csv или test.csv со столбцами cookie_id, cookie_created_at, window_*
    Возврат: одна строка на cookie_id из meta, порядок строк совпадает с meta.
    """
    ev = clip_to_window(events, meta)
    ev["platform_norm"] = normalize_platform(ev.platform)
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort")

    g = ev.groupby("cookie_id", sort=True)
    f = pd.DataFrame(index=g.size().index)

    # A. Объём и интенсивность
    f["n_events"] = g.size()
    span = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds()
    f["active_span_s"] = span
    # +1 в знаменателе, чтобы куки с одним событием не давали inf.
    f["events_per_min"] = f.n_events / (span / 60.0 + 1.0)
    f["n_active_hours"] = g.event_ts.agg(lambda s: s.dt.floor("h").nunique())
    f["events_per_active_hour"] = f.n_events / f.n_active_hours

    # B. Тайминг. Человек ходит рывками, бот чаще работает с ровным шагом.
    ev["dt"] = g.event_ts.diff().dt.total_seconds()
    dg = ev.dropna(subset=["dt"]).groupby("cookie_id")
    f["dt_mean"] = dg.dt.mean()
    f["dt_median"] = dg.dt.median()
    f["dt_std"] = dg.dt.std()
    f["dt_min"] = dg.dt.min()
    f["dt_max"] = dg.dt.max()
    f["dt_p10"] = dg.dt.quantile(0.10)
    f["dt_p90"] = dg.dt.quantile(0.90)
    # Доля длинных пауз: живой пользователь отвлекается, автомат идёт ровно.
    f["dt_long_share"] = dg.dt.agg(lambda s: (s > 300).mean())
    # Коэффициент вариации: низкий = подозрительно равномерные интервалы.
    f["dt_cv"] = f.dt_std / f.dt_mean.replace(0, np.nan)
    f["dt_fast_share"] = dg.dt.agg(lambda s: (s < 2.0).mean())
    f["dt_veryfast_share"] = dg.dt.agg(lambda s: (s < 0.5).mean())
    # Доля интервалов, совпадающих с самым частым (с точностью до секунды).
    f["dt_mode_share"] = dg.dt.agg(
        lambda s: s.round().value_counts(normalize=True).iloc[0] if len(s) else np.nan
    )
    # Интервалы отдельно между просмотрами объявлений: у разных типов событий свой
    # темп, и смесь типов размывает сигнал.
    iv = ev[ev.eid == 200].copy()
    iv["dt_item"] = iv.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    ig = iv.dropna(subset=["dt_item"]).groupby("cookie_id")
    f["dt_item_median"] = ig.dt_item.median()
    f["dt_item_min"] = ig.dt_item.min()

    # C. Типы событий
    cnt = ev.pivot_table(index="cookie_id", columns="eid", values="event_ts", aggfunc="size").fillna(0)
    cnt = cnt.reindex(columns=list(EVENT_CODES), fill_value=0)
    # captcha_shown (900) пропущен: в данных это событие встречается только после
    # конца окна, поэтому внутри окна признак всегда нулевой. Проверка в solution.ipynb.
    for code, name in EVENT_CODES.items():
        if code == 900:
            continue
        f[f"n_{name}"] = cnt[code]
        f[f"sh_{name}"] = cnt[code] / f.n_events
    f["sh_contact"] = cnt[CONTACT_CODES].sum(axis=1) / f.n_events
    f["sh_engagement"] = cnt[ENGAGEMENT_CODES].sum(axis=1) / f.n_events
    # Просмотров объявлений на один заход в выдачу.
    f["items_per_search"] = cnt[200] / (cnt[100] + 1.0)
    f["contacts_per_item"] = cnt[CONTACT_CODES].sum(axis=1) / (cnt[200] + 1.0)

    # Доля подряд идущих однотипных действий: след обхода списком.
    ev["prev_eid"] = g.eid.shift()
    same_kind = (ev.eid == ev.prev_eid)
    f["sh_same_kind_transition"] = same_kind.groupby(ev.cookie_id).mean()
    f["sh_item_to_item"] = ((ev.eid == 200) & (ev.prev_eid == 200)).groupby(ev.cookie_id).mean()

    # D. Разнообразие контента
    f["item_nunique"] = g.item_id.nunique()
    f["cat_nunique"] = g.item_category.nunique()
    f["loc_nunique"] = g.item_location.nunique()
    f["item_repeat_rate"] = cnt[200] / (f.item_nunique + 1.0)
    f["cat_entropy"] = _group_entropy(ev, "item_category")
    f["loc_entropy"] = _group_entropy(ev, "item_location")
    f["sh_seller_pro"] = g.seller_type.agg(lambda s: (s == "pro").mean() if s.notna().any() else np.nan)
    # Концентрация обхода: доля самой частой категории и локации.
    f["top_cat_share"] = ev.groupby(["cookie_id", "item_category"]).size().groupby(level=0).max() / f.n_events
    f["top_loc_share"] = ev.groupby(["cookie_id", "item_location"]).size().groupby(level=0).max() / f.n_events
    # Листание фото на один просмотр: автомату листать незачем.
    f["photo_per_item"] = cnt[210] / (cnt[200] + 1.0)

    # Поиск: глубина пагинации и повторяемость запроса.
    srch = ev.dropna(subset=["search_page"]).groupby("cookie_id")
    f["search_page_max"] = srch.search_page.max()
    f["search_page_mean"] = srch.search_page.mean()
    f["query_nunique"] = srch.search_query.nunique()
    f["pages_per_query"] = f.search_page_max / (f.query_nunique + 1.0)
    f["query_repeat_rate"] = srch.search_query.size() / (f.query_nunique + 1.0)

    # E. User-Agent и платформа
    ua = ev.user_agent.fillna("")
    ev["ua_headless"] = ua.str.contains("Headless", case=False, regex=False)
    ev["ua_bot_token"] = ua.str.contains("bot|crawl|spider|python|curl|wget|java|scrapy", case=False, regex=True)
    f["ua_headless"] = ev.groupby("cookie_id").ua_headless.max().astype(int)
    f["ua_bot_token"] = ev.groupby("cookie_id").ua_bot_token.max().astype(int)
    f["ua_nunique"] = g.user_agent.nunique()
    f["platform_nunique"] = ev.groupby("cookie_id").platform_norm.nunique()
    # Доля мобильного трафика. Флаг "смешанная платформа" не строю: смешения
    # mobile/desktop внутри куки в данных нет, platform_nunique=2 это всегда
    # web/desktop либо android/ios, то есть разная запись одной платформы.
    is_mobile = ev.platform_norm.isin(["android", "ios"])
    f["sh_mobile"] = is_mobile.groupby(ev.cookie_id).mean()

    # F. Курсор. На android/ios pointer_* всегда пуст, поэтому считаю только по
    # web/desktop: иначе признак вырождается в индикатор мобильного устройства.
    web = ev[~is_mobile]
    if len(web):
        wg = web.groupby("cookie_id")
        f["pointer_coverage"] = wg.pointer_x.agg(lambda s: s.notna().mean())
        f["pointer_x_std"] = wg.pointer_x.std()
        f["pointer_y_std"] = wg.pointer_y.std()
        pos = web.dropna(subset=["pointer_x"]).copy()
        pos["pos"] = pos.pointer_x.round().astype(int).astype(str) + "_" + pos.pointer_y.round().astype(int).astype(str)
        pg = pos.groupby("cookie_id")
        # Низкая доля уникальных позиций означает, что курсор стоит на месте.
        f["pointer_pos_uniq_rate"] = pg.pos.nunique() / pg.size()

    # G. Суточный профиль
    ev["_hour"] = ev.event_ts.dt.hour
    f["hour_entropy"] = _group_entropy(ev, "_hour")
    f["sh_night"] = ev._hour.between(1, 5).groupby(ev.cookie_id).mean()

    # H. Дубликаты. Повтор одного события характерен для автоматического обхода,
    # поэтому дубли не удаляю, а считаю их долю.
    dup_full = ev.duplicated(subset=["cookie_id", "event_ts", "eid", "item_id", "search_query"], keep="first")
    f["dup_event_share"] = dup_full.groupby(ev.cookie_id).mean()
    same_ts = ev.duplicated(subset=["cookie_id", "event_ts"], keep="first")
    f["same_ts_share"] = same_ts.groupby(ev.cookie_id).mean()

    # I. Метаданные куки. Флаг "создана внутри окна" не строю: таких кук в данных нет.
    # День недели тоже не беру, сигнала в нём нет, а тест лежит в другой неделе.
    out = meta[["cookie_id"]].merge(f.reset_index(), on="cookie_id", how="left")
    out["cookie_age_days"] = (
        (meta.window_end_ts - meta.cookie_created_at).dt.total_seconds() / 86400.0
    ).to_numpy()

    # Счётчики зануляю, статистики по интервалам нет: NaN там значит "событие было
    # одно", это само по себе информация, а LightGBM работает с NaN.
    count_cols = [c for c in out.columns if c.startswith(("n_", "sh_", "has_", "ua_", "dup_", "same_ts"))]
    out[count_cols] = out[count_cols].fillna(0)
    return out
