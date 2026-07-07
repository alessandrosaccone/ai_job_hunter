from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from models.job import MatchResult, ScanResult
from models.user_profile import ExperienceLevelRule, FundamentalCriteria, UserProfile
from orchestrator import run_scan_sync
from storage.memory import JobMemory
from storage.profile_registry import (
    ProfilePaths,
    create_profile,
    delete_profile,
    list_profiles,
    resolve_active_profile,
    set_last_active,
)
from agents.job_listing_expander import match_salary_sort_key
from agents.search_providers.router import JobSearchRouter
from storage.search_quota import exhausted_providers
from storage.saved_jobs import SavedJobsStore
from storage.scan_history import ROME, ScanHistoryStore, format_italian_date

load_dotenv()

EXAMPLE_PROFILE_PATH = Path("config/user_profile.example.json")
CAREER_FIELDS_PATH = Path("config/career_fields.json")

WORK_MODE_OPTIONS = ["Remote", "Hybrid", "Full-time in office"]
EXPERIENCE_LEVEL_OPTIONS = [
    ("graduate", "Neolaureato / Graduate"),
    ("internship", "Stage / Internship"),
    ("entry", "Entry level"),
    ("mid", "Mid level"),
    ("senior", "Senior"),
    ("manager", "Manager / Lead"),
]
EXPERIENCE_LEVEL_RULE_OPTIONS = [
    ("exact", "Questo livello"),
    ("all_lower", "Questo livello o inferiori"),
    ("or_lower", "Questo livello o X inferiori"),
    ("or_higher", "Questo livello o X superiori"),
]
MATCH_SORT_OPTIONS = [
    ("match_desc", "Punteggio match (migliori prima)"),
    ("match_asc", "Punteggio match (peggiori prima)"),
    ("salary_desc", "RAL (più alta prima, senza RAL in fondo)"),
    ("company", "Azienda (A→Z)"),
    ("title", "Ruolo (A→Z)"),
]
HISTORY_LAYOUT_OPTIONS = [
    ("by_day", "Raggruppa per giorno"),
    ("match_desc", "Punteggio match (migliori prima)"),
    ("match_asc", "Punteggio match (peggiori prima)"),
    ("salary_desc", "RAL (più alta prima, senza RAL in fondo)"),
]
SAVED_SORT_OPTIONS = [
    ("saved_desc", "Data salvataggio (recenti)"),
    ("match_desc", "Punteggio match (migliori prima)"),
    ("match_asc", "Punteggio match (peggiori prima)"),
    ("salary_desc", "RAL (più alta prima, senza RAL in fondo)"),
    ("saved_asc", "Data salvataggio (vecchi)"),
]
APPLICATION_CHANNEL_LABELS = {
    "human_recruiter": "Recruiter umano",
    "ats": "ATS / portale",
    "mixed": "Misto (ATS + umano)",
    "unknown": "Non chiaro",
}
CV_TYPE_HINTS = {
    "human_recruiter": "CV human-friendly: chiaro, leggibile, focus su risultati e percorso.",
    "ats": "CV testuale: keyword dell'annuncio, formato semplice e parsabile dagli ATS.",
    "mixed": "Prepara entrambe le versioni o un ibrido ATS + human-friendly.",
}


def _load_career_fields() -> dict:
    if not CAREER_FIELDS_PATH.exists():
        return {}
    with CAREER_FIELDS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


CAREER_FIELDS = _load_career_fields()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle { color: #6b7280; margin-bottom: 1.5rem; }
        .job-card {
            border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem;
            margin-bottom: 1rem; background: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        }
        .score-badge {
            display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
            font-weight: 600; color: white; background: #16a34a;
        }
        section[data-testid="stSidebar"] button[aria-label="Elimina profilo e tutti i suoi dati"] {
            border: 1px solid #fecaca !important;
            background: linear-gradient(180deg, #fff5f5 0%, #fee2e2 100%) !important;
            color: #dc2626 !important;
            border-radius: 10px !important;
            min-height: 2.35rem !important;
            padding: 0.2rem 0.55rem !important;
            box-shadow: 0 1px 2px rgba(220, 38, 38, 0.08);
            transition: all 0.15s ease;
        }
        section[data-testid="stSidebar"] button[aria-label="Elimina profilo e tutti i suoi dati"]:hover:not(:disabled) {
            border-color: #f87171 !important;
            background: linear-gradient(180deg, #fee2e2 0%, #fecaca 100%) !important;
            color: #b91c1c !important;
            box-shadow: 0 2px 6px rgba(220, 38, 38, 0.18);
            transform: translateY(-1px);
        }
        section[data-testid="stSidebar"] button[aria-label="Elimina profilo e tutti i suoi dati"]:disabled {
            opacity: 0.35;
            filter: grayscale(0.4);
        }
        section[data-testid="stSidebar"] button[aria-label="Elimina profilo e tutti i suoi dati"] p {
            font-size: 1.05rem !important;
            line-height: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _api_key_status(label: str, env_name: str) -> None:
    value = os.getenv(env_name, "")
    configured = bool(value) and "your_" not in value
    color = "#16a34a" if configured else "#dc2626"
    status = "Configured" if configured else "Missing"
    st.markdown(
        f"**{label}:** <span style='color:{color};'>{status}</span>",
        unsafe_allow_html=True,
    )


def _saved_store(paths: ProfilePaths) -> SavedJobsStore:
    return SavedJobsStore(paths.saved_jobs_path, JobMemory(paths.memory_path))


@st.fragment
def _render_profile_selector() -> ProfilePaths:
    st.header("Profili")
    profiles = list_profiles()
    active = resolve_active_profile(st.session_state.get("active_profile_slug"))
    can_delete = len(profiles) > 1

    pending_slug = st.session_state.get("confirm_delete_slug")
    if pending_slug:
        pending = next((profile for profile in profiles if profile.slug == pending_slug), None)
        if pending:
            st.warning(
                f"Eliminare **{pending.display_name}**? "
                "Verranno rimossi profilo, memoria, candidature salvate e scansioni."
            )
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Conferma", type="primary", use_container_width=True, key="confirm-delete"):
                try:
                    fallback = delete_profile(pending_slug)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("confirm_delete_slug", None)
                    if st.session_state.get("active_profile_slug") == pending_slug:
                        st.session_state["active_profile_slug"] = fallback.slug if fallback else None
                    st.session_state.pop("last_scan_result", None)
                    st.session_state.pop("live_matches", None)
                    st.toast(f"Profilo «{pending.display_name}» eliminato.")
                    st.rerun()
            if cancel_col.button("Annulla", use_container_width=True, key="cancel-delete"):
                st.session_state.pop("confirm_delete_slug", None)
                st.rerun()
        else:
            st.session_state.pop("confirm_delete_slug", None)

    st.caption("Ogni profilo ha memoria, salvati e scansioni separati.")
    for profile in profiles:
        is_active = profile.slug == active.slug
        name_col, delete_col = st.columns([5, 1])
        with name_col:
            if st.button(
                f"{'● ' if is_active else '○ '}{profile.display_name}",
                key=f"select-profile-{profile.slug}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state["active_profile_slug"] = profile.slug
                    set_last_active(profile.slug)
                    st.session_state.pop("last_scan_result", None)
                    st.session_state.pop("live_matches", None)
                    st.rerun()
        with delete_col:
            if st.button(
                "",
                key=f"delete-profile-{profile.slug}",
                icon=":material/delete_outline:",
                type="secondary",
                disabled=not can_delete,
                help="Elimina profilo e tutti i suoi dati",
                use_container_width=True,
            ):
                st.session_state["confirm_delete_slug"] = profile.slug
                st.rerun()

    new_name = st.text_input("Nuovo profilo", placeholder="es. Giulia, Sara...")
    if st.button("Crea profilo", use_container_width=True):
        if new_name.strip():
            paths = create_profile(new_name.strip())
            st.session_state["active_profile_slug"] = paths.slug
            st.toast(f"Profilo «{paths.display_name}» creato.")
            st.rerun()
        else:
            st.warning("Inserisci un nome per il nuovo profilo.")

    return resolve_active_profile(st.session_state.get("active_profile_slug"))


def _load_profile_for_form(paths: ProfilePaths) -> UserProfile:
    profile = UserProfile.load(paths.profile_path)
    if profile:
        return profile

    example = UserProfile.load(EXAMPLE_PROFILE_PATH)
    if example:
        return example

    return UserProfile(
        career_field="tech",
        experience_level="mid",
        education="",
        passions=[],
        target_roles=["software engineer"],
        desired_salary_eur=None,
        location="Milan, Italy",
        work_mode="Remote",
        free_text_preferences="",
        fundamental_criteria=FundamentalCriteria(location=True),
    )


_PROVIDER_LABELS = {
    "serpapi": "SerpApi",
    "serper": "Serper",
    "dataforseo": "DataForSEO",
    "apify": "Apify",
    "scraperapi": "ScraperAPI",
    "duckduckgo": "DuckDuckGo",
    "deepseek": "DeepSeek web",
}


def _format_provider_stats_lines(stats: dict) -> list[str]:
    if not stats:
        return ["Nessun provider di ricerca web chiamato in questa fase."]
    lines: list[str] = []
    for name, data in stats.items():
        label = _PROVIDER_LABELS.get(name, name)
        lines.append(
            f"{label}: {data.get('ok', 0)} OK · "
            f"{data.get('empty', 0)} vuoti · "
            f"{data.get('fail', 0)} errori · "
            f"{data.get('results', 0)} risultati"
        )
    return lines


def _render_provider_stats(stats: dict) -> None:
    if not stats:
        return
    with st.expander("Provider di ricerca usati", expanded=False):
        for line in _format_provider_stats_lines(stats):
            st.markdown(f"- {line}")
        st.caption("Include ricerca annunci (Startup Discoverer) e ricerca RAL.")


def _score_color(score: float) -> str:
    if score >= 8:
        return "#16a34a"
    if score >= 7:
        return "#ca8a04"
    return "#dc2626"


def _sort_matches(matches: list[MatchResult], mode: str) -> list[MatchResult]:
    if mode == "match_desc":
        return sorted(matches, key=lambda item: item.match_score, reverse=True)
    if mode == "match_asc":
        return sorted(matches, key=lambda item: item.match_score)
    if mode == "salary_desc":
        return sorted(matches, key=match_salary_sort_key)
    if mode == "company":
        return sorted(matches, key=lambda item: item.job.company.lower())
    if mode == "title":
        return sorted(matches, key=lambda item: item.job.title.lower())
    return list(matches)


def _render_sort_selectbox(
    options: list[tuple[str, str]],
    widget_key: str,
    *,
    label: str = "Ordina per",
) -> str:
    keys = [key for key, _ in options]
    labels = [label_text for _, label_text in options]
    selected_label = st.selectbox(label, options=labels, key=widget_key)
    return keys[labels.index(selected_label)]


def _career_field_labels() -> list[tuple[str, str]]:
    return [(key, value.get("label", key)) for key, value in CAREER_FIELDS.items()]


def render_profile_tab(paths: ProfilePaths) -> None:
    current = _load_profile_for_form(paths)
    field_labels = _career_field_labels()
    field_keys = [key for key, _ in field_labels]
    field_display = [label for _, label in field_labels]

    with st.form("profile_form"):
        st.subheader("1. Campo professionale")
        st.caption("Scegli il settore prima di tutto: adatta aziende, passioni e ricerche web.")

        current_field_index = field_keys.index(current.career_field) if current.career_field in field_keys else 0
        selected_field_label = st.selectbox(
            "Campo",
            options=field_display,
            index=current_field_index,
        )
        selected_field = field_keys[field_display.index(selected_field_label)]
        field_config = CAREER_FIELDS.get(selected_field, {})
        passion_options = field_config.get("passions", [])
        role_hints = field_config.get("role_hints", [])

        if role_hints:
            st.info(f"Ruoli suggeriti per questo campo: {', '.join(role_hints)}")

        st.subheader("2. Profilo dettagliato")

        level_keys = [key for key, _ in EXPERIENCE_LEVEL_OPTIONS]
        level_labels = [label for _, label in EXPERIENCE_LEVEL_OPTIONS]
        level_index = level_keys.index(current.experience_level) if current.experience_level in level_keys else 3
        experience_level = st.selectbox(
            "Livello",
            options=level_labels,
            index=level_index,
        )
        selected_level = level_keys[level_labels.index(experience_level)]

        education = st.text_area("Formazione", value=current.education, height=100)
        default_passions = [p for p in current.passions if p in passion_options]
        passions = st.multiselect("Passioni suggerite per il campo", options=passion_options, default=default_passions)
        custom_passions_existing = [p for p in current.passions if p not in passion_options]
        custom_passions_text = st.text_input(
            "Altre passioni (separate da virgola)",
            value=", ".join(custom_passions_existing),
            help="Puoi aggiungere passioni non presenti nella lista suggerita.",
            placeholder="es. Wine Tech, EdTech, Luxury",
        )

        default_roles = ", ".join(current.target_roles) if current.target_roles else ", ".join(role_hints[:2])
        roles_text = st.text_input("Ruoli target (keyword separate da virgola)", value=default_roles)
        desired_salary = st.number_input(
            "RAL desiderata (EUR lordi/anno)",
            min_value=0,
            value=current.desired_salary_eur or 0,
            step=1000,
        )
        location = st.text_input(
            "Città / paesi (separate da virgola)",
            value=current.location,
            help=(
                "Ogni voce separata da virgola genera ricerche distinte: "
                "es. Milano, Italy, Spain → query su Milano, su tutta Italy e su tutta Spain."
            ),
            placeholder="Milano, Italy, Spain",
        )
        work_mode = st.selectbox(
            "Modalità di lavoro",
            options=WORK_MODE_OPTIONS,
            index=WORK_MODE_OPTIONS.index(current.work_mode),
        )
        free_text = st.text_area("Preferenze libere", value=current.free_text_preferences, height=120)

        st.markdown("### Criteri rilevanti")
        st.caption(
            "Criteri che scartano a priori le candidature se non vengono soddisfatti esattamente, "
            "prima del matching AI. Attiva solo quelli obbligatori per te."
        )
        crit_location = st.checkbox("Città / area geografica", value=current.fundamental_criteria.location)
        crit_role = st.checkbox(
            "Ruolo target",
            value=current.fundamental_criteria.target_role,
            help="DeepSeek verifica se il ruolo dell'annuncio è compatibile (anche con sinonimi/traduzioni). Se poco chiaro, passa all'AI.",
        )
        crit_salary = st.checkbox(
            "Almeno questa RAL (max −4.000 € se indicata)",
            value=current.fundamental_criteria.salary,
            help=(
                "Se la RAL è nell'annuncio, deve essere entro la tolleranza. "
                "Se non è indicata, l'annuncio passa comunque: la RAL viene cercata "
                "su fonti web (Glassdoor, Levels.fyi, ecc.) tramite i provider di ricerca."
            ),
        )
        crit_work_mode = st.checkbox("Modalità di lavoro", value=current.fundamental_criteria.work_mode)
        crit_level = st.checkbox(
            "Livello esperienza",
            value=current.fundamental_criteria.experience_level,
            help="Filtra gli annunci in base al livello indicato nel testo (stage, junior, senior, ecc.).",
        )

        rule_keys = [key for key, _ in EXPERIENCE_LEVEL_RULE_OPTIONS]
        rule_labels = [label for _, label in EXPERIENCE_LEVEL_RULE_OPTIONS]
        current_rule = current.experience_level_rule
        rule_index = rule_keys.index(current_rule.mode) if current_rule.mode in rule_keys else 0
        level_rule_mode = current_rule.mode
        level_rule_offset = current_rule.offset

        if crit_level:
            selected_rule_label = st.radio(
                "Regola livello",
                options=rule_labels,
                index=rule_index,
                help=(
                    "Questo livello: solo annunci al tuo livello. "
                    "O inferiori: anche posizioni meno senior. "
                    "X inferiori/superiori: includi fino a X gradini nella gerarchia."
                ),
            )
            level_rule_mode = rule_keys[rule_labels.index(selected_rule_label)]
            if level_rule_mode in {"or_higher", "or_lower"}:
                direction = "inferiori" if level_rule_mode == "or_lower" else "superiori"
                level_rule_offset = st.number_input(
                    f"X ({direction})",
                    min_value=0,
                    max_value=5,
                    value=current_rule.offset,
                    help=f"Quanti livelli {direction} al tuo includere oltre al livello selezionato.",
                )

        submitted = st.form_submit_button("Salva profilo", type="primary", use_container_width=True)

    if submitted:
        roles = [role.strip() for role in roles_text.split(",") if role.strip()]
        extra_passions = [p.strip() for p in custom_passions_text.split(",") if p.strip()]
        all_passions = list(dict.fromkeys([*passions, *extra_passions]))
        profile = UserProfile(
            career_field=selected_field,  # type: ignore[arg-type]
            experience_level=selected_level,  # type: ignore[arg-type]
            education=education.strip(),
            passions=all_passions,
            target_roles=roles,
            desired_salary_eur=desired_salary if desired_salary > 0 else None,
            location=location.strip(),
            work_mode=work_mode,  # type: ignore[arg-type]
            free_text_preferences=free_text.strip(),
            fundamental_criteria=FundamentalCriteria(
                location=crit_location,
                target_role=crit_role,
                salary=crit_salary,
                work_mode=crit_work_mode,
                experience_level=crit_level,
            ),
            experience_level_rule=ExperienceLevelRule(
                mode=level_rule_mode,  # type: ignore[arg-type]
                offset=level_rule_offset,
            ),
        )
        profile.save(paths.profile_path)
        st.success(f"Profilo «{paths.display_name}» salvato.")


def _card_key(result: MatchResult, prefix: str, suffix: str = "") -> str:
    base = hashlib.md5(result.job.dedup_key.encode()).hexdigest()
    parts = [prefix, base]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


def _render_match_card(
    result: MatchResult,
    saved_store: SavedJobsStore | None = None,
    *,
    allow_save: bool = True,
    key_prefix: str = "card",
    key_suffix: str = "",
) -> None:
    color = _score_color(result.match_score)
    st.markdown(
        f"""
        <div class="job-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <h3 style="margin:0;">{result.job.title}</h3>
                    <p style="margin:0.2rem 0;color:#4b5563;">{result.job.company} · {result.job.location}</p>
                </div>
                <span class="score-badge" style="background:{color};">
                    Score {result.match_score:.1f}/10
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not result.salary_indicated:
        st.caption("**RAL non indicata nell'annuncio**")
        if result.estimated_salary_eur:
            st.caption(f"Stima da ricerca web (azienda/ruolo): **{result.estimated_salary_eur}**")
        if result.salary_research_summary:
            st.caption(result.salary_research_summary)
        elif not result.estimated_salary_eur:
            st.caption(
                "Nessuna stima affidabile da fonti esterne; la mancanza di trasparenza "
                "può essere un fattore negativo."
            )
        else:
            st.caption("La mancanza di trasparenza sulla RAL può essere un fattore negativo.")
    key = _card_key(result, key_prefix, key_suffix)
    channel = result.application_channel
    if channel != "unknown" or result.cv_strategy:
        channel_label = APPLICATION_CHANNEL_LABELS.get(channel, APPLICATION_CHANNEL_LABELS["unknown"])
        type_hint = CV_TYPE_HINTS.get(channel, "")
        with st.expander("Che CV inviare?", expanded=False, key=f"cv-{key}"):
            if channel != "unknown":
                st.markdown(f"**Canale probabile:** {channel_label}")
                if type_hint:
                    st.caption(type_hint)
            if result.cv_strategy:
                st.write(result.cv_strategy)
            elif channel == "unknown":
                st.write("Segnali insufficienti per consigliare un tipo di CV specifico.")
    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        st.link_button(
            "Apri annuncio",
            result.job.url,
            use_container_width=True,
            key=f"link-{key}",
        )
    with action_col2:
        if allow_save and saved_store is not None:
            if saved_store.is_saved(result.job.dedup_key):
                st.button("Salvata", disabled=True, use_container_width=True, key=f"saved-{key}")
            elif st.button("Salva candidatura", use_container_width=True, key=f"save-{key}"):
                saved_store.add(result)
                st.toast(f"Salvata: {result.job.title} @ {result.job.company}")
                st.rerun()
    with st.expander("Motivazione AI", expanded=False, key=f"reason-{key}"):
        st.write(result.reasoning)


def _run_live_scan(profile: UserProfile, paths: ProfilePaths) -> ScanResult | None:
    status_box = st.empty()
    progress_bar = st.progress(0.0, text="Preparazione scansione...")
    metrics_cols = st.columns(5)
    found_metric = metrics_cols[0].empty()
    new_metric = metrics_cols[1].empty()
    eligible_metric = metrics_cols[2].empty()
    analyzed_metric = metrics_cols[3].empty()
    promoted_metric = metrics_cols[4].empty()
    st.markdown("#### Log attività")
    log_box = st.empty()
    provider_box = st.empty()
    st.markdown("#### Match promossi (in tempo reale)")
    results_container = st.container()

    saved_store = _saved_store(paths)

    log_lines: list[str] = []
    live_matches: list[MatchResult] = []
    totals = {"found": 0, "new": 0, "eligible": 0, "analyzed": 0, "promoted": 0}

    def append_log(message: str) -> None:
        log_lines.append(message)
        log_box.code("\n".join(log_lines[-14:]), language=None)

    def refresh_metrics() -> None:
        found_metric.metric("Trovati", totals["found"])
        new_metric.metric("Nuovi", totals["new"])
        eligible_metric.metric("Idonei", totals["eligible"])
        analyzed_metric.metric("Analizzati AI", totals["analyzed"])
        promoted_metric.metric("Promossi", totals["promoted"])

    def on_progress(event: str, payload: dict) -> None:
        if event == "status":
            status_box.info(payload["message"])
            append_log(payload["message"])
        elif event == "agent_done":
            message = f"{payload['agent']}: {payload['count']} annunci"
            append_log(message)
            status_box.info(message)
        elif event == "summary":
            totals["found"] = payload["total_found"]
            totals["new"] = payload["new_jobs"]
            totals["eligible"] = payload.get("eligible_jobs", payload["new_jobs"])
            refresh_metrics()
            append_log(
                f"Totale {payload['total_found']} | Nuovi {payload['new_jobs']} | "
                f"Pre-filtrati {payload.get('prefilter_skipped', 0)} | "
                f"Idonei per AI {payload.get('eligible_jobs', 0)}"
            )
        elif event == "analyzing":
            current = payload["current"]
            total = max(payload["total"], 1)
            progress_bar.progress(current / total, text=f"AI {current}/{total}: {payload['title']} @ {payload['company']}")
            totals["analyzed"] = current - 1
            refresh_metrics()
        elif event == "match":
            totals["analyzed"] = payload["current"]
            refresh_metrics()
        elif event == "promoted":
            result = MatchResult.model_validate(payload["result"])
            live_matches.append(result)
            totals["promoted"] = len(live_matches)
            refresh_metrics()
            append_log(f"PROMOSSO [{result.match_score:.1f}] {result.job.title} @ {result.job.company}")
            with results_container:
                _render_match_card(
                    result,
                    saved_store,
                    key_prefix="scan-live",
                    key_suffix=str(len(live_matches) - 1),
                )
        elif event == "companies_discovered":
            companies = payload.get("companies", [])
            if companies:
                append_log("── Nuove aziende ATS aggiunte a Target Hunter ──")
                for company in companies:
                    append_log(
                        f"  {company.get('name')} ({company.get('ats')}/{company.get('slug')})"
                    )
        elif event == "search_providers":
            stats = payload.get("stats", {})
            st.session_state["last_provider_stats"] = stats
            phase = payload.get("phase", "ricerca")
            append_log(f"── Provider ricerca ({phase}) ──")
            for line in _format_provider_stats_lines(stats):
                append_log(f"  {line}")
            with provider_box.container():
                st.markdown(f"**Provider ricerca** ({phase})")
                for line in _format_provider_stats_lines(stats):
                    st.caption(line)
        elif event == "complete":
            progress_bar.progress(1.0, text="Scansione completata")
            status_box.success("Scansione completata.")
            append_log("Scansione completata.")
            if payload.get("provider_stats"):
                st.session_state["last_provider_stats"] = payload["provider_stats"]
                append_log("── Provider ricerca (totale scansione) ──")
                for line in _format_provider_stats_lines(payload["provider_stats"]):
                    append_log(f"  {line}")
                with provider_box.container():
                    st.markdown("**Provider ricerca** (totale scansione)")
                    for line in _format_provider_stats_lines(payload["provider_stats"]):
                        st.caption(line)

    refresh_metrics()

    try:
        with st.status("Scansione in corso...", expanded=True) as scan_status:
            scan_result = run_scan_sync(
                profile,
                on_progress=on_progress,
                memory_path=paths.memory_path,
                scan_results_path=paths.scan_results_path,
                scan_history_path=paths.scan_history_path,
                discovered_companies_path=paths.discovered_companies_path,
            )
            scan_status.update(label="Scansione completata", state="complete", expanded=False)
        st.session_state["last_scan_result"] = scan_result.model_dump(mode="json")
        st.session_state["live_matches"] = [match.model_dump(mode="json") for match in live_matches]
        return scan_result
    except Exception as exc:
        status_box.error(f"Scansione fallita: {exc}")
        append_log(f"ERRORE: {exc}")
        return None


def render_saved_tab(paths: ProfilePaths) -> None:
    st.subheader("Candidature salvate")
    st.caption(
        f"Profilo: **{paths.display_name}**. Le candidature che salvi restano qui per il futuro. "
        "Non vengono rianalizzate nelle scansioni successive."
    )

    saved_store = _saved_store(paths)
    applications = saved_store.list_sorted()

    if not applications:
        st.info("Nessuna candidatura salvata. Salva i match che ti interessano dalla Dashboard.")
        return

    st.metric("Totale salvate", len(applications))

    sort_mode = _render_sort_selectbox(
        SAVED_SORT_OPTIONS,
        f"saved_sort_{paths.slug}",
    )
    if sort_mode == "match_desc":
        applications = sorted(applications, key=lambda app: app.match.match_score, reverse=True)
    elif sort_mode == "match_asc":
        applications = sorted(applications, key=lambda app: app.match.match_score)
    elif sort_mode == "salary_desc":
        applications = sorted(applications, key=lambda app: match_salary_sort_key(app.match))
    elif sort_mode == "saved_asc":
        applications = sorted(applications, key=lambda app: app.saved_at)
    else:
        applications = sorted(applications, key=lambda app: app.saved_at, reverse=True)

    for index, application in enumerate(applications):
        result = application.match
        card_key = _card_key(result, "saved", str(index))
        st.caption(f"Salvata il {application.saved_at.strftime('%d/%m/%Y %H:%M UTC')}")
        _render_match_card(
            result,
            saved_store,
            allow_save=False,
            key_prefix="saved",
            key_suffix=str(index),
        )
        if st.button(
            "Rimuovi dai salvati",
            key=f"remove-{card_key}",
            use_container_width=True,
        ):
            saved_store.remove(result.job.dedup_key)
            st.toast("Candidatura rimossa.")
            st.rerun()
        st.divider()


def render_history_tab(paths: ProfilePaths) -> None:
    st.subheader("Log match")
    st.caption(
        f"Profilo: **{paths.display_name}**. Storico dei job promossi dalle scansioni, "
        "raggruppati per giorno con motivazione AI."
    )

    history = ScanHistoryStore(paths.scan_history_path)
    history.migrate_if_needed(paths.memory_path, paths.scan_results_path)
    saved_store = _saved_store(paths)
    grouped = history.group_by_day()

    if not grouped:
        st.info("Nessun match nello storico. Avvia una scansione dalla Dashboard.")
        return

    scan_count = sum(len(scans) for _, scans in grouped)
    col1, col2 = st.columns(2)
    col1.metric("Match totali", history.total_matches())
    col2.metric("Giorni con match", len(grouped))

    layout_mode = _render_sort_selectbox(
        HISTORY_LAYOUT_OPTIONS,
        f"history_layout_{paths.slug}",
        label="Visualizzazione",
    )

    if layout_mode in {"match_desc", "match_asc", "salary_desc"}:
        flat_matches: list[tuple[MatchResult, str]] = []
        for day, scans in grouped:
            for scan in scans:
                local_time = scan.scanned_at.astimezone(ROME)
                scan_label = f"{format_italian_date(day)} · scansione {local_time.strftime('%H:%M')}"
                for result in scan.matches:
                    flat_matches.append((result, scan_label))
        if layout_mode == "salary_desc":
            flat_matches.sort(key=lambda item: match_salary_sort_key(item[0]))
        else:
            flat_matches.sort(
                key=lambda item: item[0].match_score,
                reverse=layout_mode == "match_desc",
            )
        grouped_flat: dict[str, list[tuple[int, MatchResult, str]]] = {}
        for index, (result, scan_label) in enumerate(flat_matches):
            day_label = scan_label.split(" · ", 1)[0]
            grouped_flat.setdefault(day_label, []).append((index, result, scan_label))

        for day_index, (day_label, items) in enumerate(grouped_flat.items()):
            with st.expander(
                f"{day_label} · {len(items)} match",
                expanded=False,
                key=f"hist-flat-day-{paths.slug}-{day_index}",
            ):
                for index, result, scan_label in items:
                    st.caption(scan_label)
                    _render_match_card(
                        result,
                        saved_store,
                        key_prefix="history-flat",
                        key_suffix=str(index),
                    )
                    st.divider()
        return

    within_scan_sort = _render_sort_selectbox(
        MATCH_SORT_OPTIONS,
        f"history_scan_sort_{paths.slug}",
        label="Ordina match in ogni scansione",
    )

    for day, scans in grouped:
        day_matches = sum(len(scan.matches) for scan in scans)
        day_label = format_italian_date(day)
        with st.expander(
            f"{day_label} · {day_matches} match · {len(scans)} scansioni",
            expanded=False,
            key=f"hist-day-{paths.slug}-{day.isoformat()}",
        ):
            for scan_index, scan in enumerate(scans):
                local_time = scan.scanned_at.astimezone(ROME)
                scan_title = (
                    f"Scansione {local_time.strftime('%H:%M')} — "
                    f"{scan.total_promoted} promossi · {scan.total_found} trovati"
                )
                with st.expander(
                    scan_title,
                    expanded=False,
                    key=f"hist-scan-{paths.slug}-{day.isoformat()}-{scan_index}",
                ):
                    sorted_matches = _sort_matches(scan.matches, within_scan_sort)
                    for match_index, result in enumerate(sorted_matches):
                        _render_match_card(
                            result,
                            saved_store,
                            key_prefix="history",
                            key_suffix=f"{day.isoformat()}-{scan_index}-{match_index}",
                        )
                        if match_index < len(sorted_matches) - 1:
                            st.divider()


def render_dashboard_tab(paths: ProfilePaths) -> None:
    st.subheader("Dashboard")

    profile = UserProfile.load(paths.profile_path)
    if not profile:
        st.warning("Salva prima il profilo nella tab Profilo.")
        return

    field_label = CAREER_FIELDS.get(profile.career_field, {}).get("label", profile.career_field)
    st.caption(
        f"Profilo: **{paths.display_name}** · Campo: **{field_label}** · "
        f"Livello: **{profile.experience_level}** · {profile.location}"
    )

    saved_store = _saved_store(paths)

    scan_running = st.session_state.get("scan_running", False)
    if st.button("Avvia Scansione", type="primary", use_container_width=True, disabled=scan_running):
        st.session_state["scan_running"] = True
        st.session_state["live_matches"] = []
        _run_live_scan(profile, paths)
        st.session_state["scan_running"] = False

    scan_data = st.session_state.get("last_scan_result")
    if not scan_data:
        cached = ScanResult.load(paths.scan_results_path)
        if cached:
            scan_data = cached.model_dump(mode="json")

    if not scan_data:
        st.info("Nessun risultato. Avvia una scansione.")
        return

    scan_result = ScanResult.model_validate(scan_data)
    st.divider()
    st.markdown("#### Riepilogo ultima scansione")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trovati", scan_result.total_found)
    c2.metric("Analizzati AI", scan_result.total_analyzed)
    c3.metric("Pre-filtrati", scan_result.total_prefilter_skipped)
    c4.metric("Promossi", scan_result.total_promoted)
    c5.metric("Soglia", os.getenv("MATCH_SCORE_THRESHOLD", "7"))
    st.caption(f"Ultima scansione: {scan_result.scanned_at.isoformat()}")
    _render_provider_stats(st.session_state.get("last_provider_stats", {}))

    live_data = st.session_state.get("live_matches")
    if live_data:
        st.markdown("#### Match promossi")
        sort_mode = _render_sort_selectbox(MATCH_SORT_OPTIONS, f"dash_sort_{paths.slug}")
        live_matches = _sort_matches(
            [MatchResult.model_validate(item) for item in live_data],
            sort_mode,
        )
        for index, result in enumerate(live_matches):
            _render_match_card(
                result,
                saved_store,
                key_prefix="dash-live",
                key_suffix=str(index),
            )
        return

    if not scan_result.matches:
        st.warning("Nessun match promosso nell'ultima scansione.")
        return

    st.markdown("#### Match promossi")
    sort_mode = _render_sort_selectbox(MATCH_SORT_OPTIONS, f"dash_sort_{paths.slug}")
    sorted_matches = _sort_matches(scan_result.matches, sort_mode)
    for index, result in enumerate(sorted_matches):
        _render_match_card(
            result,
            saved_store,
            key_prefix="dash",
            key_suffix=str(index),
        )


def main() -> None:
    st.set_page_config(page_title="AI Job Hunter", page_icon="🎯", layout="wide")
    _inject_styles()

    st.markdown('<p class="main-title">AI Job Hunter</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Pipeline multi-agente per la ricerca intelligente di lavoro.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        profile_paths = _render_profile_selector()
        st.divider()
        st.header("Stato")
        _api_key_status("DeepSeek", "DEEPSEEK_API_KEY")
        st.caption("Ricerca annunci (ordine fallback)")
        _api_key_status("SerpApi", "SERPAPI_API_KEY")
        _api_key_status("Serper", "SERPER_API_KEY")
        _api_key_status("DataForSEO", "DATAFORSEO_LOGIN")
        _api_key_status("Apify", "APIFY_API_TOKEN")
        _api_key_status("ScraperAPI", "SCRAPERAPI_API_KEY")
        exhausted = exhausted_providers()
        if exhausted:
            labels = ", ".join(name.capitalize() for name in exhausted)
            st.caption(f"Quota esaurita questo mese: {labels}")
        st.caption("Fallback: DuckDuckGo → DeepSeek web")
        st.caption(f"Modello: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")

    profile_tab, dashboard_tab, history_tab, saved_tab = st.tabs(
        ["Profilo", "Dashboard", "Log match", "Salvati"],
    )
    with profile_tab:
        render_profile_tab(profile_paths)
    with dashboard_tab:
        render_dashboard_tab(profile_paths)
    with history_tab:
        render_history_tab(profile_paths)
    with saved_tab:
        render_saved_tab(profile_paths)


if __name__ == "__main__":
    main()
