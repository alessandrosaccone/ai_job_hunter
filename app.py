from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from models.job import MatchResult, ScanResult
from models.user_profile import FundamentalCriteria, UserProfile
from orchestrator import run_scan_sync

load_dotenv()

PROFILE_PATH = Path("config/user_profile.json")
EXAMPLE_PROFILE_PATH = Path("config/user_profile.example.json")
SCAN_RESULTS_PATH = Path("data/scan_results.json")
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


def _load_profile_for_form() -> UserProfile:
    profile = UserProfile.load(PROFILE_PATH)
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


def _score_color(score: float) -> str:
    if score >= 8:
        return "#16a34a"
    if score >= 7:
        return "#ca8a04"
    return "#dc2626"


def _career_field_labels() -> list[tuple[str, str]]:
    return [(key, value.get("label", key)) for key, value in CAREER_FIELDS.items()]


def render_profile_tab() -> None:
    current = _load_profile_for_form()
    field_labels = _career_field_labels()
    field_keys = [key for key, _ in field_labels]
    field_display = [label for _, label in field_labels]

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

    with st.form("profile_form"):
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
            help="Es: Milano, Madrid, Italy, Spain. DeepSeek gestisce traduzioni (Milan/Milano) e match per paese.",
            placeholder="Milano, Italy",
        )
        work_mode = st.selectbox(
            "Modalità di lavoro",
            options=WORK_MODE_OPTIONS,
            index=WORK_MODE_OPTIONS.index(current.work_mode),
        )
        free_text = st.text_area("Preferenze libere", value=current.free_text_preferences, height=120)

        st.markdown("### Criteri fondamentali")
        st.caption(
            "I criteri con DeepSeek (città, ruolo) usano poche token in batch prima del matching completo. "
            "RAL e altri criteri sono istantanei."
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
            help="Se la RAL non è nell'annuncio, l'annuncio passa comunque all'analisi AI.",
        )
        crit_work_mode = st.checkbox("Modalità di lavoro", value=current.fundamental_criteria.work_mode)
        crit_level = st.checkbox("Livello esperienza", value=current.fundamental_criteria.experience_level)

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
        )
        profile.save(PROFILE_PATH)
        st.success("Profilo salvato.")


def _render_match_card(result: MatchResult) -> None:
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
    st.link_button(
        "Apri annuncio",
        result.job.url,
        use_container_width=False,
        key=f"link-{hashlib.md5(result.job.dedup_key.encode()).hexdigest()}",
    )
    with st.expander("Motivazione AI"):
        st.write(result.reasoning)


def _run_live_scan(profile: UserProfile) -> ScanResult | None:
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
    st.markdown("#### Match promossi (in tempo reale)")
    results_container = st.container()

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
                _render_match_card(result)
        elif event == "complete":
            progress_bar.progress(1.0, text="Scansione completata")
            status_box.success("Scansione completata.")
            append_log("Scansione completata.")

    refresh_metrics()

    try:
        with st.status("Scansione in corso...", expanded=True) as scan_status:
            scan_result = run_scan_sync(profile, on_progress=on_progress)
            scan_status.update(label="Scansione completata", state="complete", expanded=False)
        st.session_state["last_scan_result"] = scan_result.model_dump(mode="json")
        st.session_state["live_matches"] = [match.model_dump(mode="json") for match in live_matches]
        return scan_result
    except Exception as exc:
        status_box.error(f"Scansione fallita: {exc}")
        append_log(f"ERRORE: {exc}")
        return None


def render_dashboard_tab() -> None:
    st.subheader("Dashboard")

    profile = UserProfile.load(PROFILE_PATH)
    if not profile:
        st.warning("Salva prima il profilo nella tab Profilo.")
        return

    field_label = CAREER_FIELDS.get(profile.career_field, {}).get("label", profile.career_field)
    st.caption(f"Campo: **{field_label}** · Livello: **{profile.experience_level}** · {profile.location}")

    scan_running = st.session_state.get("scan_running", False)
    if st.button("Avvia Scansione", type="primary", use_container_width=True, disabled=scan_running):
        st.session_state["scan_running"] = True
        st.session_state["live_matches"] = []
        _run_live_scan(profile)
        st.session_state["scan_running"] = False

    scan_data = st.session_state.get("last_scan_result")
    if not scan_data:
        cached = ScanResult.load(SCAN_RESULTS_PATH)
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

    live_data = st.session_state.get("live_matches")
    if live_data:
        st.markdown("#### Match promossi")
        for item in live_data:
            _render_match_card(MatchResult.model_validate(item))
        return

    if not scan_result.matches:
        st.warning("Nessun match promosso nell'ultima scansione.")
        return

    for result in scan_result.matches:
        _render_match_card(result)


def main() -> None:
    st.set_page_config(page_title="AI Job Hunter", page_icon="🎯", layout="wide")
    _inject_styles()

    st.markdown('<p class="main-title">AI Job Hunter</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Pipeline multi-agente per la ricerca intelligente di lavoro.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Stato")
        _api_key_status("DeepSeek", "DEEPSEEK_API_KEY")
        _api_key_status("SerpApi", "SERPAPI_API_KEY")
        st.caption(f"Modello: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")

    profile_tab, dashboard_tab = st.tabs(["Profilo", "Dashboard"])
    with profile_tab:
        render_profile_tab()
    with dashboard_tab:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
