"""Unified Streamlit hub for the ML-Experiment-Projects repository."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / ".hub" / "logs"

WORKSPACES: Dict[str, dict] = {
    "mline": {
        "name": "MLine",
        "label": "Enterprise AutoML Lab",
        "summary": (
            "AutoML focado em classificacao, regressao, series temporais e NLP, "
            "com gerenciamento de experimentos, MLflow e explicabilidade."
        ),
        "folder": ROOT_DIR / "MLine",
        "script": ROOT_DIR / "MLine" / "app.py",
        "port": 8502,
        "command": 'streamlit run "MLine/app.py"',
        "requirements": ROOT_DIR / "MLine" / "requirements.txt",
        "highlights": [
            "Experimentos persistidos com MLflow e SQLite",
            "Fluxos para tabular, time series e text classification",
            "Resultados detalhados, leaderboard e exportacao de artefatos",
        ],
    },
    "studio": {
        "name": "AutoML Studio",
        "label": "SageMaker-inspired Workspace",
        "summary": (
            "Studio no-code inspirado em AWS SageMaker Canvas para ingestao, "
            "profiling, AutoML tabular e historico de predicoes."
        ),
        "folder": ROOT_DIR / "sagemaker based",
        "script": ROOT_DIR / "sagemaker based" / "app.py",
        "port": 8503,
        "command": 'streamlit run "sagemaker based/app.py"',
        "requirements": ROOT_DIR / "sagemaker based" / "requirements.txt",
        "highlights": [
            "Workflows de datasets, build, modelos e predicoes",
            "AutoGluon com fallback para FLAML",
            "MLOps, MLflow e interface multipagina pronta para demo",
        ],
    },
    "pyramid": {
        "name": "Flexible Ensemble Pyramid",
        "label": "Hierarchical Ensemble Explorer",
        "summary": (
            "Experimento de ensemble hierarquico com visualizacao das camadas, "
            "metricas ao vivo e configuracao flexivel para NLP de sentimento."
        ),
        "folder": ROOT_DIR / "Flexible Ensemble Pyramid",
        "script": ROOT_DIR / "Flexible Ensemble Pyramid" / "flexible_ensemble_pyramid_ui_enhanced.py",
        "port": 8504,
        "command": 'streamlit run "Flexible Ensemble Pyramid/flexible_ensemble_pyramid_ui_enhanced.py"',
        "requirements": None,
        "highlights": [
            "Visualizacao interativa da piramide e das conexoes entre modelos",
            "Treino com RL meta-learner, NAS opcional e MLflow",
            "Agora suporta uso com arquivos CSV proprios ou dataset local do repo",
        ],
    },
}

WORKSPACE_ORDER = ["mline", "studio", "pyramid"]
OVERVIEW_KEY = "overview"


st.set_page_config(
    page_title="ML Experiment Projects Hub",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --hub-bg: #081119;
            --hub-panel: rgba(9, 20, 29, 0.84);
            --hub-panel-soft: rgba(12, 30, 43, 0.68);
            --hub-border: rgba(103, 190, 180, 0.18);
            --hub-text: #e9f5f2;
            --hub-muted: #9ec2bd;
            --hub-primary: #ff8a3d;
            --hub-secondary: #63d2c6;
            --hub-success: #56d39c;
            --hub-danger: #ff6c57;
            --hub-warning: #f7c66c;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: "Space Grotesk", sans-serif;
            background:
                radial-gradient(circle at 10% 10%, rgba(255, 138, 61, 0.18), transparent 28%),
                radial-gradient(circle at 90% 5%, rgba(99, 210, 198, 0.20), transparent 24%),
                linear-gradient(180deg, #050b11 0%, #081119 100%);
            color: var(--hub-text);
        }

        .stApp {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: rgba(3, 10, 15, 0.92);
            border-right: 1px solid var(--hub-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--hub-text);
        }

        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 2rem;
        }

        .hub-hero {
            padding: 2rem 2rem 1.8rem 2rem;
            border: 1px solid var(--hub-border);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(255, 138, 61, 0.12), rgba(99, 210, 198, 0.10)),
                rgba(4, 12, 18, 0.88);
            box-shadow: 0 18px 54px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.4rem;
        }

        .hub-kicker {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(99, 210, 198, 0.16);
            color: var(--hub-secondary);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hub-title {
            margin: 0.9rem 0 0.35rem 0;
            font-size: 3rem;
            line-height: 1.03;
            color: var(--hub-text);
            font-weight: 700;
        }

        .hub-title-accent {
            color: var(--hub-primary);
        }

        .hub-subtitle {
            margin: 0;
            max-width: 900px;
            color: var(--hub-muted);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .hub-card {
            min-height: 285px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            border-radius: 22px;
            border: 1px solid var(--hub-border);
            background:
                linear-gradient(180deg, rgba(14, 29, 40, 0.96) 0%, rgba(7, 18, 27, 0.94) 100%);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
        }

        .hub-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }

        .hub-card-name {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--hub-text);
        }

        .hub-card-label {
            margin: 0.2rem 0 0 0;
            font-size: 0.82rem;
            color: var(--hub-secondary);
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .hub-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .hub-status-running {
            color: var(--hub-success);
            background: rgba(86, 211, 156, 0.12);
        }

        .hub-status-stopped {
            color: var(--hub-warning);
            background: rgba(247, 198, 108, 0.10);
        }

        .hub-card-summary {
            color: var(--hub-muted);
            font-size: 0.94rem;
            line-height: 1.65;
            margin-bottom: 1rem;
        }

        .hub-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 1rem;
        }

        .hub-chip {
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            background: rgba(99, 210, 198, 0.10);
            color: var(--hub-secondary);
            border: 1px solid rgba(99, 210, 198, 0.14);
        }

        .hub-meta {
            color: var(--hub-muted);
            font-size: 0.83rem;
            line-height: 1.7;
            margin-top: auto;
        }

        .hub-section {
            border: 1px solid var(--hub-border);
            border-radius: 22px;
            background: var(--hub-panel);
            padding: 1.2rem 1.2rem 1.1rem 1.2rem;
        }

        .hub-inline-note {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(99, 210, 198, 0.16);
            background: var(--hub-panel-soft);
            color: var(--hub-muted);
            line-height: 1.6;
        }

        .hub-code {
            font-family: "IBM Plex Mono", monospace;
            color: var(--hub-secondary);
        }

        .stButton > button,
        [data-testid="stLinkButton"] a {
            border-radius: 14px !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 0.65rem 1rem !important;
            background: linear-gradient(135deg, #ff8a3d, #ffb347) !important;
            color: #091117 !important;
            box-shadow: 0 10px 24px rgba(255, 138, 61, 0.22) !important;
        }

        .stButton > button:hover,
        [data-testid="stLinkButton"] a:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(255, 138, 61, 0.30) !important;
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--hub-border);
            border-radius: 18px;
            background: rgba(8, 22, 30, 0.78);
            padding: 0.45rem 0.8rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--hub-muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--hub-text) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.3rem;
            border-radius: 16px;
            background: rgba(6, 16, 24, 0.76);
            border: 1px solid var(--hub-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            color: var(--hub-muted);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255, 138, 61, 0.16) !important;
            color: var(--hub-primary) !important;
        }

        [data-testid="stCodeBlock"] {
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def workspace_url(workspace_key: str) -> str:
    return f"http://127.0.0.1:{WORKSPACES[workspace_key]['port']}"


def log_path(workspace_key: str) -> Path:
    return LOG_DIR / f"{workspace_key}.log"


def read_recent_log(workspace_key: str, max_chars: int = 5000) -> str:
    path = log_path(workspace_key)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:].strip()


def workspace_is_live(workspace_key: str, timeout: float = 1.2) -> bool:
    try:
        with urlopen(workspace_url(workspace_key), timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 500
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def launch_workspace(workspace_key: str, wait_seconds: int = 45) -> bool:
    if workspace_is_live(workspace_key):
        return True

    spec = WORKSPACES[workspace_key]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    launch_log = log_path(workspace_key)
    handle = launch_log.open("a", encoding="utf-8")
    handle.write(
        f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {spec['name']} on port {spec['port']}\n"
    )
    handle.flush()

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(spec["script"]),
        "--server.port",
        str(spec["port"]),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    popen_kwargs = {
        "cwd": str(spec["folder"]),
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": {**os.environ, "PYTHONUTF8": "1"},
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(command, **popen_kwargs)
    finally:
        handle.close()

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if workspace_is_live(workspace_key):
            return True
        time.sleep(1.0)
    return False


def status_label(workspace_key: str) -> tuple[str, str]:
    if workspace_is_live(workspace_key):
        return "Running", "hub-status hub-status-running"
    return "Stopped", "hub-status hub-status-stopped"


def set_workspace(workspace_key: str) -> None:
    st.session_state["selected_workspace"] = workspace_key


def render_sidebar() -> str:
    if "selected_workspace" not in st.session_state:
        st.session_state["selected_workspace"] = OVERVIEW_KEY

    option_map = {OVERVIEW_KEY: "Overview"}
    option_map.update({key: WORKSPACES[key]["name"] for key in WORKSPACE_ORDER})
    reverse_map = {label: key for key, label in option_map.items()}

    with st.sidebar:
        st.markdown("## Unified Hub")
        st.caption("Escolha um workspace para abrir o app sem sair da interface central.")

        current_label = option_map[st.session_state["selected_workspace"]]
        selected_label = st.radio(
            "Ambiente",
            options=list(option_map.values()),
            index=list(option_map.values()).index(current_label),
        )
        selected_workspace = reverse_map[selected_label]
        st.session_state["selected_workspace"] = selected_workspace

        st.divider()
        st.markdown("### Runtime")
        active_count = sum(1 for key in WORKSPACE_ORDER if workspace_is_live(key))
        st.metric("Workspaces ativos", active_count)
        st.metric("Workspaces totais", len(WORKSPACE_ORDER))

        st.divider()
        st.markdown("### Portas padrao")
        for key in WORKSPACE_ORDER:
            spec = WORKSPACES[key]
            st.caption(f"{spec['name']}: {workspace_url(key)}")

        st.divider()
        st.markdown("### Dica")
        st.caption(
            "O hub inicia o projeto selecionado em segundo plano quando necessario "
            "e o exibe abaixo em iframe."
        )

    return st.session_state["selected_workspace"]


def render_overview() -> None:
    st.markdown(
        """
        <section class="hub-hero">
            <span class="hub-kicker">ML Experiment Projects</span>
            <h1 class="hub-title">
                Um unico ponto de entrada para os
                <span class="hub-title-accent">3 laboratorios Streamlit</span>
            </h1>
            <p class="hub-subtitle">
                Este hub centraliza os projetos do repositorio em uma unica interface:
                MLine, AutoML Studio e Flexible Ensemble Pyramid. Escolha um workspace,
                abra o app integrado e use os projetos sem ficar alternando entre
                janelas ou comandos separados.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    total_live = sum(1 for key in WORKSPACE_ORDER if workspace_is_live(key))
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Projetos no repo", len(WORKSPACE_ORDER))
    metric_col2.metric("Workspaces ativos", total_live)
    metric_col3.metric("Entrada principal", "app.py")

    st.markdown("")

    card_columns = st.columns(3, gap="large")
    for column, workspace_key in zip(card_columns, WORKSPACE_ORDER):
        spec = WORKSPACES[workspace_key]
        status_text, status_class = status_label(workspace_key)
        with column:
            st.markdown(
                f"""
                <div class="hub-card">
                    <div class="hub-card-top">
                        <div>
                            <p class="hub-card-name">{spec['name']}</p>
                            <p class="hub-card-label">{spec['label']}</p>
                        </div>
                        <span class="{status_class}">{status_text}</span>
                    </div>
                    <p class="hub-card-summary">{spec['summary']}</p>
                    <div class="hub-chip-row">
                        {''.join(f'<span class="hub-chip">{item}</span>' for item in spec['highlights'])}
                    </div>
                    <div class="hub-meta">
                        Pasta: <span class="hub-code">{spec['folder'].name}</span><br>
                        Porta sugerida: <span class="hub-code">{spec['port']}</span><br>
                        Comando direto: <span class="hub-code">{spec['command']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Abrir {spec['name']}", key=f"open_{workspace_key}", use_container_width=True):
                set_workspace(workspace_key)
                st.rerun()

    st.markdown("")
    st.markdown(
        """
        <div class="hub-section">
            <h3 style="margin-top:0;">Como o hub funciona</h3>
            <div class="hub-inline-note">
                1. O hub roda em <span class="hub-code">streamlit run app.py</span>.<br>
                2. Ao abrir um workspace, ele sobe o app correspondente em segundo plano na porta definida.<br>
                3. O projeto e exibido dentro desta pagina via iframe, e voce ainda pode abrir a URL direta se quiser mais espaco.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace(workspace_key: str) -> None:
    spec = WORKSPACES[workspace_key]
    url = workspace_url(workspace_key)
    status_text, status_class = status_label(workspace_key)

    st.markdown(
        f"""
        <section class="hub-hero">
            <span class="hub-kicker">Workspace</span>
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1.5rem;flex-wrap:wrap;">
                <div>
                    <h1 class="hub-title" style="margin-bottom:0.4rem;">{spec['name']}</h1>
                    <p class="hub-subtitle" style="max-width:760px;">{spec['summary']}</p>
                </div>
                <span class="{status_class}" style="margin-top:0.4rem;">{status_text}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    info_col, action_col = st.columns([1.6, 1], gap="large")
    with info_col:
        st.markdown(
            f"""
            <div class="hub-section">
                <h3 style="margin-top:0;">Contexto rapido</h3>
                <div class="hub-inline-note">
                    Pasta: <span class="hub-code">{spec['folder']}</span><br>
                    Porta: <span class="hub-code">{url}</span><br>
                    Comando direto: <span class="hub-code">{spec['command']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        st.link_button("Abrir URL direta", url, use_container_width=True)
        if st.button("Atualizar status", key=f"refresh_{workspace_key}", use_container_width=True):
            st.rerun()

    tab_workspace, tab_guide, tab_logs = st.tabs(["Workspace", "Guia rapido", "Logs"])

    with tab_workspace:
        is_live = workspace_is_live(workspace_key)
        if not is_live:
            with st.spinner(f"Iniciando {spec['name']} em segundo plano..."):
                is_live = launch_workspace(workspace_key)

        if is_live:
            st.success(f"{spec['name']} disponivel em {url}")
            st.caption(
                "Se o iframe demorar para responder ou voce quiser mais area de trabalho, use o botao 'Abrir URL direta'."
            )
            components.iframe(url, height=1180, scrolling=True)
        else:
            st.error(
                f"Nao foi possivel confirmar o startup de {spec['name']} na porta {spec['port']}."
            )
            st.code(read_recent_log(workspace_key) or "Nenhum log disponivel ainda.", language="text")

    with tab_guide:
        st.markdown("### O que voce encontra aqui")
        for item in spec["highlights"]:
            st.markdown(f"- {item}")

        if spec["requirements"] is not None:
            st.markdown("### Dependencias")
            st.code(f'pip install -r "{spec["requirements"]}"', language="bash")
        else:
            st.markdown("### Dependencias")
            st.markdown(
                "- Este projeto usa as bibliotecas declaradas nos scripts da raiz e pode compartilhar o mesmo ambiente dos outros apps."
            )

        st.markdown("### Execucao direta")
        st.code(spec["command"], language="bash")

        if workspace_key == "pyramid":
            st.markdown("### Formato esperado para CSV proprio")
            st.markdown("- Colunas obrigatorias: `text` e `sentiment`.")
            st.markdown("- Colunas opcionais: `tweet_id` e `entity`.")
            st.markdown("- Se voce enviar apenas o treino, a validacao e criada automaticamente por split.")

    with tab_logs:
        log_text = read_recent_log(workspace_key)
        if log_text:
            st.code(log_text, language="text")
        else:
            st.info("Os logs aparecem aqui depois da primeira tentativa de inicializacao.")


def main() -> None:
    inject_styles()
    selected_workspace = render_sidebar()
    if selected_workspace == OVERVIEW_KEY:
        render_overview()
        return
    render_workspace(selected_workspace)


if __name__ == "__main__":
    main()
