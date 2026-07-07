# AI Job Hunter

Pipeline multi-agente per la ricerca intelligente di offerte di lavoro. L'app raccoglie annunci da aziende target e da startup, li pre-filtra in base al profilo utente e li valuta con modelli AI per restituire solo i match più rilevanti.

Funziona in **locale** tramite interfaccia web [Streamlit](https://streamlit.io/).

## Come funziona

1. **Profilo** — Configuri ruoli target, località, modalità di lavoro, stipendio desiderato e preferenze testuali.
2. **Raccolta annunci** — Due agenti lavorano in parallelo:
   - **Target Hunter**: interroga le API ATS (Lever, Greenhouse) delle aziende in `config/target_companies.json`.
   - **Startup Discoverer**: usa SerpApi per trovare annunci su job board e siti di startup.
3. **Pre-filtro** — Criteri fondamentali (località, ruolo, stipendio, ecc.) e matcher AI leggeri riducono il volume prima dell'analisi completa.
4. **Matching AI** — DeepSeek valuta ogni annuncio rimasto e assegna un punteggio da 0 a 10.
5. **Risultati** — Gli annunci con score ≥ soglia configurata vengono promossi e mostrati nella dashboard.

## Requisiti

- Python 3.11+
- Chiave API [DeepSeek](https://platform.deepseek.com/)
- Chiave API [SerpApi](https://serpapi.com/)

## Installazione

```bash
git clone <url-del-repo>
cd ai_proj

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Configurazione

### Variabili d'ambiente

Il progetto **non include** il file `.env` con le credenziali reali. Per far funzionare l'app in locale devi crearlo tu:

```bash
cp .env.example .env
```

Apri `.env` e inserisci **i tuoi** valori per ogni variabile:

| Variabile | Descrizione |
|-----------|-------------|
| `DEEPSEEK_API_KEY` | Chiave API DeepSeek |
| `DEEPSEEK_MODEL` | Modello da usare (default: `deepseek-chat`) |
| `DEEPSEEK_BASE_URL` | Endpoint API (default: `https://api.deepseek.com`) |
| `SERPAPI_API_KEY` | Chiave API SerpApi |
| `MATCH_SCORE_THRESHOLD` | Soglia minima di match (default: `7`) |

### Profilo utente

Copia il profilo di esempio e personalizzalo:

```bash
cp config/user_profile.example.json config/user_profile.json
```

Modifica `config/user_profile.json` con i tuoi dati (ruoli, località, preferenze, ecc.). Anche questo file è escluso da Git per non pubblicare informazioni personali.

## Avvio

```bash
streamlit run app.py
```

L'app si apre nel browser (di default su `http://localhost:8501`).

1. Vai alla tab **Profilo**, compila i campi e salva.
2. Vai alla tab **Dashboard** e clicca **Avvia Scansione**.
3. Controlla nella sidebar che le API key risultino configurate.

## Struttura del progetto

```
ai_proj/
├── app.py                  # Interfaccia Streamlit
├── orchestrator.py         # Coordinamento pipeline e agenti
├── agents/
│   ├── target_hunter.py    # Annunci da aziende target (Lever/Greenhouse)
│   ├── startup_discoverer.py
│   ├── ai_matcher.py       # Valutazione AI completa
│   ├── location_matcher.py
│   ├── role_matcher.py
│   ├── job_prefilter.py
│   └── keyword_expander.py
├── models/                 # Modelli Pydantic (Job, Profilo, Risultati)
├── storage/                # Memoria annunci già visti
├── config/
│   ├── user_profile.example.json
│   ├── target_companies.json
│   └── career_fields.json
├── data/                   # Risultati scan e memoria (generati a runtime)
├── requirements.txt
├── .env.example            # Template variabili d'ambiente (senza valori)
└── .gitignore
```

## File esclusi da Git

Per sicurezza, i seguenti file **non vanno caricati su GitHub**:

- `.env` — credenziali API
- `config/user_profile.json` — dati personali
- `data/` — risultati delle scansioni e memoria locale
- `.venv/` — ambiente virtuale Python

Sono già elencati in `.gitignore`.

## Licenza

Uso personale / educativo. Verifica i termini d'uso delle API DeepSeek e SerpApi prima dell'utilizzo.
