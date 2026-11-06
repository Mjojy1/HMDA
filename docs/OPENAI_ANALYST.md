# OpenAI-connected HMDA analyst in Domino

The existing application and agent execute in Domino. LLM inference executes at
OpenAI: this option is not entirely Domino-hosted inference. The agent uses the
existing read-only HMDA tools; the saved classifier, Dataset and training workflow
are unchanged. No PydanticAI or OpenAI SDK installation is required: the adapter
uses the already-pinned `requests` dependency and native Chat Completions tool calls.

## Configure the server runtime

1. Create a project API key in the [OpenAI API dashboard](https://platform.openai.com/api-keys).
   Ensure that the project has API funding/quota and access to the chosen model.
2. Supply `OPENAI_API_KEY` through your approved Domino secret/environment-variable
   configuration for the runtime that will execute the backend. Do not put it in
   source, notebook cells/outputs, Git, frontend variables, logs, or chat messages.
3. Set these non-secret backend environment variables:

   ```text
   HMDA_LLM_PROVIDER=openai
   HMDA_OPENAI_MODEL=gpt-5.4-mini
   ```

4. Restart the backend in that configured runtime. Workspace shell settings do not
   automatically transfer to a separate App/Preview container. Configure that
   runtime separately when preparing an approved preview update.

For a temporary test in a **private Bash terminal inside the Domino Workspace**,
enter the key using a hidden prompt (the key is not part of command history):

```bash
read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
export HMDA_LLM_PROVIDER=openai
export HMDA_OPENAI_MODEL=gpt-5.4-mini
python scripts/check_analyst_connection.py
```

Use the prepared project Python (`.venv/bin/python` in the existing workspace if
needed). Do not run the prompt through an assistant tool or paste the key into chat.
This live check consumes API tokens and sends a fixed aggregate research question
and resulting evidence to OpenAI. It does not write reports or train models.

If `VERIFIED` prints and the command exits 0, the model completed a multi-step
conversation and cited model, fairness, and definitions evidence. The script checks
actual tool use, not just a successful “Hello” response. Missing reports can prevent
this check; validate existing artifacts first with `python -m scripts.validate_artifacts`.

To review the UI, launch/restart the backend from the same configured shell on your
available workspace port, using the existing startup script. Do not start a second
process on an occupied port. Open **Analyst copilot** and ask:

> Review model performance and fairness, and explain the target.

Expect tool reads, observation IDs and final citations. `/api/analyst/status` exposes
configuration only, not a live health probe: `connection_verified` remains false
there. A successful analyst request or the explicit check verifies live connectivity.
No automatic inference call occurs when loading the status panel.

## Model and provider behavior

- Default model: `gpt-5.4-mini`, matching the requested example. It supports native
  function calling and Chat Completions. Set `HMDA_OPENAI_MODEL` explicitly to pin a
  supported snapshot if needed; alternate models must support this API/tool contract.
- OpenAI selection is explicit. Merely having `OPENAI_API_KEY` does not activate it.
- `HMDA_LLM_PROVIDER=local` explicitly selects non-agentic local topic lookup.
- `HMDA_LLM_PROVIDER=gateway` uses the existing Domino Gateway adapter/settings.
  Existing Gateway configuration still works when the provider variable is absent.
- Selecting OpenAI with no key fails visibly; it does not silently use local lookup
  or another provider. Authentication, permissions/model access, rate/quota limits,
  timeouts, malformed calls and network failures have sanitized errors.
- Requests use the fixed `https://api.openai.com/v1/chat/completions` URL, refuse
  redirects, set `store=false`, require one tool call per round, and cap completion
  tokens at 2,048 per request. The selected mini model uses reasoning effort `none`.
  The existing loop caps calls at six, including the final evidence selection;
  it does not automatically retry. This bounds each question, not total application
  spend across users. Restrict App access and configure provider usage controls.

## Data and evidence boundaries

OpenAI receives the current question, up to six prior user questions, tool schemas,
and selected aggregate observations (including report metadata such as hashes and
source paths when present). Tools do not send individual Parquet rows, model binaries,
or credentials. User-entered text is sent, so do not enter personal/confidential data.
`store=false` is not a claim of zero provider retention; provider/account data controls
still apply.

The LLM chooses tools and cites observed evidence; server code renders the numerical
results. It cannot execute code, modify files, retrain, or make individual decisions.
The existing eligibility/refusal checks and evidence-ID validation remain in place.
The API key is server-only and is never returned in status or bundled into React.

## Verification status

The adapter and agent loop can be tested with controlled responses using:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
bash scripts/check.sh
```

Tests clear inference configuration automatically to avoid accidental paid calls.
A live OpenAI call has **not** been verified at implementation time because no
`OPENAI_API_KEY` was available. Configure the key and run the explicit check before
claiming that the external integration works. No App publication/deployment is implied.

Official references:
- [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [API authentication](https://developers.openai.com/api/reference/overview)

Local verification on 2026-11-05: 108 tests passed; TypeScript, isolated production
build, and whitespace checks passed. The local frontend bundle was rebuilt. Existing
scikit-learn preprocessing-wrapper/dateutil deprecation warnings and the Vite bundle-size
advisory remain; they are not live inference failures. One existing leakage test was
updated to inspect the wrapper's inner fitted transformer without weakening its assertions.
No production model was retrained, Dataset artifact changed, commit pushed, or App deployed.

## Domino tracing and deployment handoff

Each configured agent request is wrapped in Domino's
`@add_tracing(name="hmda_analyst_request", span_type="AGENT")`. It records the
final server-rendered answer and evidence IDs/artifact names. Each HTTP provider
round is a child `LLM` span, and each validated internal tool execution is a child
`TOOL` span named `overview`, `trends`, `model`, `fairness`, or `definitions`.
Tool spans record the year, evidence ID, status and repository-relative artifact
names only. No raw question, LLM request/response payload, authorization header,
or full aggregate tool payload is recorded. The final aggregate answer and its
citations are intentionally attached to the parent span. Domino's Agent guidance
uses `@add_tracing` for production requests; `DominoAgentContext` is for Agent
development/evaluation Jobs that create versioned Agent runs. Do not wrap
production requests in `DominoRun` or call `mlflow.set_experiment()`. The existing
provider uses `requests`, so OpenAI SDK autologging would not observe its HTTP
calls; each selection round is traced explicitly. Tracing is optional at runtime
and failures do not break a working analytics request.

For the Agent Compute Environment, install the pinned trace dependencies
alongside the backend dependencies:

```bash
pip install -r backend/requirements.txt -r backend/tracing-requirements.txt
```

The Domino tracing setup guide specifies MLflow 3.2.0 and
`dominodatalab[data,agents]==2.2.0`; those are pinned separately so the existing
training workflow can retain its independently verified MLflow 3.3.2 pin. This
Workspace has MLflow 3.3.2 and Domino SDK 2.2.0 installed; a synthetic local trace
produced an `AGENT` root with a `fairness` `TOOL` child. That verifies local API
compatibility only, not the deployed Agent Performance destination.

### Verified Domino records

The read-only Apps API showed two separate deployments in this project:

- `GPT agent` (`AISYSTEM`, version 1) points to `backend/analyst_agent.py`.
- `HMDA Mortgage Intelligence` (`STANDARD`, version 2) points to `./preview.sh`.

The GPT agent instance is **Failed**. Its runtime log confirms the launch command
`./backend/analyst_agent.py` and the failure:
`ModuleNotFoundError: No module named 'backend'` at its import of
`backend.analyst`. The module is an internal helper, requires a `DataService` and
`Question`, and has no standalone server/CLI entry point. This confirms the
selected file is not a valid deployment entry point; the import error is the
immediate failure cause.

Domino's Agent documentation says production Agents are deployed as web apps and
the Agent deployment wizard takes a version entrypoint. For this project's
dedicated `AISYSTEM` Agent, set the version's `content.entryScript` to
`./preview.sh`. It prepares validated fairness/evaluation reports from the
read-only HMDA Dataset, then execs `./app.sh`, which starts
`python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8888}"`. Keep
the existing Python 3.12 environment, Dataset mount and `small-k8s` tier; the
environment revision is healthy and the Workspace reports the pinned Domino
tracing SDK available. This creates a new Agent version and preserves failed v1
for rollback. The separate `STANDARD` App record remains on `./preview.sh`; do
not change or redeploy it for this task.

The correct documented Apps beta instance-log route returned the failure above.
No commit, push, Agent update, or deployed Performance trace was performed; the
tracing code is currently only in the Workspace. The standard App was not
updated. A real visible deployed Agent trace remains the release gate.

### Demo prompts and expected evidence

For the normal outcome path, ask:

> Show aggregate outcomes for 2025.

Expected internal tool: `overview`; expected citation artifact: `hmda_2025.parquet`.

For the fairness trace, ask:

> For the 2025 cohort, report the saved model's denial recall and predicted-positive rate by race, summarize historical outcome rates by ethnicity and sex, state that ethnicity/sex model error rates are unavailable, and cite the fairness evidence.

Expected internal tool: `fairness`; the result reads 2025 aggregate outcomes across
race, ethnicity and sex, plus the validated race-only saved-model report. Expected
evidence ID is `E1` for the first read; its trace citation names
`hmda_2025.parquet` and `gradient_boosting_fairness_audit_2025.csv`. The existing
model artifact does not provide ethnicity/sex prediction-error metrics; do not
describe those outcome-only groups as model fairness/error results.

After updating the dedicated Agent from the traced revision, open
**Domino → Deployments → Apps & Agents → GPT agent → Performance**, select the latest request trace, expand
`hmda_analyst_request`, then expand `fairness`. The tool span should show its
`TOOL` type, `year: 2025`, `evidence_id: E1`, success status and both artifact names;
the parent output should show the deterministic answer and matching citation.
Workspace checks and a synthetic local trace do not replace this deployed verification.
