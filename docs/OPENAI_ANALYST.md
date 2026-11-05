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
