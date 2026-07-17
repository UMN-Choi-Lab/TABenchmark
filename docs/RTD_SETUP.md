---
orphan: true
---

# Connecting TABenchmark to Read the Docs (PI-only, one time)

Everything else about the documentation site is automated by
[`.readthedocs.yaml`](https://github.com/UMN-Choi-Lab/TABenchmark/blob/main/.readthedocs.yaml)
and the `docs` CI job. This one step needs a person with **admin on
`UMN-Choi-Lab/TABenchmark`** and is the only manual action to bring the site live.

**Merge the config to `main` first.** The importer triggers the first build the
moment the project is added, so `.readthedocs.yaml` (and the `docs` extra) must
already be on `main`.

1. Go to <https://app.readthedocs.org/> and **Log in with GitHub** (the account
   needs admin on the repo).
2. Dashboard → **Add project** → type `TABenchmark` and click the repo in the list.
   - If it does not appear, install the **Read the Docs** GitHub App when prompted:
     on GitHub choose the `UMN-Choi-Lab` org → *Only select repositories* →
     `TABenchmark` → **Install** (an org owner must approve if you are not one).
     RTD uses a GitHub App now, so there is **no manual webhook** to configure.
3. Click **Continue** → in the pre-filled fields set **Name** to `tabenchmark`
   (this becomes the slug and URL — `https://tabenchmark.readthedocs.io/` — and it
   was free as of 2026-07-16; confirm it still shows available). Default branch
   `main` → **Next** → at the configuration-file step click **This file exists**.
   The first build starts automatically.
4. Verify pushes build: push any commit to `main` and watch the build appear under
   **Builds**. (No webhook step — the GitHub App handles it.)
5. **Enable PR previews:** project **Settings → Pull request builds** → tick
   *Build pull requests for this project* → **Update**. A
   `docs/readthedocs.org:tabenchmark` status then appears on PRs, serving the
   preview from `org.readthedocs.build`. Leave it **non-required** — the `docs` CI
   job is the merge gate, so merges never wait on RTD Community's 2-build
   concurrency. PRs opened *before* enabling need one new commit to get a preview.
6. **Versions:** **Admin → Versions** — keep `latest` (tracks `main`) as the
   default. Do **not** activate `stable` yet: the repo is 0.x with a weekly-moving
   roster, so a frozen `stable` would only make the landing page stale. At the first
   public release tag (paper / v1.0), push the semver tag, activate the `stable`
   version RTD materializes, and set it as the default.

**Watch the first build's wall time.** The RTD Community tier caps a build at 15
minutes (7 GB RAM, 2 concurrent builds). The cold build measured **~3.3 min
locally** (mirror assembly + 47 executed notebooks); on an RTD builder — at the
plan's ~3× slowdown, plus the `.[docs,viz]` install and the ~131 MB data prefetch —
expect roughly **~10 min**, i.e. inside the 15-min cap but not by a wide margin. If a
future tree pushes past it, the pressure valve is `nb_execution_excludepatterns` in
`docs/conf.py`: exclude the slowest notebooks from execution (they still render
un-executed) rather than dropping them.

Because RTD cannot be built without the PI's account, this is also where the config's
`sphinx:`-key requirement (see `.readthedocs.yaml`) gets its first real end-to-end
check: confirm the first build runs the `.[docs,viz]` install and the two-step
`build.jobs.build.html` — if the install is skipped, the doctype inference regressed.
