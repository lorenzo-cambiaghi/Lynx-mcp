# steampipe-plugin-lynx

A [Steampipe](https://steampipe.io) plugin that exposes a local
[Lynx](https://github.com/lorenzo-cambiaghi/LynxMCP) instance as SQL tables:
semantic code search and the code knowledge graph, joinable with Steampipe's
connectors (GitHub, Jira, AWS, …).

It is a thin SQL skin over Lynx's stable `/api/v1` HTTP API. The design
(tables, columns, qual to query-param mapping) is in
[`../DESIGN.md`](../DESIGN.md).

> Platform note: the Steampipe engine runs on macOS and Linux (or WSL2 on
> Windows). There is no native Windows build, so on Windows run it inside
> WSL2. You can still write and compile the plugin on any OS
> (`go build ./...`); only running it needs the engine.

## Install (prebuilt — no Go toolchain)

Grab the build for your platform from the
[latest release](https://github.com/lorenzo-cambiaghi/LynxMCP/releases?q=steampipe)
(assets named `steampipe-plugin-lynx_<version>_<os>_<arch>.tar.gz`):

```bash
mkdir -p ~/.steampipe/plugins/local/lynx ~/.steampipe/config
tar xzf steampipe-plugin-lynx_*_<os>_<arch>.tar.gz
mv lynx.plugin ~/.steampipe/plugins/local/lynx/lynx.plugin
cp lynx.spc  ~/.steampipe/config/lynx.spc       # only if you don't have one yet
```

Then start the Lynx backend and query (see [Query](#query) below). Optionally
verify the download against the release's `SHA256SUMS`.

To build from source instead, see
[Develop / build / test](#develop--build--test).

## Tables

| Table | Endpoint | Required quals |
|---|---|---|
| `lynx_source` | `GET /api/v1/sources` | — |
| `lynx_search` | `GET /api/v1/search` | `query` |
| `lynx_graph`  | `GET /api/v1/graph`  | `operation`, `symbol` |

## Why Steampipe (vs the Coral source)

Steampipe pushes WHERE quals down and runs a nested loop in joins, so it
calls the Lynx API once per qual value. `lynx_search` can therefore be driven
by another table's column:

```sql
select t.key, h.file, h.symbol, h.score
from jira_issue t
join lynx_search h on h.query = t.summary   -- one Lynx call per issue
where t.status = 'Open';
```

(Coral resolves table-function args at plan time, so there you batch instead.
The cost here: N joined rows means N embedding passes. For large N, prefer
Lynx's `POST /api/v1/search` batch endpoint from a script.)

## Develop / build / test

Writing and compile-checking work anywhere, Windows included. Running needs the
Steampipe engine, so macOS, Linux, or WSL2. `go.sum` is committed, so
`go mod tidy` is only needed when you change dependencies.

```bash
# compile-check (Windows-friendly, no engine)
make build      # or: go build ./...
go vet ./...

# build + install as a local plugin (macOS/Linux/WSL2)
make install    # -> ~/.steampipe/plugins/local/lynx/lynx.plugin

# config: copy config/lynx.spc -> ~/.steampipe/config/lynx.spc
cp config/lynx.spc ~/.steampipe/config/lynx.spc
```

## Query

Start the backend first (`lynx manager ui --port 8765 --no-browser`).

First, a smoke test that works on any index. It lists the sources you have
indexed and confirms the plugin can reach the Lynx API:

```bash
steampipe query "select name, type, chunk_count from lynx_source;"
```

The search and graph queries need values from your own codebase, so the
examples below are placeholders. Substitute a real `source` name (from the
smoke test above) and symbols or behaviors that exist in your code. The
literals `framework`, `camera zoom` and `ApplyDamage` come from an example
source index that isn't included, so they won't match yours:

```bash
# semantic + lexical search: describe the behavior in words.
# `source` and `top_k` are optional; omit `source` to search everything.
steampipe query "select file, symbol, score from lynx_search where query = '<what the code does>' and source = '<your-source>' and top_k = 10;"

# code graph: who calls a symbol that exists in your code?
steampipe query "select from_symbol, from_file from lynx_graph where operation = 'callers' and symbol = '<YourSymbol>';"
```

### Per-row joins

Steampipe pushes `WHERE` quals down and runs a nested loop in joins, so
`lynx_search` can be driven by another table's column, one search per row.
That is the per-row fan-out the plan-time engines (Coral, DuckDB) cannot do
without a batch helper:

```sql
-- for each of your open GitHub issues, find the code that matches its title
SELECT i.number, i.title, s.file, s.symbol, s.score
FROM github_my_issue i
JOIN lynx_search s ON s.query = i.title
WHERE i.state = 'open' AND s.source = 'app'
ORDER BY i.number, s.score DESC;
```

## CI & releasing

Two GitHub Actions workflows cover this plugin:

- `steampipe-plugin-ci.yml`: on every push or PR that touches
  `integrations/steampipe/**`, it runs `go build`, `go vet`, a cross-compile
  sanity check for all release targets, and a load-validation job that installs
  the Steampipe engine and asserts it accepts the plugin. That last check
  catches bugs invisible to `go build` (e.g. a key column without a matching
  column), which is the safety net when developing on Windows.
- `steampipe-plugin-release.yml`: cross-compiles macOS/Linux × amd64/arm64,
  packages each as a `.tar.gz` (binary + `lynx.spc` + this README) plus a
  `SHA256SUMS`, and publishes them to a GitHub Release.

Cut a release by pushing a dedicated tag (independent of the Lynx app's `v*`
tags):

```bash
git tag steampipe-v0.1.0
git push origin steampipe-v0.1.0
```

## Connection config

See [`config/lynx.spc`](config/lynx.spc). `api_url` is optional and falls back to
the `LYNX_API` env var, then `http://127.0.0.1:8765`.

## Layout

```
main.go                       plugin.Serve(PluginFunc: lynx.Plugin)
lynx/plugin.go                Plugin(): TableMap + config schema + default transform
lynx/connection_config.go     lynxConfig{ api_url } + GetConfig
lynx/client.go                HTTP client: GET /api/v1/... ?format=ndjson, stream rows
lynx/table_lynx_source.go     lynx_source
lynx/table_lynx_search.go     lynx_search  (query required; source, top_k optional)
lynx/table_lynx_graph.go      lynx_graph   (operation, symbol required; ...)
```

## License

Apache-2.0, to match Lynx.
