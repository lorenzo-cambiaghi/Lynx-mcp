# Roadmap

This is a list of things we're considering, not commitments or a timeline.
Priorities follow real usage and feedback: items here may change, ship
later, or be dropped. Lynx works fully today without any of them. Issues and
PRs that make a case for (or against) these are welcome.

## Under evaluation

### Vector store: evaluate LanceDB as the local backend
ChromaDB serves Lynx well today, but we've hit cross-version index-format
fragility: a version-incompatible index could blank the dashboard or crash
the process, which is why Lynx now ships an out-of-process integrity probe and
`lynx reset`. [LanceDB](https://lancedb.com) is embedded, file-based, no
server, Rust, with a stable versioned on-disk format and more headroom for
large indexes. That is a better fit for the local-first design.

- Qdrant and the others run as a separate server or daemon. Adding
  infrastructure cuts against the "one command, 100% local, no services"
  promise. Embedded and file-based is the bar.
- It is not done yet because it is a real migration: a vector-store backend
  rewrite, a reindex for existing users, and retesting. Worth it only if
  Chroma's stability or scale becomes a recurring pain in practice, not as a
  speculative rewrite.

### Embeddings: an optional "quality" preset
The default stays `bge-small-en-v1.5` (384-dim, ~130 MB, fast on CPU).
That is what keeps the install light and the on-save re-index near-instant
without a GPU. For users with more RAM or compute who want higher ranking
quality, we're considering documenting a vetted step-up preset
(e.g. `bge-base-en-v1.5` or `bge-m3`).

- The embedding model is already swappable via `embedding.model_name`,
  as long as the repo ships the ONNX export Lynx runs (`onnx/model.onnx`;
  the BGE family does, `nomic-embed-text-v1.5` does not in its main repo).
  This would only recommend an alternative and document the trade-offs.
- Changing the model means a full reindex (different vector
  space), and bigger models are slower on CPU and produce larger indexes.
- Lynx is hybrid (dense + BM25 + RRF), so the dense model doesn't
  carry retrieval alone. The marginal gain from a larger embedder is smaller
  here than in a dense-only system.

## On the radar (additive, lower-risk)

### Coral community source: merged, the graph functions are next
Lynx is an official community source in Coral's
[`sources/community/`](https://github.com/withcoral/coral/tree/main/sources/community/lynx)
directory (PR [#1297](https://github.com/withcoral/coral/pull/1297), merged
June 2026): `coral source add --file sources/community/lynx/manifest.yaml`
gives Coral users `lynx.search` without leaving the registry. The manifest
kept in this repo ([`integrations/coral/manifest.yaml`](integrations/coral/manifest.yaml))
goes further, with six graph functions (`lynx.callers`, `callees`, `subclasses`,
`superclasses`, `imports`, `neighbors`) over `/api/v1/graph`. A
follow-up PR bringing them to the registry is staged, not yet opened.

### More language parsers
Lynx ships tree-sitter grammars for 18+ languages today (Python, TypeScript,
JavaScript, C#, C, C++, Go, Rust, Java, Ruby, PHP, Kotlin, Swift, Bash, SQL,
Scala, Lua, Objective-C). Adding more is cheap and additive: each grammar is a small wheel
bundled in the install, no runtime download. Demand decides what comes next
(Elixir, Zig and HTML/CSS are candidates); open an issue with the language you need and it
moves up. A few (notably Dart and R) are on hold: their community tree-sitter
grammars don't currently expose a usable parser for our chunker, so they wait
until that changes upstream.

## Non-goals

Some things we don't plan to do:

- No hosted or cloud version, no telemetry, no code upload. "100% local"
  is the point, not a phase. (The self-host Docker image runs *your* server,
  on *your* machine; see the `Dockerfile`.)
- No per-source tool explosion. The MCP tool surface stays fixed and small,
  so your client's context window doesn't grow with the number of sources.
