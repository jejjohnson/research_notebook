# Notebooks

This directory holds the showcase notebooks rendered into the MyST docs site.

## Layout

Notebooks are organized by the source repo or topic they showcase:

```
notebooks/
├── <source-or-topic>/
│   ├── <name>.ipynb        # executed notebook, outputs embedded
│   └── <name>.md           # optional prose write-up / landing page
└── README.md               # this file
```

## Authoring workflow

1. Open JupyterLab: `pixi run -e jupyterlab lab`.
2. Author the notebook under `notebooks/<source-or-topic>/<name>.ipynb`.
3. Execute all cells end-to-end so outputs (figures, prints, tables) embed
   into the `.ipynb` — the committed file is both the source and the
   rendered output.
4. Optionally add `notebooks/<source-or-topic>/<name>.md` alongside it for
   longer prose commentary that links back to the notebook.
5. Commit. MyST picks up both `.ipynb` and `.md` via the `Notebooks`
   section pattern in `myst.yml` — no TOC edit needed.

Figures render inline via the notebook's own outputs; do not save separate
PNGs or reference a `docs/images/` directory.
