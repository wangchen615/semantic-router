# Model flywheel proposal diagrams

Figures for
[`website/docs/proposals/model-flywheel-distillation-and-router-finetuning.md`](../../../../docs/proposals/model-flywheel-distillation-and-router-finetuning.md).

- `flywheel.mmd` — the six-stage data-and-model flywheel
- `architecture.mmd` — data path versus control plane
- `campaign.mmd` — campaign rounds and gates
- `mermaid-config.json` — shared theme, spacing, and text wrapping width

Edit the `.mmd` sources, then regenerate:

```bash
./render.sh
```

The SVGs are build output. Do not hand-edit them.

## Why pre-rendered instead of inline fences

**Layout.** `flywheel.mmd` is a cycle. Mermaid's default dagre layout breaks a cycle by
reversing an edge, which reorders ranks and inverts the intended reading order. The ELK
layout engine handles it correctly, but is not bundled with the Docusaurus Mermaid theme
— hence rendering here. ELK is not a universal fix: `campaign.mmd` still came out
inverted with a back-edge, so it states its recursion in the final node's label rather
than drawing it. `architecture.mmd` sidesteps the issue by unrolling one turn of the loop
into alternating data-path and control-plane passes, which is a DAG.

**Size.** Each figure's natural width is set by its longest unbroken label line, capped
by `wrappingWidth` in `mermaid-config.json`. Note that explicit `<br/>` breaks win over
`wrappingWidth`: if every line is already shorter than the cap, raising the cap does
nothing and you must lengthen the lines instead. All three are tuned to roughly 800px so
they render near 1:1 in the documentation column. A figure much wider than the column is
scaled down until its labels are unreadable; one much narrower is scaled up until its
labels dwarf the body text.

`render.sh` also strips the inline `max-width` that `mermaid-cli` emits next to
`width="100%"`, which would otherwise pin each figure to its natural width.
