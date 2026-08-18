#!/usr/bin/env bash
# Regenerate the SVG diagrams for the model-flywheel proposal.
#
#   ./render.sh
#
# Sources are the .mmd files beside this script. Edit those, not the SVGs.
# Requires network access on first run (npx fetches mermaid-cli).
#
# Why pre-rendered SVG instead of ```mermaid``` fences: these diagrams contain
# cycles (the flywheel loops back on itself). Mermaid's default dagre layout
# breaks cycles by reversing an edge, which reorders the ranks and puts the
# last stage above the first. The ELK layout engine handles cycles correctly,
# but is not bundled with the Docusaurus mermaid theme -- so we render with
# ELK here and commit the result.
set -euo pipefail

cd "$(dirname "$0")"
MMDC="npx --yes @mermaid-js/mermaid-cli@11"

for src in *.mmd; do
  out="${src%.mmd}.svg"
  echo "rendering $src -> $out"
  $MMDC -c mermaid-config.json -i "$src" -o "$out" -b white --quiet

  # mermaid-cli emits width="100%" together with an inline
  # `max-width: <natural>px`, which pins the figure to its natural width and
  # leaves it small in a wide doc column. Drop the cap so the viewBox scales
  # the diagram up to the container.
  python3 - "$out" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'max-width:\s*[0-9.]+px;\s*', '', s, count=1)
open(p, 'w').write(s)
PY
done

echo "done"
