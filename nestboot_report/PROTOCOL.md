# Protocol: NestBoot on ENCODE K562 (chapter section 3.5, Figures 5 and 6)

This is the cut-and-paste protocol for the K562 NestBoot section. It uses the
100 outer × 100 inner LSCO run that the chapter numbers come from. A bare
`uv run pygs nestboot` is a different, smaller procedure: it does not fill
dropouts, does not drop failed knockdowns, and does not sweep the zeta grid
this section reports. Do not substitute it for the commands below.

Run every command from the pyGS repository root.

## What this section does

NestBoot (Morgan et al., Bioinformatics 35:1026, 2019) asks, at each sparsity,
how often a link reappears across bootstrap fits of the measured data, and
compares that with the same fits on gene-shuffled data. The FDR at a support
cutoff B is the shuffled tail divided by the measured tail. The network keeps
links whose support reaches that cutoff in every outer run.

On this table the FDR rule is met at many sparsities. Zeta, the threshold on
|A|, is what chooses the network. Figure 5 shows that choice at one sparsity
(zeta 0.158): link frequency and the overlap of the outer runs, for measured
and shuffled links. Figure 6 draws one GRN from a nearby sparsity (zeta 0.063),
thinned to about three links per gene.

## Inputs

| File | What it is |
|---|---|
| `ymatrix_k562.csv` | ENCODE K562 RNA-binding-protein knockdowns. 232 genes × 464 samples. Rows are genes. Column names are the knocked-down gene, two replicates each. Values are log-fold-change-like (median −0.18). |
| Perturbation design P | Not a separate file. A column named `GENE` or `GENE.1` is a knockdown of that gene. P is genes × samples, with −1 in the knocked-down gene. |

Source, pinned so the table cannot move:

```
https://bitbucket.org/sonnhammergrni/genesnake_publication_reproduction/raw/e43daafc97608925acd17225d473aaa080c29798/figure_4/encode_data/ENCODE/ymatrix_k562.csv
```

Three problems in that table are fixed before inference. They change the GRN.

1. **Dropouts.** 44 values sit between −12.5 and −19.6. Every other value is at least −5. They are filled from the other replicate of the same knockdown, or from the gene median if both replicates are dropouts (`--dropout-below -10`).
2. **Failed knockdowns.** 14 genes did not fall in their own knockdown (own-gene robust z > 0): AGO2, AKAP8, AQR, CKAP4, DKC1, EIF3G, FUBP3, HNRNPU, PES1, PKM, PRPF8, TAF15, TBRG4, YBX3. Those 28 samples are removed. The genes stay in the network as possible regulators but receive no incoming links (`--max-knockdown-z 0`).
3. **Knockdown sign.** The loader must code P = −1. LSCO fits A = −P · pinv(Y). A +1 design flips the sign of every link.

Gene scaling (`--scale-genes`) is a sensitivity check only. The chapter network does not use it. Scaling reverses which genes become hubs.

## Outputs

The 100 × 100 run writes one directory, `nestboot_report/runs_100x100/`.

| File | What it is |
|---|---|
| `real.npy` | Binary GRN at each zeta. Shape genes × genes × 41. Entry (i, j, k) is 1 when regulator j → target i is kept at zeta k (support at or above the FDR cutoff in every outer run). |
| `real_signed.npy` | Same shape, with the sign of the link. |
| `real_support.npz` | The data behind Figures 5 and 6: `zetavec` (41 values from 10⁻⁴ to 1), `support_cutoff`, `fdr_at_cutoff`, `mean_support`, `min_support`, `run_histograms` (outer run × zeta × support bin × measured/shuffled), gene names, knockdown z, and which knockdowns were removed. |
| `real_support.tsv` | One row per zeta and support bin: link counts, overlap of outer runs, and the shuffled/measured tail ratio. This is the table Figure 5 is drawn from. |
| `real.txt` | Short text summary of the run. |

Figure files, written by the two commands at the end:

| File | What it is |
|---|---|
| `figures/fig5_support_overlap.png` and `.pdf` | Figure 5. At zeta 0.158: link frequency (fraction of gene pairs in each support bin) and overlap (links present in every outer run / links present in any outer run), measured and shuffled. |
| `grn/grn_zeta0.0631_3pergene_edges.csv` | Figure 6 edge table: regulator, target, sign, mean support, minimum support. |
| `grn/grn_zeta0.0631_3pergene_nodes.csv` | Figure 6 node table: degree, knockdown z, gene SD. |
| A Graphistry URL | Printed only if Graphistry credentials are set. The CSV tables are the figure data either way. |

## 1. Install

```bash
git clone https://github.com/dcolinmorgan/pyGS.git
cd pyGS
uv sync
```

## 2. Fetch the expression table

```bash
mkdir -p .gs_cache/downloads
curl -fsSL -o .gs_cache/downloads/134c84bebbd65203.csv \
  "https://bitbucket.org/sonnhammergrni/genesnake_publication_reproduction/raw/e43daafc97608925acd17225d473aaa080c29798/figure_4/encode_data/ENCODE/ymatrix_k562.csv"
```

Check the shape and the knockdown design. This does not fit a network.

```bash
uv run python - << 'PY'
from bio.io import load_inference_inputs
from bio.knockdown import fill_dropouts, knockdown_z, drop_failed_knockdowns

X, genes, P = load_inference_inputs(".gs_cache/downloads/134c84bebbd65203.csv")
Y, design = X.T, P.T          # genes × samples
print(f"{len(genes)} genes × {Y.shape[1]} samples")
Y2, mask = fill_dropouts(Y, design, below=-10)
print(f"dropouts filled: {int(mask.sum())}")
z = knockdown_z(Y2, design)
failed = sorted(genes[i] for i in range(len(genes)) if z[i] > 0)
print(f"failed knockdowns ({len(failed)}): {', '.join(failed)}")
Y3, P3, _ = drop_failed_knockdowns(Y2, design, max_z=0)
print(f"after removing failed knockdown samples: {Y3.shape[1]} samples")
PY
```

Expect 232 genes × 464 samples, 44 dropouts, 14 failed knockdowns, and 436 samples left.

## 3. The NestBoot run behind the chapter

The chapter numbers are the finished run in `nestboot_report/runs_100x100/`.
Regenerating it is:

```bash
bash nestboot_report/run_100x100.sh
```

which is this command:

```bash
uv run pygs nestboot .gs_cache/downloads/134c84bebbd65203.csv \
  -m lsco -i 100 -o 100 \
  --zeta=-4:0:41 --seed 42 \
  --dropout-below -10 --max-knockdown-z 0 \
  --output nestboot_report/runs_100x100/real.npy
```

What each flag does:

| Flag | Meaning |
|---|---|
| `-m lsco` | A = −P · pinv(Y), then keep links with \|A\| > zeta. Self-links are dropped. |
| `-o 100` | Outer runs. Each one is a fresh set of inner bootstraps. |
| `-i 100` | Inner bootstraps of Y, and the same number on one gene-shuffled copy of Y. Samples are redrawn inside each knockdown's replicates. |
| `--zeta=-4:0:41` | 41 thresholds, log-spaced from 10⁻⁴ to 1, on \|A\| after Y is rescaled to [−1, 1] as in GeneSPIDER `lsco.m`. |
| `--dropout-below -10` | Fill the 44 dropout values before any fit. |
| `--max-knockdown-z 0` | Drop the 14 knockdowns that did not lower their own gene. |
| `--seed 42` | Bootstrap seed. |

FDR is 0.05, the default. A link's support in one outer run is the fraction of the 100 inner fits that contain it. The cutoff is the lowest support whose shuffled tail is at most 5% of the measured tail. The saved network keeps a link only when every outer run reaches that cutoff.

This is a long run (the deposited one finished as a full 100 × 100). The files listed above are the result. If `real_support.tsv` is already present, skip this step and draw the figures from it.

## 4. Figure 5: support, frequency, and overlap

Figure 5 is one sparsity, zeta 0.158, in the layout of Figure 2 of Morgan et al. (2019). It reads `real.npy` and `real_support.npz`. It does not refit LSCO.

```bash
uv run python nestboot_report/make_figures.py \
  --run nestboot_report/runs_100x100/real.npy \
  --zeta 0.158
```

This also writes Figures 1–4 and the three-link-per-gene panel. Figure 5 is `nestboot_report/figures/fig5_support_overlap.png`.

Read the same numbers from the table. Support 0 is omitted from the frequency curve because it is the mass of links that never passed zeta.

```bash
uv run python - << 'PY'
import pandas as pd
df = pd.read_csv("nestboot_report/runs_100x100/real_support.tsv", sep="\t")
# one row per zeta: the FDR cutoff is where the tail ratio is stored
cut = (df.sort_values("support_from")
         .groupby("zeta", as_index=False)
         .tail(1))
row = cut.iloc[(cut.zeta - 0.158).abs().argmin()]
print(f"zeta {row.zeta:.3f}")
print(f"support cutoff {row.support_from:.2f}")
print(f"shuffled/measured tail {row.tail_area_ratio:.3f}")
print(f"measured overlap at the cutoff {row.measured_overlap:.3f}")
print(f"shuffled overlap at the cutoff {row.shuffled_overlap:.3f}")
part = df[df.zeta == row.zeta]
print(f"measured links in the top bin {part.measured_links.max():.0f}")
print(f"shuffled links in the top bin {part.shuffled_links.max():.0f}")
PY
```

At zeta 0.158 the deposited run keeps 4,491 links, the support cutoff is 0.59, and the shuffled/measured ratio at that cutoff is 0.049. Shuffled links are concentrated at low support. Measured overlap is the fraction of links seen in any outer run that are seen in every outer run.

## 5. Figure 6: the GRN

Figure 6 is the network at zeta 0.063 (1,222 links, 5.3 per gene, each link in at least 94 of 100 bootstraps in every outer run, FDR 4.7% at the cutoff), thinned to the highest-support links until the density is nearest 3 links per connected gene. The script writes the edge and node tables either way. The Graphistry upload is optional.

```bash
uv run --with graphistry python nestboot_report/plot_graphistry.py \
  --run nestboot_report/runs_100x100/real.npy \
  --zeta 0.063 --per-gene 3 --no-plot
```

Tables:

```
nestboot_report/grn/grn_zeta0.0631_3pergene_edges.csv
nestboot_report/grn/grn_zeta0.0631_3pergene_nodes.csv
```

`edges.csv` columns are `regulator`, `target`, `sign` (`activation` or `repression`), `mean_support`, and `min_support`. A row is the link regulator → target. `nodes.csv` has the degree of each gene, its knockdown z, and its standard deviation. Green edges are activation and red edges are repression when the plot is drawn.

To upload that graph to Graphistry and print the URL:

```bash
export GRAPHISTRY_USERNAME=you@example.com
# the password is prompted for, or set GRAPHISTRY_PASSWORD
uv run --with graphistry python nestboot_report/plot_graphistry.py \
  --run nestboot_report/runs_100x100/real.npy \
  --zeta 0.063 --per-gene 3
```

`GRAPHISTRY_PERSONAL_KEY_ID` and `GRAPHISTRY_PERSONAL_KEY_SECRET` work in place of the username. The server defaults to `hub.graphistry.com`.

A short description of the network the figure shows:

```bash
uv run python - << 'PY'
import pandas as pd
edges = pd.read_csv("nestboot_report/grn/grn_zeta0.0631_3pergene_edges.csv")
nodes = pd.read_csv("nestboot_report/grn/grn_zeta0.0631_3pergene_nodes.csv")
print(f"{len(edges)} links, {nodes.gene.nunique()} genes in the node table")
print(f"activation { (edges.sign == 'activation').mean():.0%}")
print("top regulators")
print(edges.groupby("regulator").size().sort_values(ascending=False).head(5))
print(f"median mean support {edges.mean_support.median():.2f}")
print(f"lowest min support {edges.min_support.min():.2f}")
PY
```

## What the two figures are for

Figure 5 is the reason a support cutoff exists. At zeta 0.158 the shuffled links do not recur across outer runs, and the measured links do, so the FDR cutoff (0.59) sits where the shuffled tail has fallen to 5% of the measured tail. Overlap is the stricter summary: a link has to be in every outer run, not merely above the cutoff on average.

Figure 6 is one network chosen from that family, not a second FDR. Zeta 0.063 is the FDR-controlled sparsity nearest 3–5 links per gene (1,222 links before thinning). Thinning to 3 links per gene keeps the highest-support links of that same fit. It does not rerun inference.
