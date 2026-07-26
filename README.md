# sam4xtal

Point-prompt SAM3 segmentation for SEM crystal micrographs.

Stack:

- **Next.js + shadcn** UI — load images, click crystals, measure size, save annotations
- **Python sidecar** — local SAM3 inference HTTP API aligned with [Roboflow Inference SAM3](https://inference.roboflow.com/foundation/sam3/)
- **Vercel eve stub** — inert `agent/` scaffold for future agent workflows

## Quick start

### 1. Python sidecar (port 9001)

```bash
cd sidecar
# macOS/Linux
chmod +x run.sh && ./run.sh          # real SAM (transformers + CUDA)
./run.sh --mock                      # flood-fill stub
# Windows
.\run.ps1                            # real SAM
.\run.ps1 --mock                     # stub
```

Default is SAM 3 Tracker via Hugging Face (`facebook/sam3`, point/box PVS). Pass `--mock` for a no-weights flood-fill stub. The model is gated — accept the license on Hugging Face and set `HF_TOKEN` in `sidecar/.env` if download fails.

### 2. Next.js app

```bash
cd web
cp .env.example .env.local
pnpm install --config.minimumReleaseAge=0   # if packages are newer than the age policy
pnpm approve-builds --all                   # allow sharp / native postinstalls
pnpm dev
```

Open http://localhost:3000

### Sample SEM images

`samples/crystals/` has single-panel SEM crops cut from arXiv `2607.07877v1` for testing the UI. Full multi-panel figures are in `samples/figures/`. Good starters: `fig02_A`, `fig03_A`, `fig04_A`, `fig04_D`.

### 3. Workflow

1. Choose image files
2. Optionally enter SEM resolution in **nm / px**
3. Click the crystal (positive points); use Negative for refinements
4. **Run segmentation**
5. Review size in px (and nm when resolution is set)
6. **Save annotation** → downloads `<image>.mask.json` (metadata) and `<image>.mask.png` (0/255 binary mask)

## Swapping the sidecar for Roboflow

The Next.js proxy at `/api/sam3/*` forwards to whatever `INFERENCE_URL` points at, using the same paths as Roboflow:

| Local sidecar | Roboflow-compatible path |
| --- | --- |
| `POST /sam3/embed_image` | same |
| `POST /sam3/visual_segment` | same |
| `POST /sam3/concept_segment` | same |

To use Roboflow Serverless instead of the local sidecar:

```bash
# web/.env.local
INFERENCE_URL=https://serverless.roboflow.com
ROBOFLOW_API_KEY=your_key
```

Or point at a self-hosted Inference server:

```bash
INFERENCE_URL=http://localhost:9001
```

Request body for interactive clicks (matches Roboflow PVS):

```json
{
  "image": { "type": "base64", "value": "..." },
  "prompts": [
    { "points": [{ "x": 320, "y": 240, "positive": true }] }
  ],
  "multimask_output": false,
  "format": "json"
}
```

## Eve agent (stub)

`web/agent/` contains a minimal eve agent that does nothing useful yet:

- `agent/agent.ts`
- `agent/instructions.md`
- `agent/tools/noop.ts`

## Repo layout

```
sam4xtal/
├── README.md
├── samples/          # SEM figures + crystal panel crops for testing
├── sidecar/          # FastAPI SAM3 sidecar
│   ├── app/
│   ├── pyproject.toml
│   ├── run.sh
│   └── run.ps1
└── web/              # Next.js + shadcn + eve stub
    ├── agent/
    ├── src/
    └── package.json
```
