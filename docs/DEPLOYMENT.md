# Deploying PaperPilot

This app is not a typical web service, and the difference decides where it
can be hosted. The backend holds a sentence-transformer, a spaCy pipeline
and PyTorch in memory for the life of the process. That rules out every
serverless-function platform and most of the 512 MB free tiers before you
write a line of config.

So the first section is measurements, not options.

---

## What it actually needs

Measured on the running backend, models loaded, after serving one real
search (`graph neural networks`, 60 papers across three sources):

| | |
|---|---|
| **Resident memory, idle after warm-up** | **513 MB** |
| **Peak during a search** | **865 MB** |
| Cold start to first healthy response | ~6 s (embedding model load) |
| Backend image, CPU-only torch | ~1.2 GB |
| Backend image if you forget `--index-url .../cpu` | ~2.5 GB |
| Frontend bundle | 236 KB JS + 35 KB CSS (73 KB + 8 KB gzipped) |

Reproduce the memory figure yourself — guessing at this is how you end up
debugging an OOM kill at 3am:

```bash
# Linux/macOS, with the backend running and warmed by one search
ps -o rss= -p $(pgrep -f 'uvicorn app.main') | awk '{printf "%.0f MB\n", $1/1024}'
```

**The practical floor is 1 GB of RAM.** 512 MB will start, serve `/health`,
and then get OOM-killed on the first real search — the worst possible
failure mode, because it looks like it deployed fine.

Two other constraints that matter:

- **Outbound HTTPS** to `eutils.ncbi.nlm.nih.gov`, `export.arxiv.org`,
  `api.crossref.org` and `api.mistral.ai`. Any host that whitelists
  outbound traffic (PythonAnywhere's free tier, for one) cannot run this.
- **Disk is optional.** The SQLite files under `/app/data` are a summary
  cache and a paper store. Losing them costs re-generation, not
  correctness, so ephemeral filesystems are fine — you just pay for
  summaries again after each deploy.

---

## Where to deploy free

Ranked by how well they fit *these* numbers, not by general popularity.

| Host | Free? | RAM | Verdict |
|---|---|---|---|
| **Hugging Face Spaces** | Yes, no card | **16 GB** | **Best fit.** Built for exactly this. |
| **Google Cloud Run** | Yes, within free tier | up to 32 GB | Best if you want real infrastructure. Card required. |
| **Oracle Cloud Always Free** | Yes, forever | **24 GB** | Most capable by far, and the most manual. [See below.](#option-d--oracle-cloud-always-free) |
| Fly.io | No — ~$2–4/mo | 1 GB | Cheapest genuinely-good paid option. |
| Render free | Yes | 512 MB | **Will OOM.** Below our 865 MB peak. |
| Vercel / Netlify functions | Yes | n/a | **Impossible.** No persistent process; torch exceeds the bundle limit. |
| PythonAnywhere free | Yes | 512 MB | **Impossible.** Whitelists outbound HTTP; PubMed and arXiv are blocked. |

The frontend is a static bundle and is free essentially everywhere —
Vercel, Netlify, Cloudflare Pages and GitHub Pages all work without
qualification.

### Recommendation

**Hugging Face Spaces for the backend, Vercel for the frontend.** No credit
card, 16 GB of RAM against a 865 MB peak, and Spaces exists precisely to
host models. The rest of this document assumes that pairing and then covers
Cloud Run for anyone who wants something less ML-flavoured on their CV.

---

## Option A — Hugging Face Spaces + Vercel

### 1. Backend on Spaces

Create a Space with **SDK: Docker**, then push this repo's `backend/`
directory to it. The Space needs a `README.md` at its root with this
frontmatter — the `app_port` line is what tells Spaces where to route:

```yaml
---
title: PaperPilot API
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---
```

Spaces sets `PORT=7860`. The image reads it:

```dockerfile
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

That line is load-bearing and was a bug until this guide was written — a
hard-coded `8000` means the platform health-checks a port nothing is
listening on, and the deploy fails with no useful error.

Then set these in **Settings → Variables and secrets**:

| Name | Value | Kind |
|---|---|---|
| `PAPERPILOT_CORS_ORIGINS` | `https://<your-app>.vercel.app` | Variable |
| `PAPERPILOT_ENVIRONMENT` | `production` | Variable |
| `PAPERPILOT_MISTRAL_API_KEY` | your key | **Secret** |
| `PAPERPILOT_CROSSREF_MAILTO` | your email | Variable |

`PAPERPILOT_CORS_ORIGINS` accepts a bare URL, a comma-separated list, or a
JSON array. Comma-separated is what you want for a preview plus production
origin:

```
https://paperpilot.vercel.app,https://paperpilot-git-main-you.vercel.app
```

> The first build takes 10–20 minutes. The image installs CPU-only torch
> and bakes both models in at build time, deliberately: the alternative is
> every cold start downloading 100 MB before it can answer.

### 2. Frontend on Vercel

Import the repo, then:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_BASE` = `https://<user>-<space>.hf.space` |

`VITE_API_BASE` is read at **build** time, not runtime
([`src/lib/api.ts`](../frontend/src/lib/api.ts)), so changing it needs a
redeploy, not just a restart. Left unset it defaults to `''`, which means
same-origin — correct for Docker Compose, wrong for split hosting, and the
symptom is every request 404ing against the Vercel domain.

### 3. Verify the deploy

Do not trust a green checkmark; check the things that actually break.

```bash
API=https://<user>-<space>.hf.space

curl -s $API/health | jq                      # three sources registered?
curl -s "$API/api/search?q=CRISPR&limit=3" | jq '.sources'   # upstream reachable?

# CORS is the one that silently fails in the browser only:
curl -si -X OPTIONS "$API/api/search" \
  -H "Origin: https://<your-app>.vercel.app" \
  -H "Access-Control-Request-Method: GET" | grep -i access-control-allow-origin
```

That last command is the whole deployment in one line. If it returns no
header, the frontend will load, look perfect, and fail every search with a
console error — which is exactly how this class of bug reaches production.

---

## Option B — Google Cloud Run

Better story for a backend-engineering CV: real autoscaling, scale-to-zero,
and a free tier that genuinely covers portfolio traffic (2M requests and
360k GiB-seconds per month).

```bash
gcloud run deploy paperpilot-api \
  --source backend \
  --region europe-west1 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --timeout 120 \
  --allow-unauthenticated \
  --set-env-vars "PAPERPILOT_CORS_ORIGINS=https://your-app.vercel.app" \
  --set-secrets "PAPERPILOT_MISTRAL_API_KEY=mistral-key:latest"
```

Four flags deserve comment:

- **`--memory 1Gi`** — from the 865 MB measurement above. 512Mi OOM-kills
  mid-search.
- **`--timeout 120`** — a cold search is a model load plus a three-source
  fan-out plus summarization. The 60 s default cuts it off.
- **`--min-instances 0`** — scale to zero is what keeps this free, and the
  price is a ~15 s cold start while the model loads. `--min-instances 1`
  removes the wait and leaves the free tier.
- **`--cpu 1`** — torch will happily use more, and you will pay for it.

Cloud Run injects `$PORT` (8080), which the image already honours.

---

## Option C — everything in one box

If you have any host that runs Docker Compose — a cheap VPS, a spare
machine, or the Oracle instance in Option D — this is the least
configuration of all:

```bash
git clone https://github.com/zainabraza06/PaperPilot && cd PaperPilot
printf 'PAPERPILOT_MISTRAL_API_KEY=%s\n' "$KEY" > .env
docker compose up --build -d
```

Both containers, one network, nginx proxying `/api` to the backend by
service name. **No `VITE_API_BASE` and no CORS setting are needed**, because
the app is same-origin — which is why the compose setup is the one that
works with zero configuration. Put a reverse proxy with TLS in front
(Caddy is two lines) and it is done.

Option D is this, on hardware that costs nothing.

---

## Option D — Oracle Cloud Always Free

On the numbers this is the best free option that exists, and it is the only
one on the list with no expiry, no sleep and no cold start. It is also the
only one where you are the sysadmin.

### What the Always Free tier gives you

The part that matters is the **Ampere A1 (ARM64)** allowance: **4 OCPUs and
24 GB of RAM**, which you can pour into a single VM or split across up to
four. Alongside it: 200 GB of block storage and 10 TB/month of egress.
Always Free means always — it is not a 12-month trial, though Oracle also
gives you trial credits on top at signup.

Against this app's 865 MB peak that is roughly **27× headroom**, which
changes what you can do:

- **No cold start.** The model stays loaded. Cloud Run pays ~15 s on a
  scaled-to-zero request and a sleeping Space pays more; here the process
  simply never stops.
- **The SQLite cache is permanent.** Summaries survive reboots and
  redeploys, so each paper is sent to Mistral once *ever* rather than once
  per deploy. On the ephemeral hosts you re-pay after every push.
- **Compose works as-is.** One box runs both containers on one network,
  so the app is same-origin — no `VITE_API_BASE`, no
  `PAPERPILOT_CORS_ORIGINS`, none of the split-hosting failure modes in
  this document apply.

### ARM64: checked, and it is fine

The A1 shapes are aarch64, not x86, which is the first thing to worry about
for a PyTorch service. It turns out not to be a problem — the exact wheel
this image pins exists on the exact index it pins:

```bash
$ curl -s https://download.pytorch.org/whl/cpu/torch/ | grep 'torch-2.5.1-cp312.*aarch64'
torch-2.5.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

spaCy ships 10 aarch64 wheels and scikit-learn 7, so the rest of the stack
resolves too. **`backend/Dockerfile` builds unmodified on an A1 instance.**
Build it *on* the instance rather than cross-building on an x86 laptop —
`docker buildx` under QEMU emulation will work and will take the better
part of an hour.

The one way to get this wrong is shape selection. The tier also includes
two AMD `VM.Standard.E2.1.Micro` instances, and those are **1 GB of RAM
each** — under the 865 MB peak once the OS takes its share. Pick
`VM.Standard.A1.Flex`.

### The three things that actually go wrong

**"Out of host capacity."** The notorious one. A1 capacity in popular
regions is frequently exhausted and launches fail for days. Your home
region is fixed at signup and cannot be changed afterwards, so choose a
quieter one then rather than the nearest big one. Upgrading to
Pay-As-You-Go improves your priority and still costs nothing while you
stay inside the Always Free limits.

**Idle reclamation.** Oracle reclaims Always Free compute that has been
idle for 7 days — roughly, under 20% CPU with low network and memory use.
A portfolio demo nobody visits is *precisely* that profile, so the instance
you set up in January is gone in February. Upgrading to Pay-As-You-Go
exempts you from reclamation and remains $0 inside the free limits; that
upgrade is the single most useful thing you can do to this account.

**The firewall is in two places.** Opening a port in the OCI Security List
is half the job: Oracle's Ubuntu and Oracle Linux images also ship
restrictive `iptables` rules that drop everything except SSH. Traffic dies
silently at the second one, and the symptom is a connection timeout that
looks exactly like a wrong Security List.

```bash
# on the instance, after opening 80/443 in the OCI Security List
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save        # Ubuntu; otherwise iptables-save
```

### Setting it up

```bash
# 1. Launch VM.Standard.A1.Flex — 4 OCPU / 24 GB, Ubuntu 22.04 or 24.04.
#    Add your SSH key. Open 80 and 443 in the Security List.

# 2. On the instance:
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu && newgrp docker

git clone https://github.com/zainabraza06/PaperPilot && cd PaperPilot
printf 'PAPERPILOT_MISTRAL_API_KEY=%s\n' "$KEY" > .env
docker compose up --build -d          # first build ~15 min on 4 ARM cores
```

That is already a working deployment on port 5173. For a real hostname
with TLS, put Caddy in front — it obtains and renews the certificate
itself:

```caddyfile
# /etc/caddy/Caddyfile
paperpilot.example.com {
    reverse_proxy localhost:5173
}
```

Because everything is behind one origin, nothing in the app needs to know
its own public URL.

### Should you use it?

**Yes, if** you want the demo to be instant rather than cold-starting, you
want the summary cache to persist, or you want a VM on your CV.

**No, if** you would rather not own a public Linux box. This is the only
option here where you are responsible for TLS renewal, patching, and the
fact that an exposed instance is a real target. Hugging Face Spaces gets
you a working public demo in twenty minutes with none of that, and for a
portfolio link that is usually the better trade.

---

## Costs you should expect

Everything above is free except the LLM. With `ministral-8b-latest` at
roughly 700 input and 100 output tokens per paper, a 50-paper search costs
well under a cent — and the SQLite cache means each paper is summarized
once, ever, not once per search.

Leave `PAPERPILOT_MISTRAL_API_KEY` unset and the app still works: summaries
become extractive, labelled `From abstract` in the UI, and every other
feature is unaffected. That degradation is deliberate and worth keeping in
a public demo — it caps your spend at zero while still showing the pipeline.

---

## Things that will bite you

Each of these was hit while writing this guide, not imagined.

**A hard-coded port.** Fixed in `backend/Dockerfile`, but if you write your
own entrypoint, read `$PORT`. The failure gives you a health-check timeout
and no clue.

**`PAPERPILOT_CORS_ORIGINS` as a bare string used to crash the container at
startup.** Pydantic parses a `list[str]` from the environment as JSON and
nothing else, so the obvious value raised `SettingsError` during import —
a stack trace about JSON decoding that never names the variable you set.
It now accepts bare, comma-separated and JSON forms. If you pin an older
commit, use the JSON array.

**The first Space build looks hung.** It is pulling ~600 MB of wheels and
baking in two models. 10–20 minutes is normal; the build log is the only
honest progress indicator.

**arXiv rate-limits by IP, and the block outlasts the documented
1-request-per-3-seconds window.** On shared hosting you may inherit someone
else's block. The app degrades correctly — the banner names arXiv as
unavailable and the other two sources still answer — but do not mistake it
for a bug in your deploy.

**Spaces sleep after 48 hours of inactivity.** The first request afterwards
pays the cold start. For a portfolio link that matters; a weekly cron
hitting `/health` is enough to keep it warm.
