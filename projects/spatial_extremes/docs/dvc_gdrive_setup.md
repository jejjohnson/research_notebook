---
title: DVC + Google Drive setup (spatial-extremes data)
---

# Storing the CDS data cache in Google Drive with DVC

The real spatial-extremes data is the CDS in-situ land archive cached under
`projects/spatial_extremes/data/cds_insitu_land/`. It is too large/derived to
commit to git, so it is tracked with **DVC** and stored in **Google Drive**.

## One-time prerequisites

1. **Accept the CDS dataset licence** (required before any download):
   <https://cds.climate.copernicus.eu/datasets/insitu-observations-surface-land?tab=download#manage-licences>
2. **CDS credentials** in `~/.cdsapirc` (production endpoint):
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-key>
   ```
3. **`dvc-gdrive`** backend — DVC's Google Drive plugin. Install it into the
   environment you run `dvc` from, e.g. `pip install dvc-gdrive` (or
   `uv pip install dvc-gdrive`).

## Remote configuration

The remote is a Google Drive folder identified by its **folder ID** (the part
after `/folders/` in the Drive URL). This is already configured in
`.dvc/config`:

```bash
dvc remote add -d gdrive_remote gdrive://1cwwp2AfKylTASzQucMmUSiRNvpox0pKQ
```

### ⚠️ Auth: the built-in client is blocked

DVC's *shared* OAuth client is now **blocked by Google** for the Drive scope
("This app is blocked"), so a plain `dvc push` fails at the consent screen. Pick
one of the two working paths below.

**Option A — your own OAuth client (recommended for personal Gmail).**
Files stay owned by you, on your quota.

1. [Google Cloud Console](https://console.cloud.google.com/) → create a project.
2. Enable the **Google Drive API** for it.
3. **OAuth consent screen** → *External* → add your Gmail as a **Test user**
   (leave the app in *Testing* mode — no verification needed).
4. **Credentials → Create credentials → OAuth client ID → Desktop app** → copy
   the **client ID** and **client secret**.
5. Wire them into DVC (secret goes in the gitignored `.dvc/config.local`):
   ```bash
   dvc remote modify --local gdrive_remote gdrive_client_id     '<CLIENT_ID>'
   dvc remote modify --local gdrive_remote gdrive_client_secret '<CLIENT_SECRET>'
   dvc push      # browser opens with YOUR client; you're a test user → allowed
   ```

**Option B — service account (only with Google Workspace + a Shared Drive).**
A service account has no Drive quota of its own, so uploads to a personal
"My Drive" folder fail with *storage quota exceeded*; it only works if the
target folder lives on a **Shared Drive**.

1. GCP → create a **service account** → create a **JSON key**, download it to
   `projects/spatial_extremes/.secrets/gdrive-sa.json` (gitignored).
2. Share the Drive folder with the service-account email (Editor).
3. ```bash
   dvc remote modify --local gdrive_remote gdrive_use_service_account true
   dvc remote modify --local gdrive_remote \
       gdrive_service_account_json_file_path projects/spatial_extremes/.secrets/gdrive-sa.json
   dvc push
   ```

Either way, the token/secret lands in `.dvc/config.local` (gitignored); no
secrets enter git.

## Day-to-day workflow

```bash
# 1. download real data (uses ~/.cdsapirc; only fetches missing years)
#    all commands below are run from the repo root
projects/spatial_extremes/.venv/bin/python projects/spatial_extremes/scripts/fetch_cds_insitu.py

# 2. track the cache with DVC (creates data/cds_insitu_land.dvc, gitignores the data)
dvc add projects/spatial_extremes/data/cds_insitu_land

# 3. push the data to Google Drive
dvc push

# 4. commit the pointer + config
git add projects/spatial_extremes/data/cds_insitu_land.dvc .dvc/config .gitignore
git commit -m "data(spatial_extremes): track CDS in-situ land cache in Drive"
```

To retrieve the data on another machine: `git pull` then `dvc pull`.

## Notes

* The notebooks never require this. `spatial_extremes.data.load_station_daily`
  falls back to a deterministic synthetic series when the cache is absent, so
  the curriculum runs offline; it switches to real data automatically once the
  cache (or `dvc pull`) is in place.
* If OAuth gets blocked by Google's app-verification screen, switch to a
  service account: set `gdrive_use_service_account true` and
  `gdrive_service_account_json_file_path` via `dvc remote modify --local`.
