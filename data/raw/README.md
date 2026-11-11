# This folder is intentionally empty in the repo

The 7 OULAD CSV files normally go here. They are **not** committed to
this repository — `studentVle.csv` alone is ~430 MB uncompressed, far
too large to version in git.

## To populate this folder

1. Download the dataset (CC BY 4.0 licensed) from the official UCI
   Machine Learning Repository page:
   https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
2. Place all 7 files directly in this folder (`data/raw/`), not in a
   subfolder:
   - `courses.csv`
   - `studentInfo.csv`
   - `studentRegistration.csv`
   - `assessments.csv`
   - `studentAssessment.csv`
   - `vle.csv`
   - `studentVle.csv`
3. From the project root, validate the download:
   ```bash
   python -m src.ingest
   ```
   This checks schema and row counts for all 7 tables and will tell
   you clearly if anything is missing or looks wrong.

See the main `README.md` ("Data preparation") for full setup context.
