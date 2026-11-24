# Marriage During University and Academic Performance

Web scraping, data integration, preprocessing and analysis in one Jupyter notebook.

**Question:** does being married during university go with better or worse academic performance than being unmarried? (Association only. The data cannot show cause and effect.)

## Files
| File | What it is |
|---|---|
| `Marriage_and_Academic_Performance.ipynb` | The full executed notebook (all outputs and charts included) |
| `Marriage_and_Academic_Performance.py` | The same project as a Python script in "percent" cells (`# %%`). Run it with `python`, open it in VS Code/Jupyter, or convert with `jupytext --to ipynb` |
| `Marriage_Academic_Combined.csv` | The unified, cleaned table (4,436 rows: 4,424 student-level + 12 aggregate cells) |
| `figures/` | Charts saved by the notebook |
| `raw/` | Raw copies of what was downloaded (UCI CSV, PMC XML) |

## Data sources (both real, nothing simulated)
1. **UCI Machine Learning Repository, dataset 697** (Realinho et al., 2021): 4,424 anonymised student records from a Portuguese polytechnic institute, with marital status, age, attendance mode, scholarship, unit counts, semester grades and dropout outcome. Student-level. Fetched with `requests` (JSON API + dataset ZIP). https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
2. **PubMed Central, PMC11585708** (Wube, Asgedom and Jemal, 2024, *Global Epidemiology* 8:100175): a published study of 633 female students at Aksum University, Ethiopia. Table 1 gives marital status by performance level (Poor/Satisfactory/Good/Excellent). Aggregate counts. Fetched with `requests` through NCBI's official E-utilities API and parsed with `BeautifulSoup`. The website itself shows a bot challenge to scripts; it was not bypassed. https://pmc.ncbi.nlm.nih.gov/articles/PMC11585708/

Every row has a `Source` and a `Record_Type` column (`student-level` or `aggregate-count`). No personal identifiers are collected.

**Compatibility note.** Both sources have marital status and academic performance for university students, but one is student-level and the other aggregated, with different populations and grade scales. They are therefore integrated into one table (`pd.concat`) with an explicit `N` column, harmonised onto the same four performance levels, and analysed **side by side, never pooled into one statistic**. Aggregate counts are never expanded into made-up students. The harmonisation of Source A (rescaling the paper's CGPA cut-offs to the 0-20 scale) is an assumption and is documented in the notebook.

## What the three requirements map to
1. **Web scraping:** sections 2 and 3 (one per source).
2. **Data integration:** section 4.
3. **Data preprocessing:** section 5, with `head()`, `info()`, `describe()` and `isnull().sum()` before and after.

## Main results (details and caveats in the notebook)
- Raw mean grade (UCI): 12.53 married vs 12.72 single (0-20). After adjusting for age, gender, scholarship and evening attendance: +0.13 for married (p = 0.21, not significant). Among students aged 25+ the means are about the same.
- Aksum: no statistically significant difference in performance level between married and unmarried women (p = 0.22 to 0.32).
- Raw dropout is higher for married students (47% vs 30%), but among students aged 25+ it is lower for married (49% vs 60%). Married students are much older, more often in evening classes and less often on scholarship.
- Benefits, challenges, study habits and motivation split by marital status are **not** available in either source, so they are not analysed.

## How to run
```bash
pip install pandas numpy scipy matplotlib seaborn requests beautifulsoup4 lxml notebook
jupyter notebook Marriage_and_Academic_Performance.ipynb      # or: python Marriage_and_Academic_Performance.py
```
Needs internet access (it downloads both sources). Outputs may differ slightly if the sources are updated.

## Limitations
Observational data; different populations and grade scales; small married groups; marital status measured once; no marriage timing; no data on work, income or motivation. See sections 8 and 9 of the notebook.
