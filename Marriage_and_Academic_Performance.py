# Marriage during university and academic performance
# Install: pip install pandas requests beautifulsoup4 lxml matplotlib
# Run: python Marriage_and_Academic_Performance.py
# Sources: UCI dataset 697 and PMC11585708, Table 1.
# Marital status is not tracked over time, so this cannot test the effect of getting married.

import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

folder = Path(__file__).resolve().parent
headers = {"User-Agent": "University coursework data analysis/1.0"}
uci_url = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
pmc_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11585708/"

# Download the public UCI file. One row represents one student.
url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
response = requests.get(url, headers=headers, timeout=120)
response.raise_for_status()
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    students = pd.read_csv(archive.open("data.csv"), sep=";")
students.columns = [re.sub(r"\s+", " ", name).strip() for name in students.columns]

columns = {
    "Marital status": "Marital_Status_Raw",
    "Age at enrollment": "Age",
    "Daytime/evening attendance": "Attendance_Raw",
    "Scholarship holder": "Scholarship_Raw",
    "Curricular units 1st sem (grade)": "Semester_1_Grade",
    "Curricular units 2nd sem (grade)": "Semester_2_Grade",
    "Curricular units 1st sem (evaluations)": "Semester_1_Evaluations",
    "Curricular units 2nd sem (evaluations)": "Semester_2_Evaluations",
    "Target": "Outcome",
}
a = students[list(columns)].rename(columns=columns).copy()
a["Source"] = "UCI dataset 697"
a["Source_URL"] = uci_url
a["Record_Type"] = "student"
a["Count_Raw"] = "1"
# Check duplicates using every original field, not just the selected columns.
# Without student IDs, identical records cannot be confirmed as the same person.
a["Exact_Duplicate"] = students.duplicated()
a["Marital_Status_Raw"] = a["Marital_Status_Raw"].astype(str)
print("UCI initial rows:")
print(a.head())

# Read the paper through NCBI's official API, rather than bypassing a website challenge.
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11585708&retmode=xml"
response = requests.get(url, headers=headers, timeout=90)
response.raise_for_status()
soup = BeautifulSoup(response.content, "xml")
table = soup.find("table")
if table is None:
    raise ValueError("The article table was not returned.")

levels = ["Poor", "Satisfactory", "Good", "Excellent"]
rows = []
variable = ""
for tr in table.find_all("tr")[1:]:
    cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["td", "th"])]
    if len(cells) == 7:
        variable, category = cells[:2]
        values = cells[2:]
    elif len(cells) == 6:
        category = cells[0]
        values = cells[1:]
    else:
        continue
    if variable.strip().lower() != "marital status":
        continue
    counts = [int(re.match(r"\d+", value).group()) for value in values]
    if sum(counts[:4]) != counts[4]:
        raise ValueError("Published row total does not match the parsed counts.")
    for level, value in zip(levels, values[:4]):
        rows.append({"Marital_Status_Raw": category, "Performance_Level": level, "Count_Raw": value})

b = pd.DataFrame(rows)
if len(b) != 12:
    raise ValueError("Expected three marital groups and four performance levels in Table 1.")
b["Source"] = "Wube et al. 2024, PMC11585708 Table 1"
b["Source_URL"] = pmc_url
b["Record_Type"] = "published count"
b["Exact_Duplicate"] = False
print("PMC initial rows:")
print(b.head())

# Keep aggregate counts as counts. Do not create imaginary student records from them.
raw = pd.concat([a, b], ignore_index=True, sort=False)
raw.to_csv(folder / "combined_raw.csv", index=False)


def inspect(frame, label):
    print("\n" + label)
    print(frame.head())
    frame.info()
    print(frame.describe(include="all").to_string())
    print("Missing values:")
    print(frame.isnull().sum())


inspect(raw, "Before cleaning")
clean = raw.copy()
for column in clean.select_dtypes(include="object"):
    clean[column] = clean[column].map(
        lambda value: re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()
        if isinstance(value, str) else value
    )

status_codes = {
    "1": "Single", "2": "Married", "3": "Previously married",
    "4": "Previously married", "5": "Cohabiting", "6": "Previously married",
    "single": "Single", "married": "Married", "with fiancée": "Engaged", "with fiancee": "Engaged",
}
clean["Marital_Status"] = clean["Marital_Status_Raw"].str.lower().map(status_codes)
if clean["Marital_Status"].isna().any():
    raise ValueError("An unfamiliar marital status needs checking.")
clean["N"] = clean["Count_Raw"].str.extract(r"^(\d+)")[0].astype(int)
student_rows = clean["Record_Type"].eq("student")
if clean.loc[~student_rows, "N"].sum() != 633:
    raise ValueError("The article counts should total 633 women.")

numeric = ["Age", "Semester_1_Grade", "Semester_2_Grade", "Semester_1_Evaluations", "Semester_2_Evaluations"]
for column in numeric:
    clean[column] = pd.to_numeric(clean[column], errors="coerce")

# Only include a semester grade when that semester has recorded evaluations.
# Also exclude zero averages from this particular comparison, and report how many.
grades = clean[["Semester_1_Grade", "Semester_2_Grade"]].copy()
for semester in [1, 2]:
    grades.loc[clean[f"Semester_{semester}_Evaluations"].le(0), f"Semester_{semester}_Grade"] = float("nan")
print("Zero semester grades excluded:", int(grades.eq(0).sum().sum()))
clean["Mean_Grade_20"] = grades.where(grades.gt(0)).mean(axis=1)
clean["Attendance_Mode"] = pd.to_numeric(clean["Attendance_Raw"], errors="coerce").map({1: "Daytime", 0: "Evening"})
clean["Scholarship"] = pd.to_numeric(clean["Scholarship_Raw"], errors="coerce").map({1: "Yes", 0: "No"})
clean["Outcome"] = clean["Outcome"].str.title()
clean["Dropout"] = clean["Outcome"].map({"Dropout": 1, "Enrolled": 0, "Graduate": 0})

# Do not pretend the two grade systems are interchangeable.
# UCI keeps numeric grades out of 20; PMC keeps the paper's four CGPA categories.
clean["Performance_Measure"] = clean["Record_Type"].map({
    "student": "Mean evaluated semester grade, 0-20",
    "published count": "Paper CGPA category, 0-4 scale",
})
print("Identical full UCI rows removed:", int(clean["Exact_Duplicate"].sum()))
clean = clean.loc[~clean["Exact_Duplicate"]].drop(columns=["Exact_Duplicate", "Count_Raw"]).reset_index(drop=True)
# Other missing fields are not supplied by that source, so leave them missing.
clean.to_csv(folder / "combined_clean.csv", index=False)
inspect(clean, "After cleaning")
print("Rows before / after:", len(raw), len(clean))

# Compare within each source, never pool student rows with published table counts.
a = clean.loc[clean["Record_Type"].eq("student")].copy()
b = clean.loc[clean["Record_Type"].eq("published count")].copy()
summary = a.groupby("Marital_Status").agg(
    Students=("N", "sum"), Mean_Age=("Age", "mean"),
    Mean_Grade=("Mean_Grade_20", "mean"), Dropout_Rate=("Dropout", "mean"),
)
print("\nUCI results by marital status:")
print(summary.round(3))
print("\nUCI students aged 25 and over:")
print(a.loc[a["Age"].ge(25)].groupby("Marital_Status")[["Mean_Grade_20", "Dropout"]].mean().round(3))
print("\nUCI attendance and scholarship:")
print(pd.crosstab(a["Marital_Status"], a["Attendance_Mode"], normalize="index").round(3))
print(pd.crosstab(a["Marital_Status"], a["Scholarship"], normalize="index").round(3))
counts = b.pivot_table(index="Marital_Status", columns="Performance_Level", values="N", aggfunc="sum").reindex(columns=levels)
print("\nPMC published counts:")
print(counts)
percentages = counts.div(counts.sum(axis=1), axis=0) * 100
print("\nPMC percentages within each marital group:")
print(percentages.round(1))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
summary["Mean_Grade"].plot.bar(ax=axes[0], color="steelblue", title="UCI: mean grade out of 20")
(summary["Dropout_Rate"] * 100).plot.bar(ax=axes[1], color="coral", title="UCI: dropout (%)")
percentages.plot.barh(stacked=True, ax=axes[2], title="PMC: performance categories (%)")
axes[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=35)
    ax.set_xlabel("")
fig.tight_layout()
plt.show()

print("\nThe grade and dropout summaries are associations, not effects of marriage.")
print("Check the age-restricted results: the married group is much older in UCI.")
print("Daytime/evening is attendance mode, not a record of classes missed.")
print("Neither source measures marriage timing, study habits, motivation or time management.")
print("Benefits and challenges of marriage cannot be established from these data alone.")
print("The sources use different populations and performance measures; do not pool their results.")
