import pandas as pd

OFFICIAL_URL = (
    "https://raw.githubusercontent.com/"
    "Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"
    "official_data_en.csv"
)

official = pd.read_csv(
    OFFICIAL_URL,
    parse_dates=["started_at", "finished_at"],
)

print("\n=== OFFICIAL DATASET ===")
print("columns:", official.columns.tolist())
print("rows:", len(official))
print(
    "date range:",
    official["started_at"].min(),
    "->",
    official["started_at"].max(),
)

print("\nRows by geographic level:")
print(official["level"].value_counts(dropna=False))

official_oblast = official.loc[official["level"].eq("oblast")].copy()

print("\n=== OFFICIAL OBLAST-LEVEL SUBSET ===")
print("rows:", len(official_oblast))
print("unique oblasts:", official_oblast["oblast"].nunique())
print(
    "date range:",
    official_oblast["started_at"].min(),
    "->",
    official_oblast["started_at"].max(),
)

duplicate_mask = official_oblast.duplicated(
    subset=["oblast", "started_at", "finished_at"],
    keep=False,
)

duplicate_rows = official_oblast.loc[duplicate_mask].sort_values(
    ["oblast", "started_at", "finished_at"]
)

duplicate_groups = (
    duplicate_rows.groupby(
        ["oblast", "started_at", "finished_at"],
        dropna=False,
    )
    .size()
    .reset_index(name="copies")
)

print("\n=== EXACT DUPLICATES INSIDE OFFICIAL OBLAST LEVEL ===")
print("duplicate rows:", len(duplicate_rows))
print("duplicate interval groups:", len(duplicate_groups))

if duplicate_rows.empty:
    print("No exact duplicates found.")
else:
    print(
        duplicate_rows[
            ["oblast", "started_at", "finished_at", "source"]
        ]
        .head(20)
        .to_string(index=False)
    )

interval_levels = (
    official.groupby(
        ["oblast", "started_at", "finished_at"],
        dropna=False,
    )["level"]
    .agg(lambda values: tuple(sorted(set(values.dropna()))))
    .reset_index(name="levels")
)

cross_level_intervals = interval_levels[
    interval_levels["levels"].map(len) > 1
    ]

print("\n=== EXACT INTERVALS PRESENT AT MULTIPLE LEVELS ===")
print("count:", len(cross_level_intervals))

if not cross_level_intervals.empty:
    print(cross_level_intervals.head(20).to_string(index=False))

deduplicated_official_oblast = (
    official_oblast.drop_duplicates(
        subset=["oblast", "started_at", "finished_at"],
        keep="first",
    )
    .sort_values(["oblast", "started_at", "finished_at"])
    .reset_index(drop=True)
)

removed_rows = len(official_oblast) - len(deduplicated_official_oblast)

print("\n=== DEDUPLICATED OFFICIAL OBLAST SUBSET ===")
print("rows before deduplication:", len(official_oblast))
print("rows removed:", removed_rows)
print("rows after deduplication:", len(deduplicated_official_oblast))
print(
    "unique event keys:",
    deduplicated_official_oblast[
        ["oblast", "started_at", "finished_at"]
    ].drop_duplicates().shape[0],
)

assert (
        len(deduplicated_official_oblast)
        == deduplicated_official_oblast[
            ["oblast", "started_at", "finished_at"]
        ].drop_duplicates().shape[0]
), "Deduplication did not produce unique event keys."