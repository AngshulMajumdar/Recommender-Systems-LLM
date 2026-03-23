from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

REQUIRED_FILES = ["u1.base", "u1.test", "u.user", "u.item"]


def resolve_dataset_root(dataset_path: str | Path, work_dir: str | Path) -> Path:
    dataset_path = Path(dataset_path)
    work_dir = Path(work_dir)
    extract_dir = work_dir / "dataset_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as zf:
            zf.extractall(extract_dir)
        if (extract_dir / "ml-100k").exists():
            root = extract_dir / "ml-100k"
        else:
            child_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(child_dirs) == 1:
                root = child_dirs[0]
            else:
                root = extract_dir
    elif dataset_path.is_dir() and (dataset_path / "ml-100k").exists():
        root = dataset_path / "ml-100k"
    elif dataset_path.is_dir():
        root = dataset_path
    else:
        raise FileNotFoundError(f"Could not resolve dataset path from: {dataset_path}")

    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {missing}")
    return root


def load_users(root: Path) -> pd.DataFrame:
    users = pd.read_csv(
        root / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )
    users["user_id"] = users["user_id"].astype(int)
    users["age"] = users["age"].astype(int)
    return users


def load_items(root: Path) -> pd.DataFrame:
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
    items = pd.read_csv(
        root / "u.item",
        sep="|",
        header=None,
        names=item_cols,
        encoding="latin-1",
    )
    items["item_id"] = items["item_id"].astype(int)
    items["genres"] = items[GENRE_COLUMNS].apply(
        lambda row: [g for g in GENRE_COLUMNS if int(row[g]) == 1], axis=1
    )
    return items


def load_ratings(path: Path, positive_threshold: int = 4) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings[["user_id", "item_id", "rating", "timestamp"]] = ratings[["user_id", "item_id", "rating", "timestamp"]].astype(int)
    ratings["label"] = (ratings["rating"] >= positive_threshold).astype(int)
    return ratings


def build_user_profiles(
    train_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    max_history: int = 10,
) -> Dict[int, dict]:
    item_lookup = items_df.set_index("item_id")[["title", "genres"]].to_dict("index")
    profiles: Dict[int, dict] = {}
    train_sorted = train_df.sort_values(["user_id", "timestamp"])

    for user_id, grp in train_sorted.groupby("user_id"):
        liked_ids = grp.loc[grp["label"] == 1, "item_id"].tolist()[-max_history:]
        disliked_ids = grp.loc[grp["label"] == 0, "item_id"].tolist()[-max_history:]
        liked_titles = [item_lookup[i]["title"] for i in liked_ids if i in item_lookup]
        disliked_titles = [item_lookup[i]["title"] for i in disliked_ids if i in item_lookup]

        genre_pref: Dict[str, int] = {}
        for iid in grp.loc[grp["label"] == 1, "item_id"].tolist():
            if iid in item_lookup:
                for genre in item_lookup[iid]["genres"]:
                    genre_pref[genre] = genre_pref.get(genre, 0) + 1
        genre_pref = dict(sorted(genre_pref.items(), key=lambda kv: (-kv[1], kv[0]))[:5])

        meta_rows = users_df.loc[users_df["user_id"] == user_id]
        meta = meta_rows.iloc[0].to_dict() if len(meta_rows) else {"age": None, "gender": None, "occupation": None, "zip_code": None}
        profiles[int(user_id)] = {
            "user_id": int(user_id),
            "age": None if pd.isna(meta["age"]) else int(meta["age"]),
            "gender": None if pd.isna(meta["gender"]) else str(meta["gender"]),
            "occupation": None if pd.isna(meta["occupation"]) else str(meta["occupation"]),
            "zip_code": None if pd.isna(meta["zip_code"]) else str(meta["zip_code"]),
            "liked_movie_ids": liked_ids,
            "disliked_movie_ids": disliked_ids,
            "liked_movie_titles": liked_titles,
            "disliked_movie_titles": disliked_titles,
            "top_positive_genres": genre_pref,
            "num_train_interactions": int(len(grp)),
            "num_positive_train": int((grp["label"] == 1).sum()),
            "num_negative_train": int((grp["label"] == 0).sum()),
        }
    return profiles


def build_movie_profiles(items_df: pd.DataFrame, train_df: pd.DataFrame) -> Dict[int, dict]:
    stats = train_df.groupby("item_id").agg(
        train_count=("rating", "size"),
        train_positive_count=("label", "sum"),
        train_avg_rating=("rating", "mean"),
    ).reset_index()
    merged = items_df.merge(stats, on="item_id", how="left")
    merged["train_count"] = merged["train_count"].fillna(0).astype(int)
    merged["train_positive_count"] = merged["train_positive_count"].fillna(0).astype(int)
    merged["train_avg_rating"] = merged["train_avg_rating"].fillna(0.0)

    profiles: Dict[int, dict] = {}
    for _, row in merged.iterrows():
        profiles[int(row["item_id"])] = {
            "item_id": int(row["item_id"]),
            "title": "" if pd.isna(row["title"]) else str(row["title"]),
            "release_date": "" if pd.isna(row["release_date"]) else str(row["release_date"]),
            "imdb_url": "" if pd.isna(row["imdb_url"]) else str(row["imdb_url"]),
            "genres": list(row["genres"]),
            "train_count": int(row["train_count"]),
            "train_positive_count": int(row["train_positive_count"]),
            "train_avg_rating": float(row["train_avg_rating"]),
        }
    return profiles


def make_example_row(row, user_profiles: Dict[int, dict], movie_profiles: Dict[int, dict]) -> dict:
    uid = int(row["user_id"])
    iid = int(row["item_id"])
    up = user_profiles.get(uid, {"user_id": uid})
    mp = movie_profiles.get(iid, {"item_id": iid})
    return {
        "user_id": uid,
        "item_id": iid,
        "rating": int(row["rating"]),
        "label": int(row["label"]),
        "user_age": up.get("age"),
        "user_gender": up.get("gender"),
        "user_occupation": up.get("occupation"),
        "user_top_positive_genres": json.dumps(up.get("top_positive_genres", {}), ensure_ascii=False),
        "user_liked_movie_titles": " || ".join(up.get("liked_movie_titles", [])),
        "user_disliked_movie_titles": " || ".join(up.get("disliked_movie_titles", [])),
        "movie_title": mp.get("title"),
        "movie_genres": " | ".join(mp.get("genres", [])),
        "movie_train_count": mp.get("train_count", 0),
        "movie_train_positive_count": mp.get("train_positive_count", 0),
        "movie_train_avg_rating": mp.get("train_avg_rating", 0.0),
    }


def build_examples(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_profiles: Dict[int, dict],
    movie_profiles: Dict[int, dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_examples = pd.DataFrame([make_example_row(r, user_profiles, movie_profiles) for _, r in train_df.iterrows()])
    test_examples = pd.DataFrame([make_example_row(r, user_profiles, movie_profiles) for _, r in test_df.iterrows()])
    return train_examples, test_examples


def prepare_data(dataset_path: str | Path, work_dir: str | Path, positive_threshold: int = 4, max_history: int = 10) -> dict:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    root = resolve_dataset_root(dataset_path, work_dir)
    users_df = load_users(root)
    items_df = load_items(root)
    train_df = load_ratings(root / "u1.base", positive_threshold)
    test_df = load_ratings(root / "u1.test", positive_threshold)
    user_profiles = build_user_profiles(train_df, users_df, items_df, max_history)
    movie_profiles = build_movie_profiles(items_df, train_df)
    train_examples, test_examples = build_examples(train_df, test_df, user_profiles, movie_profiles)

    return {
        "dataset_root": str(root),
        "users_df": users_df,
        "items_df": items_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "user_profiles": user_profiles,
        "movie_profiles": movie_profiles,
    }


def save_builder_outputs(data: dict, out_dir: str | Path) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []

    csv_map = {
        "users_clean.csv": data["users_df"],
        "items_clean.csv": data["items_df"],
        "train_binary.csv": data["train_df"],
        "test_binary.csv": data["test_df"],
        "train_examples.csv": data["train_examples"],
        "test_examples.csv": data["test_examples"],
    }
    for name, df in csv_map.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        files.append(str(path))

    json_map = {
        "user_profiles.json": data["user_profiles"],
        "movie_profiles.json": data["movie_profiles"],
        "summary.json": {
            "dataset_root": data["dataset_root"],
            "num_users": int(data["users_df"]["user_id"].nunique()),
            "num_items": int(data["items_df"]["item_id"].nunique()),
            "train_rows": int(len(data["train_df"])),
            "test_rows": int(len(data["test_df"])),
        },
    }
    for name, obj in json_map.items():
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        files.append(str(path))
    return files
