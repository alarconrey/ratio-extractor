"""Build the paragraph dataset from SQL Server scripts of the `arbol` database.

Usage:
    python -m app.build_dataset dump_2022.sql dump_2023.zip dump_2024.sql

Each script is the output of SSMS "Generate Scripts" (schema and data, Unicode).
The `users` table is never loaded. Labels come from aligning each resolution with
the `ratiodecidenti` field of its fichas (see docs/data.md).
"""

import argparse
import io
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import GroupShuffleSplit

from app.settings import DATA_DIR, DATASET_FILE, LABEL_OVERLAP, MIN_RECOVERED_RATIO, RESOLUTIONS_FILE, SEED, SPLITS_FILE, TYPE_NAMES
from app.text import paragraph_features, split_paragraphs, word_ngrams, words

TABLES = {"resolucion", "ficha"}


def read_script(path):
    """Return the script as UTF-8 text; accepts .sql (UTF-16 or UTF-8) or a .zip containing one .sql."""
    raw = Path(path).read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            raw = archive.read(next(name for name in archive.namelist() if name.endswith(".sql")))
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[1:2] == b"\x00" else raw.decode("utf-8-sig")
    return text.replace("\x00", "").replace("\r\n", "\n")


def load_script(text, connection):
    """Load only the resolucion and ficha tables of a SQL Server script into SQLite."""
    for match in re.finditer(r"CREATE TABLE \[dbo\]\.\[(\w+)\]\((.*?)\n(?: CONSTRAINT|PRIMARY KEY|\) ON)", text, re.S):
        table, body = match.group(1), match.group(2)
        if table not in TABLES:
            continue
        columns = re.findall(r"^\s*\[([^\]]+)\]\s+\[(\w+)\]", body, re.M)
        definition = ",".join(f'"{name}" {"INTEGER" if kind in ("int", "smallint", "tinyint", "bit") else "TEXT"}' for name, kind in columns)
        connection.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({definition})')
    for statement in re.split(r"\n(?=INSERT \[dbo\]\.|SET IDENTITY_INSERT|GO\n|ALTER TABLE)", text):
        table_match = re.match(r"INSERT \[dbo\]\.\[(\w+)\]", statement)
        if not table_match or table_match.group(1) not in TABLES:
            continue
        statement = re.sub(r"CAST\(N?'([^']*)' AS \w+\)", r"'\1'", statement.strip())
        statement = re.sub(r"CAST\(([\d.]+) AS \w+(\([\d, ]+\))?\)", r"\1", statement)
        head, body = statement.split(" VALUES ", 1)
        head = re.sub(r"\[([^\]]+)\]", r'"\1"', head.replace("INSERT [dbo].", "INSERT OR IGNORE INTO "))
        body = re.sub(r"(?<![\w'])N'", "'", body)
        connection.execute(f"{head} VALUES {body}")


def ensure_keys(connection):
    connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS resolucion_id ON resolucion(idresolucion)')
    connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS ficha_id ON ficha(idficha)')


def align(connection):
    """Label every paragraph of every resolution that has at least one ficha."""
    fichas = pd.read_sql("SELECT idresolucion, idusuario, ratiodecidenti FROM ficha", connection)
    resolutions = pd.read_sql("SELECT idresolucion, resolucion, gestion, idtipo, contenido FROM resolucion", connection).set_index("idresolucion")
    paragraph_rows, resolution_rows = [], []
    for resolution_id, group in fichas.groupby("idresolucion"):
        if resolution_id not in resolutions.index:
            continue
        ratios = {text for text in group.ratiodecidenti if text}
        ratio_ngrams = set()
        for ratio in ratios:
            ratio_ngrams |= word_ngrams(words(BeautifulSoup(ratio, "lxml").get_text(" ")))
        if not ratio_ngrams:
            continue
        meta = resolutions.loc[resolution_id]
        analyst = int(group.idusuario.mode().iloc[0]) if group.idusuario.notna().any() else None
        partition = "entrenamiento" if meta.gestion <= 2022 else "prueba"
        recovered, positives = set(), 0
        features = paragraph_features(split_paragraphs(meta.contenido))
        for row in features:
            ngrams = word_ngrams(words(row["texto"]))
            overlap = len(ngrams & ratio_ngrams) / len(ngrams) if ngrams else 0.0
            label = int(overlap >= LABEL_OVERLAP)
            if label:
                recovered |= ngrams & ratio_ngrams
                positives += 1
            paragraph_rows.append({
                "id_resolucion": int(resolution_id), "resolucion": meta.resolucion, "gestion": int(meta.gestion),
                "tipo": int(meta.idtipo), "tipo_nombre": TYPE_NAMES.get(int(meta.idtipo), "Otro"), "analista": analyst,
                "n_fichas": len(ratios), "particion": partition, **{key: row[key] for key in ["idx", "n_parrafos", "pos_rel", "seccion", "encabezado", "n_palabras"]},
                "solape": round(overlap, 3), "es_ratio": label, "texto": row["texto"],
            })
        coverage = len(recovered) / len(ratio_ngrams)
        resolution_rows.append({
            "id_resolucion": int(resolution_id), "resolucion": meta.resolucion, "gestion": int(meta.gestion), "tipo": int(meta.idtipo),
            "tipo_nombre": TYPE_NAMES.get(int(meta.idtipo), "Otro"), "analista": analyst, "n_fichas": len(ratios),
            "n_parrafos": len(features), "parrafos_ratio": positives, "cobertura_ratio": round(coverage, 3),
            "revisar": int(coverage < MIN_RECOVERED_RATIO), "particion": partition,
        })
    return pd.DataFrame(paragraph_rows), pd.DataFrame(resolution_rows)


def make_splits(resolutions):
    """Temporal test (2023+); validation is 15% of earlier resolutions, grouped by resolution."""
    usable = resolutions[resolutions.revisar == 0]
    earlier = usable[usable.gestion <= 2022]
    train_index, validation_index = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(earlier, groups=earlier.id_resolucion))
    assignment = {**{rid: "train" for rid in earlier.iloc[train_index].id_resolucion},
                  **{rid: "validation" for rid in earlier.iloc[validation_index].id_resolucion},
                  **{rid: "test" for rid in usable[usable.gestion >= 2023].id_resolucion}}
    result = resolutions[["id_resolucion", "resolucion", "gestion"]].copy()
    result["split"] = result.id_resolucion.map(assignment).fillna("excluded")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build the ratio decidendi paragraph dataset from SQL Server scripts.")
    parser.add_argument("scripts", nargs="+", help=".sql or .zip files generated by SSMS")
    args = parser.parse_args()
    connection = sqlite3.connect(":memory:")
    for path in args.scripts:
        print(f"Loading {path}...", flush=True)
        load_script(read_script(path), connection)
        ensure_keys(connection)
    paragraphs, resolutions = align(connection)
    DATA_DIR.mkdir(exist_ok=True)
    paragraphs.to_csv(DATASET_FILE, index=False, compression={"method": "gzip", "mtime": 0})
    resolutions.to_csv(RESOLUTIONS_FILE, index=False)
    make_splits(resolutions).to_csv(SPLITS_FILE, index=False)
    print(f"{len(resolutions)} resolutions, {len(paragraphs)} paragraphs, {int(resolutions.revisar.sum())} excluded for weak alignment.")


if __name__ == "__main__":
    main()
