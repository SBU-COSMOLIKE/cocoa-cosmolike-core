"""Caches of notebook results: read when present, built when missing, never replaced.

A project's notebooks keep their expensive results (data vectors,
derivatives, Fisher matrices) in <folder>/<name>.npz, where <folder> is the
project's chains/ (not in git). read_or_build reads the file when it is
there, and otherwise computes the result and writes it, so each notebook
runs on its own on a fresh clone and reuses its caches afterwards.

A file is never replaced. A cache made with other settings stops the read,
with a message naming every difference; write_fisher_cache refuses an
existing file unless it is called with overwrite=True.

The file layout. A cache is one .npz file (numpy's zip archive of arrays):
  - each array of the result is stored under its path in the result,
    joined by "/" (e.g. "ops/b1");
  - the other entries of the result (numbers, text, lists of them) go to
    the JSON text entry "__values__";
  - the JSON text entry "__record__" says how the cache was made:

      {"name", "created", "settings": {flat key: value}, "parameters": [...],
       "data_files": {file name: sha256}}

"settings" is the settings dict the caller passed, flattened with "/"; the
keys that end in data_files/<file> are data-file checksums and go to
"data_files", keyed by the file name when that name is unique and by the
full settings path when two inputs share a name (e.g. two datasets'
mask.txt), so neither checksum can replace the other. A read with settings compares both entry by entry. The file
holds no pickles (Python's serialized objects), so reading a cache runs no
code.

Every function takes the cache folder as its folder argument; a project
wraps them with its own chains/ as the default.
"""
import datetime
import json
import os
import re

import numpy as np


# A cache name is lower-case letters, digits and underscores, e.g.
# model_derivatives_dz0p05_final: what it holds, the layout, the tier.
CACHE_NAME_PATTERN = r"[a-z0-9_]+"


def cache_name(quantity, layout=None, variant=None, tier="final"):
    """Return the name of a cache: <quantity>[_<layout>][_<variant>]_<tier>.

    Every notebook and script builds its cache names here, so that they all
    find one cache under one name. Only the layout, the variant and the tier
    go into the name. Every other setting is in the cache's record and is
    compared on a read.

    Arguments:
      quantity = what the cache holds, e.g. 'model_derivatives',
                 'model_scale_cuts', 'linear_probes'.
      layout   = the layout tag ('dcom62', 'dz0p02', 'dz0p05', 'dz0p20'), or
                 None for a cache of every layout.
      variant  = a variant of the layout ('pairmax', 'sigv600', 'lsst_ab3'),
                 or None.
      tier     = the derivative tier ('final' or 'exploration').

    Returns:
      str, e.g. cache_name('model_derivatives', 'dz0p02', 'pairmax') =
      'model_derivatives_dz0p02_pairmax_final'.

    Raises:
      ValueError when the name holds a character outside CACHE_NAME_PATTERN.
    """
    parts = [quantity]

    if layout is not None:
        parts.append(str(layout))

    if variant is not None:
        parts.append(str(variant))

    parts.append(tier)
    name = "_".join(parts)

    if re.fullmatch(CACHE_NAME_PATTERN, name) is None:
        raise ValueError(f"cache name {name!r}: lower case letters, digits and underscores")

    return name


def _cache_path(folder, name):
    """Return folder/<name>.npz.

    Raises:
      ValueError unless name is a bare cache name (no folder, no suffix).
    """
    if not isinstance(name, str) or re.fullmatch(CACHE_NAME_PATTERN, name) is None:
        raise ValueError(f"cache name {name!r}: lower case letters, digits and "
                         "underscores, without a folder or the .npz suffix")

    if folder is None:
        raise ValueError("the cache folder is required "
                         "(the project's chains/ folder)")

    return os.path.join(folder, name + ".npz")


def _plain(value):
    """Return value in JSON types: numpy scalars and arrays become numbers and lists.

    Dictionaries, lists and tuples are converted entry by entry (tuples become
    lists, dictionary keys become text). Other values are returned unchanged.
    """
    if isinstance(value, dict):
        out = {}
        for key in value:
            out[str(key)] = _plain(value[key])
        return out

    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.append(_plain(item))
        return out

    if isinstance(value, np.ndarray):
        return value.tolist()

    # np.generic is the base class of numpy scalars (np.float64, np.int64...).
    if isinstance(value, np.generic):
        return value.item()

    return value


def _flatten(value, prefix, out):
    """Fill out with {path joined by "/": leaf} of a nested dict, and return out.

    Example: _flatten({"a": {"b": 1}, "c": 2}, "", {}) = {"a/b": 1, "c": 2}.
    An empty dict is a leaf, so it is kept.

    Arguments:
      value  = the nested dict (or a leaf).
      prefix = the path of value so far ("" at the top).
      out    = the dict being filled.
    """
    if isinstance(value, dict) and value:
        for key in value:
            if prefix:
                path = f"{prefix}/{key}"
            else:
                path = str(key)
            _flatten(value[key], path, out)
    else:
        out[prefix] = value

    return out


def cache_record(name, settings, result=None):
    """Return the record of a cache: its settings flat, the data files apart.

    Arguments:
      name     = the cache name.
      settings = the nested settings dict.
      result   = the result; its "names" entry, if any, becomes "parameters".

    Returns:
      dict {"name", "created", "settings", "data_files"[, "parameters"]},
      in JSON types.
    """
    flat = _plain(_flatten(settings, "", {}))

    record = {
        "name": name,
        "created": datetime.date.today().isoformat(),
        "settings": {},
        "data_files": {},
    }

    # A key ending in data_files/<file> is a checksum (from model_settings);
    # every other key is a setting.
    checksums = {}
    for key in flat:
        parts = key.split("/")
        if len(parts) > 1 and parts[-2] == "data_files":
            checksums[key] = flat[key]
        else:
            record["settings"][key] = flat[key]

    # A unique file name stays the short key (as in every record written
    # before); two inputs with the same file name keep their full settings
    # path, so the second checksum cannot overwrite the first.
    files = [key.split("/")[-1] for key in checksums]
    for key, value in checksums.items():
        file = key.split("/")[-1]
        record["data_files"][file if 1 == files.count(file) else key] = value

    if isinstance(result, dict) and "names" in result:
        record["parameters"] = _plain(result["names"])

    # The JSON round trip turns the current record into exactly what a stored
    # one reads back as (tuples become lists, for example), so the two compare.
    return json.loads(json.dumps(record))


def _file_differences(stored, current):
    """Return one line per data-file checksum that differs between two records.

    A record keys a checksum by the file name, or by its full settings path
    when two inputs share a name. A record written before shared names were
    kept stored one checksum under the bare name: it still matches when
    every current entry of that name carries exactly that checksum, and
    differs otherwise.

    Arguments:
      stored  = the "data_files" of the stored record.
      current = the "data_files" of the current record.

    Returns:
      list of str, empty when every checksum agrees.
    """
    out = []

    # The current entries of each file name, whether keyed short or by path.
    by_name = {}
    for key, value in current.items():
        by_name.setdefault(key.split("/")[-1], []).append((key, value))

    covered = set()
    for key, value in stored.items():
        if key in current:
            covered.add(key)
            if current[key] != value:
                out.append(f"{key}: cache {value}, now {current[key]}")
            continue

        # A bare name in the stored record: every current entry of that
        # name must carry the stored checksum.
        entries = by_name.get(key, []) if "/" not in key else []
        if not entries:
            out.append(f"{key}: cache {value}, now (absent)")
            continue
        for current_key, current_value in entries:
            covered.add(current_key)
            if current_value != value:
                out.append(f"{current_key}: cache {value}, now {current_value}")

    for key, value in current.items():
        if key not in covered:
            out.append(f"{key}: cache (absent), now {value}")

    return out


def _differences(stored, current, ignored=()):
    """Return one line per key whose value differs between two flat dicts.

    A key present on one side only shows "(absent)" on the other. Keys in
    ignored are skipped.
    """
    out = []
    every_key = sorted(set(stored) | set(current))

    for key in every_key:
        if key in ignored:
            continue

        old = stored.get(key, "(absent)")
        new = current.get(key, "(absent)")
        if old != new:
            out.append(f"{key}: cache {old}, now {new}")

    return out


def write_fisher_cache(result, name, settings, overwrite=False, folder=None):
    """Write a computed result to <folder>/<name>.npz with its record.

    Arguments:
      result    = nested dicts of numpy arrays, numbers, text and lists.
      name      = the cache name, e.g. 'model_derivatives_dz0p05_final'.
      settings  = the settings it was computed with (model_settings for each
                  model among them).
      overwrite = True to replace an existing file.
      folder    = the folder of the file; the project's cache folder (required).

    Returns:
      the path written.

    Raises:
      FileExistsError when the file exists and overwrite is False.
    """
    path = _cache_path(folder, name)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; nothing was written. Pass "
                              "overwrite=True to replace it")

    # Arrays are stored as arrays; every other leaf goes to the JSON entry.
    arrays = {}
    values = {}
    flat = _flatten(result, "", {})
    for key in flat:
        if isinstance(flat[key], np.ndarray):
            arrays[key] = flat[key]
        else:
            values[key] = _plain(flat[key])

    arrays["__values__"] = np.array(json.dumps(values))

    record = cache_record(name, settings, result)
    arrays["__record__"] = np.array(json.dumps(record, indent=1))

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write to a temporary name first, then rename: an interrupted write
    # leaves no truncated cache under the real name.
    partial = path + ".partial"
    with open(partial, "wb") as stream:
        # Given a file object, savez_compressed does not append ".npz".
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)

    print(f"wrote {path}", flush=True)
    return path


def read_fisher_cache_header(name, folder=None):
    """Return the record of <folder>/<name>.npz without reading any array."""
    path = _cache_path(folder, name)

    with np.load(path, allow_pickle=False) as data:
        record_text = data["__record__"].item()

    return json.loads(record_text)


def read_fisher_cache(name, settings=None, comparable=None, tier=None, folder=None):
    """Read <folder>/<name>.npz after checking its record; nothing is recomputed.

    Arguments:
      name       = the cache name.
      settings   = the notebook's settings for this cache, or None to skip the
                   comparison. Every flat entry (and every data-file checksum)
                   must equal the stored one.
      comparable = None, or a function(settings) returning the settings
                   without the entries that may differ (file paths, for
                   example). The entries it drops are ignored on both sides.
      tier       = None, or the tier that every "tier" entry of the stored
                   settings must have (entries set to None pass).
      folder     = the folder of the file; the project's cache folder (required).

    Returns:
      the cached result: nested dicts, as write_fisher_cache received them.

    Raises:
      FileNotFoundError without the file; ValueError naming every differing
      setting with both values, or the other tier.
    """
    path = _cache_path(folder, name)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist: compute the result and "
                                "write it with read_or_build or write_fisher_cache")

    # 1. Read the record, the JSON values and every array.
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(data["__record__"].item())
        flat = json.loads(data["__values__"].item())
        for key in data.files:
            if key not in ("__record__", "__values__"):
                flat[key] = data[key]

    # 2. Compare the stored settings and checksums with the current ones.
    if settings is not None:
        current = cache_record(name, settings)

        # The entries that comparable drops are ignored on both sides.
        ignored = set()
        if comparable is not None:
            kept = cache_record(name, comparable(settings))
            ignored = set(current["settings"]) - set(kept["settings"])

        lines = _differences(record["settings"], current["settings"], ignored)
        lines += _file_differences(record["data_files"], current["data_files"])

        if lines:
            raise ValueError(f"{path} was made with other settings ({len(lines)} "
                             "differ): " + "; ".join(lines))

    # 3. Check the tier: every stored "tier" entry that is set must match.
    if tier is not None:
        tiers = {}
        for key in record["settings"]:
            value = record["settings"][key]
            if key.split("/")[-1] == "tier" and value is not None:
                tiers[key] = value

        wrong_tier = False
        for key in tiers:
            if tiers[key] != tier:
                wrong_tier = True

        if not tiers or wrong_tier:
            raise ValueError(f"{path} has the tier entries {tiers}; this notebook "
                             f"runs the {tier!r} tier")

    # 4. Rebuild the nested dicts from the "/"-joined keys.
    result = {}
    for key in flat:
        value = flat[key]

        # A zero-dimensional array (one number) comes back as that number.
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value[()]

        # Walk down the path, creating each missing level: setdefault returns
        # the existing dict, or inserts an empty one and returns it.
        parts = key.split("/")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    print(f"read {path} (written {record['created']})", flush=True)
    return result


def read_or_build(name, settings, build, comparable=None, tier=None, folder=None):
    """Read <folder>/<name>.npz if it exists; otherwise build the result and write it.

    This is the one call a notebook makes per cache, so that it runs on a
    fresh clone and reuses its caches afterwards.
      - An existing file is read with read_fisher_cache, which raises when its
        settings differ from settings: a cache is never recomputed or replaced
        behind the user's back.
      - A missing file is computed by build() and written with
        write_fisher_cache. The result returned is then read back from the
        file, so both paths give the same types (dictionary keys as text,
        scalars as numbers).

    Arguments:
      name       = the cache name.
      settings   = the settings of the result (stored, and compared on a read).
      build      = a function with no arguments that returns the result; it is
                   called only when the file is missing.
      comparable = passed to read_fisher_cache.
      tier       = passed to read_fisher_cache.
      folder     = the folder of the file; the project's cache folder (required).

    Returns:
      the cached result, as read_fisher_cache returns it.
    """
    path = _cache_path(folder, name)

    if os.path.isfile(path):
        print(f"{name}: reading the cache", flush=True)
    else:
        print(f"{name}: no cache at {path}; computing it", flush=True)
        result = build()
        write_fisher_cache(result, name, settings, folder=folder)

    return read_fisher_cache(name, settings=settings, comparable=comparable, tier=tier,
                             folder=folder)
