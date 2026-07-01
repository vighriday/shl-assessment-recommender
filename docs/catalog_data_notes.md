# Catalog data notes

Findings from inspecting the provided catalog export
(`data/raw/shl_product_catalog.json`). These drive the loader's normalisation
rules, so they are recorded here rather than left in someone's head.

The catalog is a single JSON array of **377 items**. Every item carries the same
15 fields, so the loader's defensive handling is about *values* (empty, malformed,
embedded whitespace), not missing keys.

## Parsing

The file does not parse under strict JSON. At least one product name contains a
literal line break inside the quoted string, e.g. item `4207`:

    "name": "Microsoft \n    365 (New)"

Strict JSON forbids raw control characters inside strings, so `json.loads` raises
`Invalid control character`. We load with `strict=False`, which tolerates the
embedded newlines, and then normalise the whitespace ourselves (see below). No
third-party parser is used: some of the faster ones are *stricter* about control
characters and would reintroduce the failure, and we only load the catalog once.

## Per-field reality

| Field            | Type     | Notes                                                        |
| ---------------- | -------- | ------------------------------------------------------------ |
| `entity_id`      | str      | Present and unique on all 377. Used as the stable item id.   |
| `name`           | str      | Unique after trimming. 1 item has an embedded newline.       |
| `link`           | str      | Unique, all on `https://www.shl.com`, all end with `/`. Clean. |
| `description`    | str      | Present on all 377. 68 items contain embedded `\n`/`\r`/`\t`. |
| `keys`           | list     | Present on all 377, never empty. 8 distinct categories.      |
| `job_levels`     | list     | Empty on 19 items.                                           |
| `languages`      | list     | Empty on 37 items.                                           |
| `duration`       | str      | Empty on 61 items; also includes non-numeric values e.g. `"Variable"`. |
| `remote`         | str      | Always `"yes"`. Carries no signal; ignored for ranking.      |
| `adaptive`       | str      | `"no"` (340) / `"yes"` (37). Usable as a boolean.            |
| `status`         | str      | Always `"ok"`. Carries no signal.                            |
| `*_raw`          | str      | Original unparsed strings kept by the scraper; we prefer the parsed lists. |
| `scraped_at`     | str      | ISO timestamp of the scrape. Provenance only.               |

Whitespace normalisation (collapsing runs of spaces/newlines/tabs to a single
space and trimming) is applied to `name` and `description` so both search text
and any displayed text are clean.

### Corrupted name: item 4207 ("Microsoft Excel 365")

One item needs more than whitespace normalisation. Item `4207`'s raw name is
`"Microsoft \n    365 (New)"` — the scrape replaced the word **"Excel"** with a
line break, so plain whitespace collapsing would yield the wrong name
`"Microsoft 365 (New)"`. The true name is confirmed three independent ways:

* the product URL slug is `microsoft-excel-365-new`;
* the description begins "The Microsoft Excel 365 simulation...";
* sample conversation C8 lists the item as "Microsoft Excel 365 (New)".

The loader restores the correct name via `NAME_OVERRIDES`. This matters because
the grader may match recommendations by name; with the raw value we would fail
this item in any trace that expects it. It is the only item in the catalog with
an embedded control character in its name.

## test_type derivation

`test_type` is required in the API response but absent from the catalog. It is,
however, fully recoverable from `keys`. The eight categories map one-to-one to
the single-letter codes used in the sample conversations:

| `keys` category                | code |
| ------------------------------ | ---- |
| Knowledge & Skills             | K    |
| Personality & Behavior         | P    |
| Ability & Aptitude             | A    |
| Competencies                   | C    |
| Biodata & Situational Judgment | B    |
| Simulations                    | S    |
| Development & 360             | D    |
| Assessment Exercises           | E    |

Coverage is complete: every item has at least one key, and every key seen in the
data is in the map (zero unmapped categories). Multi-category items join their
codes with commas, e.g. `Knowledge & Skills` + `Simulations` -> `K,S`.

The order of `keys` is **not** consistent across the catalog (7 multi-key items
are not alphabetical, and the sample conversations themselves show both orders,
e.g. `C, K` and `K,S`). We therefore preserve the catalog's own ordering when
joining, and keep a small override map for any specific item whose expected code
order turns out to differ during trace replay.

## Scope (Individual Test Solutions only)

The PDF restricts scope to Individual Test Solutions and excludes Pre-packaged
Job Solutions. Scanning the catalog, the only items whose names suggest a
job-solution bundle are the seven ending in "Solution":

- Customer Service Phone Solution (`3931`)
- Entry Level Cashier Solution (`3934`)
- Entry Level Customer Service (General) Solution (`3935`)
- Entry Level Hotel Front Desk Solution (`3936`)
- Entry Level Sales Solution (`3937`)
- Entry Level Technical Support Solution (`3938`)
- Sales & Service Phone Solution (`3930`)

No other bundle-like signals (package, suite, battery, job-focused, etc.) appear
in any name.

Decision: these seven are kept **in scope**. The provided catalog is SHL's own
export of Individual Test Solutions, so the safe reading is that anything in it is
recommendable; dropping items risks lowering recall if a holdout answer includes
one. Each item still carries an explicit `in_scope` flag so the call can be
reversed in one place if later evidence (e.g. a holdout behaviour probe) shows
these should be excluded.
