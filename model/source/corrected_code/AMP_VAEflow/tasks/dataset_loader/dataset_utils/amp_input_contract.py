"""Strict, standard-library-only input boundary for the retrieved AMP root profile.

This is a new isolated adapter, not an executed or verified research loader.
It never creates descriptors/labels, truncates sequences, takes abs, normalizes,
or imports the research repository. Historical data/run identity is out of scope.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
from typing import Sequence

PROFILE = "retrieved-root-48x3-global9-v1"
AMINO_ACIDS = frozenset("ARNDCEQGHILKMFPSTWYV")
BASE_COLUMNS = ("idx", "protein", "describe", "point_describe")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
INTEGER = re.compile(r"[0-9]+\Z")
VECTOR = re.compile(r"\[([^\[\]]*)\]")


@dataclass(frozen=True)
class Issue:
    code: str
    line: int
    field: str
    message: str

    def as_dict(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class Record:
    idx: int
    protein: str
    residue_descriptors: tuple[tuple[Decimal, ...], ...]
    global_descriptor: tuple[Decimal, ...]
    class_index: int | None
    line: int
    # Original field strings are preserved separately from typed numbers.
    raw_fields: tuple[tuple[str, str], ...]
    canonical_triplet_text: str

    def legacy_fields(self) -> dict[str, str]:
        """Map-style bridge; preserve values, canonicalize only triplet separators.

        The model's historical eval is not called here. point_describe has already
        passed a strict JSON array-of-nine-finite-numbers grammar. Typed consumers
        should use global_descriptor and residue_descriptors directly instead.
        """
        fields = dict(self.raw_fields)
        fields["describe"] = self.canonical_triplet_text
        return fields


@dataclass(frozen=True)
class Validation:
    source: str
    source_sha256: str
    stage: str
    delimiter: str | None
    header: tuple[str, ...]
    rows_seen: int
    records: tuple[Record, ...]
    issues: tuple[Issue, ...]

    @property
    def valid(self):
        return not self.issues

    def summary(self):
        counts = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return {
            "profile": PROFILE, "source": self.source,
            "source_sha256": self.source_sha256, "stage": self.stage,
            "delimiter": "tab" if self.delimiter == "\t" else self.delimiter,
            "header": list(self.header), "rows_seen": self.rows_seen,
            "schema_valid": self.valid, "valid_rows_diagnostic_only": len(self.records),
            "issues": [issue.as_dict() for issue in self.issues], "issue_counts": counts,
            "scientific_validity": "NOT_EVALUATED",
            "record_policy": "Any error rejects the complete dataset; no invalid-row filtering or imputation.",
        }


class ContractError(ValueError):
    def __init__(self, result: Validation):
        self.result = result
        first = result.issues[0]
        super().__init__(f"{result.source}:{first.line}:{first.field}: {first.code}: {first.message}")


def _decimal(token: str) -> Decimal:
    if not NUMBER.fullmatch(token):
        raise ValueError("expected a finite numeric token, not an expression")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError("invalid numeric token") from exc
    if not value.is_finite():
        raise ValueError("nonfinite numeric value")
    # Both h.to(torch.float) and g.float() in the root VAE target binary32.
    # Validate that conversion without changing the retained Decimal/token.
    # Normal binary32 rounding is allowed; overflow and nonzero->zero are not.
    try:
        as_float = float(value)
        as_binary32 = struct.unpack("=f", struct.pack("=f", as_float))[0]
    except (OverflowError, ValueError, struct.error) as exc:
        raise ValueError("value cannot convert to finite float32") from exc
    # Some struct implementations produce Inf instead of raising on overflow.
    if not math.isfinite(as_float) or not math.isfinite(as_binary32):
        raise ValueError("value overflows finite float32")
    if value != 0 and (as_float == 0 or as_binary32 == 0):
        raise ValueError("nonzero value underflows to zero in float32 conversion")
    return value


def parse_triplets(text: str):
    """Parse only [n n n],[n n n] grammar; no eval/AST/expression evaluation."""
    matches = list(VECTOR.finditer(text))
    if not matches or text[:matches[0].start()].strip() or text[matches[-1].end():].strip():
        raise ValueError("expected comma-separated bracketed residue triplets")
    values, canonical = [], []
    for i, match in enumerate(matches):
        if i and text[matches[i-1].end():match.start()].strip() != ",":
            raise ValueError("expected exactly one comma between triplets")
        tokens = match.group(1).split()
        if len(tokens) != 3:
            raise ValueError("each residue must have exactly three numeric descriptors")
        values.append(tuple(_decimal(token) for token in tokens))
        canonical.append("[" + " ".join(tokens) + "]")
    return tuple(values), ",".join(canonical)


def parse_global(text: str):
    def reject_constant(value):
        raise ValueError(f"non-JSON numeric constant {value}")
    try:
        values = json.loads(text, parse_float=Decimal, parse_int=Decimal,
                            parse_constant=reject_constant)
    except (json.JSONDecodeError, InvalidOperation, ValueError, RecursionError) as exc:
        raise ValueError("expected a JSON array of nine finite numbers; expressions are forbidden") from exc
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("global descriptor must contain exactly nine numbers")
    if any(type(value) is not Decimal for value in values):
        raise ValueError("global descriptor elements must be numbers, not bool/null/string/list")
    for value in values:
        _decimal(str(value))
    return tuple(values)


def parse_model_batch(batch):
    """Validate the three string fields and return finite float lists for the model.

    This is the replacement for the historical eval/string-splitting block.
    Sequence and per-residue checks apply even if a caller bypasses the loader.
    Values must remain finite/nonzero (when originally nonzero) under the root
    consumer's float32 conversion. Returned floats retain normal Python float
    precision; no pre-rounding, abs, padding, clipping or scaling occurs.
    This input guard does not certify model operations or mixed precision.
    """
    names = ("protein", "describe", "point_describe")
    for name in names:
        if name not in batch or not isinstance(batch[name], (list, tuple)):
            raise ValueError(f"batch {name} must be a list/tuple of original strings")
    sizes = {len(batch[name]) for name in names}
    if len(sizes) != 1 or not next(iter(sizes)):
        raise ValueError("batch fields must have the same nonzero number of records")
    globals_out, triplets_out = [], []
    for row, (protein, describe, point) in enumerate(zip(*(batch[name] for name in names))):
        if not all(isinstance(value, str) for value in (protein, describe, point)):
            raise ValueError(f"batch row {row}: original field strings required")
        if not 1 <= len(protein) <= 48 or set(protein)-AMINO_ACIDS:
            raise ValueError(f"batch row {row}: sequence must contain 1..48 uppercase standard residues")
        triplets, _ = parse_triplets(describe)
        if len(triplets) != len(protein):
            raise ValueError(f"batch row {row}: residue count and triplet count differ")
        globals_out.append([float(value) for value in parse_global(point)])
        triplets_out.append([[float(value) for value in vector] for vector in triplets])
    return globals_out, triplets_out


def validate_file(path: str | Path, *, stage: str, num_classes: int = 3) -> Validation:
    """Validate against the retrieved root model's fixed structural dimensions.

    num_classes is an explicit embedding-index bound, not a biological label map.
    IDs need not be contiguous; duplicates are rejected instead of overwritten.
    """
    if stage not in {"vae", "diffusion"}:
        raise ValueError("stage must be vae or diffusion")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    required = BASE_COLUMNS + (("class",) if stage == "diffusion" else ())
    issues, records = [], []
    header, delimiter, seen, rows_seen = (), None, set(), 0

    def result():
        return Validation(str(path.resolve()), sha, stage, delimiter, header,
                          rows_seen, tuple(records), tuple(issues))

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        issues.append(Issue("ENCODING", 1, "file", "input must be UTF-8"))
        return result()
    # Dialect inference is restricted to the unambiguous required header names.
    # A comma within a descriptor never influences delimiter detection.
    candidates = []
    for candidate in (",", "\t"):
        try:
            first = next(csv.reader(io.StringIO(text, newline=""), delimiter=candidate, strict=True))
        except (StopIteration, csv.Error):
            continue
        if "idx" in first and "protein" in first and "describe" in first:
            candidates.append((candidate, tuple(first)))
    if len(candidates) != 1:
        issues.append(Issue("DELIMITER_HEADER", 1, "header", "cannot uniquely identify comma/tab AMP header"))
        return result()
    delimiter, header = candidates[0]
    if len(set(header)) != len(header):
        issues.append(Issue("DUPLICATE_COLUMN", 1, "header", "column names must be unique"))
    missing, extra = sorted(set(required)-set(header)), sorted(set(header)-set(required))
    if missing:
        issues.append(Issue("MISSING_COLUMN", 1, "header", f"missing required columns: {missing}; never fabricated"))
    if extra:
        issues.append(Issue("UNEXPECTED_COLUMN", 1, "header", f"unexpected columns for stage {stage}: {extra}"))
    if issues:
        return result()
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    next(reader)
    try:
        for row in reader:
            rows_seen += 1
            line = reader.line_num
            start_errors = len(issues)
            if len(row) != len(header):
                issues.append(Issue("COLUMN_COUNT", line, "row", f"expected {len(header)} fields, received {len(row)}"))
                continue
            fields = dict(zip(header, row))
            idx_text, protein = fields["idx"], fields["protein"]
            idx = None
            if not INTEGER.fullmatch(idx_text):
                issues.append(Issue("INVALID_ID", line, "idx", "expected a nonnegative decimal integer ID"))
            else:
                try:
                    idx = int(idx_text)
                except ValueError:
                    issues.append(Issue("INVALID_ID", line, "idx", "integer ID exceeds this Python runtime's conversion limit"))
                else:
                    if idx in seen:
                        issues.append(Issue("DUPLICATE_ID", line, "idx", "duplicate integer ID; no overwrite allowed"))
                    seen.add(idx)
            if not 1 <= len(protein) <= 48:
                issues.append(Issue("SEQUENCE_LENGTH", line, "protein", f"length {len(protein)} outside 1..48; never truncated"))
            if set(protein)-AMINO_ACIDS:
                issues.append(Issue("SEQUENCE_ALPHABET", line, "protein", "requires uppercase 20 standard residues; no silent normalization"))
            triplets, canonical, global_values, label = (), "", (), None
            try:
                triplets, canonical = parse_triplets(fields["describe"])
                if len(triplets) != len(protein):
                    issues.append(Issue("RESIDUE_ALIGNMENT", line, "describe", f"{len(triplets)} triplets for {len(protein)} residues"))
            except ValueError as exc:
                issues.append(Issue("TRIPLET_FORMAT", line, "describe", str(exc)))
            try:
                global_values = parse_global(fields["point_describe"])
            except ValueError as exc:
                issues.append(Issue("GLOBAL_FORMAT", line, "point_describe", str(exc)))
            if stage == "diffusion":
                raw_label = fields["class"]
                if not INTEGER.fullmatch(raw_label):
                    issues.append(Issue("LABEL_FORMAT", line, "class", "requires an explicit integer label; blank/N/unknown are not zero"))
                else:
                    try:
                        label = int(raw_label)
                    except ValueError:
                        issues.append(Issue("LABEL_FORMAT", line, "class", "integer label exceeds this Python runtime's conversion limit"))
                    else:
                        if label >= num_classes:
                            issues.append(Issue("LABEL_RANGE", line, "class", f"index must be in 0..{num_classes-1}; semantic mapping unverified"))
            if len(issues) == start_errors:
                records.append(Record(idx, protein, triplets, global_values, label, line,
                                      tuple(fields.items()), canonical))
    except csv.Error as exc:
        issues.append(Issue("CSV_SYNTAX", reader.line_num, "row", str(exc)))
    if rows_seen == 0:
        issues.append(Issue("EMPTY_DATASET", 1, "file", "header without data records is not a usable dataset"))
    return result()


class ValidatedDataset(Sequence):
    """Map-style CSV/TSV adapter; no torch import or inheritance needed.

    Construction validates the entire file before exposing any row. Use record()
    for typed consumers. __getitem__ supplies the historical field names only;
    actual integration into the research model is deliberately not certified.
    """
    def __init__(self, path, *, stage, num_classes=3):
        self.validation = validate_file(path, stage=stage, num_classes=num_classes)
        if not self.validation.valid:
            raise ContractError(self.validation)

    def __len__(self):
        return len(self.validation.records)

    def record(self, index):
        return self.validation.records[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [record.legacy_fields() for record in self.validation.records[index]]
        return self.validation.records[index].legacy_fields()


def export_csv(dataset: ValidatedDataset, output: str | Path):
    """Write complete valid input only; forbid replacing input/existing output."""
    path = Path(output)
    if path.resolve() == Path(dataset.validation.source):
        raise ValueError("cannot overwrite the input")
    names = BASE_COLUMNS + (("class",) if dataset.validation.stage == "diffusion" else ())
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--stage", choices=["vae", "diffusion"], required=True)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--export-csv", type=Path)
    args = parser.parse_args(argv)
    validation = validate_file(args.input, stage=args.stage, num_classes=args.num_classes)
    report = json.dumps(validation.summary(), ensure_ascii=False, indent=2) + "\n"
    if args.report:
        if args.report.resolve() == args.input.resolve():
            parser.error("report cannot overwrite input")
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(report)
    else:
        print(report, end="")
    if not validation.valid:
        return 2
    if args.export_csv:
        export_csv(ValidatedDataset(args.input, stage=args.stage, num_classes=args.num_classes), args.export_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
