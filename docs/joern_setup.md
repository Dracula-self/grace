# Joern Setup

## Install

Default install:

```bash
bash scripts/install_joern.sh
export JOERN_HOME=$PWD/tools/joern/2.0.42
export PATH=$JOERN_HOME:$PATH
```

If you later obtain the original author-provided archive, place it under:

```text
third_party/joern/joern-cli-2.0.42.zip
```

The installer will prefer the local archive over downloading from GitHub.

## Requirements

- Java 19 is recommended for recent Joern releases
- enough CPU memory for large preprocessing jobs

Local smoke-test settings:

```bash
export JAVA_OPTS="-Xms1G -Xmx8G"
```

Larger platform settings:

```bash
export JAVA_OPTS="-Xms4G -Xmx32G"
```

## Extract Node / Edge Text

Train split:

```bash
python scripts/run_joern_extract.py --dataset devign --split train --joern-home "$JOERN_HOME" --workers 4
```

Test split:

```bash
python scripts/run_joern_extract.py --dataset devign --split test --joern-home "$JOERN_HOME" --workers 4
```

Outputs:

```text
data/cpg/<dataset>/<split>.jsonl
```

Each row contains:

- `id`
- `ast`
- `node`
- `edge`
- `graph_error`

If a function fails to parse, the record is kept and `graph_error` is populated instead of aborting the entire job.

