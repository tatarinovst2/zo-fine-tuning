$ErrorActionPreference = "Stop"

$CHECKPOINTS_DIR = ""

# Collect forwarded args as an array
$FORWARD_ARGS = @()

# Parse arguments

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--checkpoints_dir" {
            $CHECKPOINTS_DIR = $args[$i + 1]
            $i++
        }
        "--output_path" {
            Write-Host "Do not pass --output_path manually."
            exit 1
        }
        "--output_dir" {
            Write-Host "Do not pass --output_dir manually."
            exit 1
        }
        default {
            $FORWARD_ARGS += $args[$i]

            # If next token is a value (not starting with --), also include it
            if (($i + 1) -lt $args.Count -and $args[$i + 1] -notmatch "^--") {
                $FORWARD_ARGS += $args[$i + 1]
                $i++
            }
        }
    }
}

if (-not $CHECKPOINTS_DIR) {
    Write-Host "Missing --checkpoints_dir"
    exit 1
}

$OUTPUT_DIR = $CHECKPOINTS_DIR

# Extract DATA_FILE from forwarded args

$DATA_FILE = ""

for ($i = 0; $i -lt $FORWARD_ARGS.Count; $i++) {
    if ($FORWARD_ARGS[$i] -eq "--data_file") {
        $DATA_FILE = $FORWARD_ARGS[$i + 1]
        break
    }
}

if (-not $DATA_FILE) {
    Write-Host "Missing --data_file"
    exit 1
}

$DATASET_NAME = Split-Path (Split-Path $DATA_FILE -Parent) -Leaf
$RUN_NAME = Split-Path $CHECKPOINTS_DIR -Leaf

Write-Host "Dataset: $DATASET_NAME"
Write-Host "Run name: $RUN_NAME"
Write-Host "Output dir: $OUTPUT_DIR"

# Iterate over checkpoints

$checkpoints = Get-ChildItem -Path $CHECKPOINTS_DIR -Directory |
    Sort-Object {
        if ($_.Name -match '(\d+)$') {
            [int]$matches[1]
        } else {
            0
        }
    }

foreach ($ckpt in $checkpoints) {

    $CKPT_PATH = $ckpt.FullName
    $CKPT_NAME = $ckpt.Name

    $OUTPUT_PATH = Join-Path $OUTPUT_DIR "$RUN_NAME`__$CKPT_NAME`__$DATASET_NAME.jsonl"

    Write-Host "======================================================"
    Write-Host "Checkpoint: $CKPT_NAME"
    Write-Host "Output:     $OUTPUT_PATH"
    Write-Host "======================================================"

    $FULL_ARGS = @(
        "--model_name_or_path", $CKPT_PATH,
        "--output_path", $OUTPUT_PATH
    ) + $FORWARD_ARGS

    Write-Host ("python pipeline/inference.py " + ($FULL_ARGS -join " "))

    & python pipeline/inference.py @FULL_ARGS
}
