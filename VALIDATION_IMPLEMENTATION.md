# TIC IDs CSV Structure Validation Implementation

## Summary

Successfully implemented the TODO item "check structure of TIC IDs CSV file" in `run_pipeline.py` line 252.

## Changes Made

### 1. Added `validate_tic_ids_csv_structure()` function to `utils.py`

**Location**: `exominer_pipeline/utils.py` (lines 144-211)

**Features**:
- Validates required columns: `tic_id` and `sector_run`
- Checks TIC IDs are numeric (handles both integer and float representations)
- Validates sector_run format follows pattern `X-Y` (e.g., "6-6", "1-39")
- Provides detailed error messages with row numbers for invalid data
- Returns `True`/`False` for validation results
- Raises `SystemExit` for critical structural issues (missing columns, empty file)

### 2. Updated `run_pipeline.py` to use validation

**Location**: `exominer_pipeline/run_pipeline.py` (lines 253-256)

**Changes**:
- Replaced TODO comment with actual validation call
- Added proper import for the validation function
- Added logging messages for validation start/completion

### 3. Validation Logic

The function validates:

1. **Required Columns**: Must have `tic_id` and `sector_run` columns
2. **Non-empty Data**: File must contain at least one row of data
3. **TIC ID Format**: Must be numeric (e.g., 167526485)
4. **Sector Run Format**: Must follow pattern `start-end` (e.g., "6-6", "1-39")

### 4. Error Handling

- **Critical Errors**: Missing columns or empty file → `SystemExit`
- **Validation Errors**: Invalid data formats → Returns `False` with detailed warnings
- **Success**: Valid data → Returns `True` with info message

## Testing

Created comprehensive tests that verify:
- ✅ Valid CSV data passes validation
- ✅ Invalid TIC IDs are detected
- ✅ Invalid sector_run formats are detected
- ✅ Missing columns trigger SystemExit
- ✅ Empty files trigger SystemExit
- ✅ Real CSV files are validated correctly

## Expected CSV Format

```csv
tic_id,sector_run
167526485,6-6
167526485,1-39
239332587,14-60
```

Where:
- `tic_id`: Numeric TIC identifier
- `sector_run`: Sector range in format "start-end"

## Integration

The validation is now automatically called during pipeline execution:
1. TIC IDs are loaded from CSV or command line
2. Structure validation runs automatically
3. Pipeline continues if validation passes
4. Pipeline stops with clear error messages if validation fails

This ensures data integrity before the ExoMiner pipeline processes the TIC IDs.
