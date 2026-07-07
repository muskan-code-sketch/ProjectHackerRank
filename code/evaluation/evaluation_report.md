# Evaluation Report

## Strategies Compared

| strategy | macro exact match | claim_status | issue_type | object_part | severity |
|---|---:|---:|---:|---:|---:|
| text_only | 0.657 | 0.600 | 0.550 | 0.700 | 0.500 |
| final | 0.664 | 0.600 | 0.550 | 0.700 | 0.500 |

`text_only` extracts issue type and object part from the conversation and assumes available evidence supports the claim. `final` adds local image-file validation, prompt-injection risk detection, and user-history risk flags.

## Final Strategy

The submission uses the `final` strategy from `code/main.py`. It is deterministic, dependency-free, reads all required CSVs, checks local image references, and writes the exact output schema.

## Operational Analysis

- Model calls: 0 for sample processing and 0 for test processing.
- Approximate token usage: 0 input and 0 output tokens because no hosted model is called.
- Images processed: 82 referenced test images are checked for local availability.
- Approximate cost: $0.00 for the full test set under the current deterministic configuration.
- Runtime: expected to be under a second for the provided CSVs on a typical laptop.
- TPM/RPM considerations: none for the default path. If a VLM is added later, cache per-image observations by file hash and batch claims by object type to avoid repeated image calls.

## Residual Risk

This baseline cannot truly inspect visual damage semantics. It is a stable evaluable floor and a clean integration point for adding a VLM observation layer without changing the output contract.
