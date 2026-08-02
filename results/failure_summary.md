# Failure Analysis Summary

> **Note**: Failure modes are heuristically inferred based on image processing rules and serve as likely candidates rather than definitive ground truth.

| Heuristically Inferred Failure Mode | Count | Likely Cause |
|---|---|---|
| Ambiguous / Indeterminate | 7 | Various |
| Complex Textured Background | 6 | False texture cues disrupt the classifier |
| Strong Glare / Overexposure | 4 | Bright spots obscure manipulation artifacts |
| Heavy JPEG Compression (Weak ELA Signal) | 3 | ELA residual signal is heavily suppressed |
