## Summary

What does this change, and why?

## Checklist

- [ ] `bash tools/check.sh` passes locally (Ruff, isort, Black, mypy, pytest)
- [ ] New/changed public functions have type annotations and, where the
      behavior isn't obvious from the signature, a docstring (array shapes,
      units, point/edge conventions)
- [ ] Tests added or updated for the behavior change, if applicable
- [ ] `docs/reproduction.md` updated, if this changes a command, flag, output
      file, or reproduction step
- [ ] No `pickle.load`/`pickle.dump` added on a user- or CLI-controlled path
      (see `SECURITY.md`)

## Related issues

Closes #
