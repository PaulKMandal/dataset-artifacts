# Table Audit Checklist

## Goal

Verify every table in the current preprint and determine whether AddSent/AddOneSent results are mislabeled.

## Audit steps

1. Locate the exact script or notebook that produced each table.
2. Locate the exact metrics files used for each table.
3. Locate the exact dataset path used for each evaluation.
4. Record dataset hash, number of examples, and file basename.
5. Regenerate the table using only scripted aggregation.
6. Compare regenerated numbers against manuscript numbers.
7. Flag any mismatch.
8. Confirm whether a table caption names the correct evalset.
9. Save the audit result in `results/logs/table_audit.md`.

## Specific issue to check

The current manuscript appears to report AddSent baseline around 53.7 EM / 60.7 F1 and AddOneSent baseline around 63.1 EM / 70.3 F1 in earlier tables, but later tables may label these values inconsistently. This must be resolved before any writing.

## Required output

For each table:

```text
Table number:
Caption evalset:
Actual evalset from metrics:
Dataset file:
Dataset hash:
Num examples:
Numbers reproduced exactly: yes/no
Mismatch details:
Conclusion:
```
