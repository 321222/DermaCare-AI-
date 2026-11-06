# Local training data

Place images you are authorized to use in the matching folders:

```text
dataset/
  Chickenpox/
  Measles/
  Monkeypox/
  Normal/
```

Accepted formats are JPG, JPEG, and PNG. Keep this data private; the repository ignores dataset contents by default. Do not use identifiable patient images without appropriate legal basis, consent, de-identification, and secure handling. The included split is stratified by image, not patient; repeated people or near-duplicate images can leak across splits. For any serious evaluation, use patient-level separation, a representative external test set, expert labels, and clinical validation.
