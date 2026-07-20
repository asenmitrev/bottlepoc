# NOTICE

## Dataset

This project trains on the **"Bottle Defects"** dataset (v28), used instead
of the originally-scoped "Bottle Caps" (`bottle-caps-p3mai`) dataset because
a labeled export of it was already present on disk with the identical
5-class taxonomy (Broken Cap, Broken Ring, Good Cap, Loose Cap, No Cap) and
an existing train/valid/test split. See `README.md` -> "Dataset provenance"
for the full explanation of this substitution.

License: **CC BY 4.0** (commercial use OK, attribution required).

Citation (built from the dataset's own `README.dataset.txt` /
`README.roboflow.txt` export metadata, since the Roboflow Universe project
page returned HTTP 403 to automated fetches when this was written — verify
the canonical BibTeX at the URL below if you need the auto-generated form):

```
@misc{ bottle-defects-o4gnx_dataset,
  title = { Bottle Defects Dataset },
  type = { Open Source Dataset },
  author = { Product Defect Detection },
  howpublished = { \url{ https://universe.roboflow.com/product-defect-detection/bottle-defects-o4gnx } },
  url = { https://universe.roboflow.com/product-defect-detection/bottle-defects-o4gnx },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2022 },
  month = { nov },
  note = { v28, exported 2023-02-07 },
}
```

## Model

**RF-DETR** (Nano / Small variants) — Apache License 2.0.
Source: https://github.com/roboflow/rf-detr

No AGPL/GPL-licensed components are used in this project.
