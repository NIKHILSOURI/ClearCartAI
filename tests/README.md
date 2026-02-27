# Tests

Run all commands from the **project root** (`ean_system_new/`).

| Script | Description |
|--------|-------------|
| `test_setup.py` | Verifies PyTorch, Transformers, SAM2, and project imports. Optional: set `TEST_IMAGE_DIR` to a folder with images to test image loading. |
| `test_single_product.py` | Interactive run on one product folder. Usage: `python tests/test_single_product.py <1-5>`. Set `TEST_BASE_DIR` to the parent of your product folders (default: `.`). |
| `test_all_products.py` | Batch interactive run over five product folders. Set `TEST_BASE_DIR` as above. |
| `test_batch_interactive.py` | Batch interactive test script. |

Example:

```bash
# From project root
python tests/test_setup.py

# With optional image-dir test
set TEST_IMAGE_DIR=C:\path\to\images
python tests/test_setup.py
```
