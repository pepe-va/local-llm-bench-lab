# models/

Favorite GGUF models, kept at hand. Everything here except this README is gitignored.

- `llb models pull <ref> --favorite` downloads into this folder as a real file (not a symlink to the Hugging Face cache).
- You can also drop your own `.gguf` files here.
- `llb models prune` only cleans the Hugging Face cache and never touches this folder.

Model resolution order for a `source:` in an experiment: `models/` → Hugging Face cache → download.
