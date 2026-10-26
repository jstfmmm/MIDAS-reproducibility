# Configuration templates

The YAML files in this directory are upstream EoR templates retained for the original `run.py` workflow. They reference local paths such as `data/datasets/`, `data/model/`, and `output/`; those paths are intentionally empty in the clean repository.

All checked-in `api_key` fields are `null`. Web retrieval reads `SERPER_API_KEY` (or `SERPAPI_API_KEY` for SerpAPI) from the environment. Review model IDs, dataset paths, GPU IDs, and cache paths before running a template.
