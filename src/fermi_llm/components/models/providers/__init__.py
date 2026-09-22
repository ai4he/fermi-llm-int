"""Built-in model providers. Importing this package registers all of them."""

from . import (codex_cli, gemini, huggingface_local, openai_api, template,
               vllm_local, vllm_managed, vllm_remote)  # noqa: F401
