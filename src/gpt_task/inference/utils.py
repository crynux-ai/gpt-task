from __future__ import annotations

from typing import Any, Dict

from gpt_task.config import Config, get_config


def load_model_kwargs(config: Config | None = None) -> Dict[str, Any]:
    if config is None:
        config = get_config()

    res = {}
    if config.hf_cache_dir is not None:
        res["cache_dir"] = config.hf_cache_dir
    if config.proxy is not None and config.proxy.host != "":
        if "://" in config.proxy.host:
            scheme, host = config.proxy.host.split("://", 2)
        else:
            scheme, host = "", config.proxy.host

        proxy_str = f"{scheme}://{config.proxy.username}:{config.proxy.password}@{host}:{config.proxy.port}"
        res["proxies"] = {"http": proxy_str, "https": proxy_str}

    return res
