from __future__ import annotations

import importlib
import inspect
import pkgutil

from .base_plugin import BasePlugin
from . import __path__ as plugins_path


class PluginManager:
    def __init__(self):
        self.plugins = []

    def discover_plugins(self):
        self.plugins.clear()

        for _, module_name, _ in pkgutil.iter_modules(plugins_path):
            if module_name in {"base_plugin", "plugin_manager"}:
                continue

            module = importlib.import_module(f"{__package__}.{module_name}")

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                    self.plugins.append(obj())

        self.plugins.sort(key=lambda p: p.display_name.lower())
        return self.plugins