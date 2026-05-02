# plugins/base_plugin.py
from __future__ import annotations

class BasePlugin:
    plugin_id = "base"
    display_name = "Base Plugin"
    icon_name = None
    tooltip = ""

    def is_enabled(self, controller) -> bool:
        return True

    def launch(self, controller):
        raise NotImplementedError