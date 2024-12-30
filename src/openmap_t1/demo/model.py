from dataclasses import dataclass
from typing import Optional

from openmap_t1.utils import UNetModels, get_device, load_models


@dataclass
class DemoModel(object):
    _models: Optional[UNetModels] = None

    def __post_init__(self):
        device = get_device()
        self._models = load_models(device)
