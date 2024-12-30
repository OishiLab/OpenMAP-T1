from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

from openmap_t1.commands.parcellation import run_parcellation
from openmap_t1.utils import get_device

if TYPE_CHECKING:
    from openmap_t1.demo.model import DemoModel
    from openmap_t1.demo.view import DemoView


@dataclass
class DemoController(object):
    model: DemoModel
    view: DemoView

    def upload_button(self, files: Union[str, List[str]]) -> None:
        if isinstance(files, str):
            return run_parcellation(nii_path=pathlib.Path(files))
        elif isinstance(files, list):
            for file in files:
                run_parcellation(
                    nii_path=pathlib.Path(file),
                    models=self.model._models,
                    device=get_device(),
                )

    def run(self) -> None:
        self.view.run(controller=self)
