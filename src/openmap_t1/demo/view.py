from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import gradio as gr

if TYPE_CHECKING:
    from openmap_t1.demo.controller import DemoController


@dataclass
class DemoView(object):
    def _bind_upload_button(self, callback):
        file_output = gr.File(file_types=["nii", "nii.gz"])
        self._upload_button.upload(
            callback, inputs=self._upload_button, outputs=file_output
        )

    def run(self, controller: DemoController) -> None:
        ui = gr.Blocks()
        with ui:
            gr.Markdown("## OpenMAP-T1 Demo")
            self._upload_button = gr.UploadButton("Click to Upload a File")
            self._bind_upload_button(controller.upload_button)

        ui.launch()
