from typing import Optional

from transformers import PretrainedConfig


class UNetConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(
        self, ch_in: Optional[int] = None, ch_out: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ch_in = ch_in
        self.ch_out = ch_out
