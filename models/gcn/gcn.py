import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..gcncaptioning_model import CaptioningModel


class GcnTransformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(GcnTransformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, selfbbox, bbox, seq=None, mode='forward'):
        enc_output, mask_enc = self.encoder(
            images, selfbbox, bbox)
        dec_output = self.decoder(seq, enc_output, mask_enc, mode)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, selfbbox, bbox, seq, state=None, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(
                    visual, selfbbox, bbox)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full(
                        (visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full(
                        (visual[0].shape[0], 1), self.bos_idx).long()

                return self.decoder(it, self.enc_output, self.mask_enc, state=state, mode=mode)

            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc, state=state, mode=mode)
