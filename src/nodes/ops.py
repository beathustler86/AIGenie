import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
	def __init__(self, dim, eps=1e-8):
		super().__init__()
		self.eps = eps
		self.scale = nn.Parameter(torch.ones(dim))

	def forward(self, x):
		norm = x.norm(dim=-1, keepdim=True)
		return x / (norm + self.eps) * self.scale

class SinCosEmbedding(nn.Module):
	def __init__(self, dim, max_len=10000):
		super().__init__()
		pe = torch.zeros(max_len, dim)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return self.pe[:x.size(1)].unsqueeze(0)


	def build_pos_embed(self, device=None, dtype=None):
		try:
			cls_type = getattr(self.operations, self.pos_emb_cls)
		except AttributeError:
			raise ValueError(f"[GeneralDIT] Unknown pos_emb_cls: {self.pos_emb_cls}")

		max_len = self.max_frames * self.max_img_h * self.max_img_w
		pos_embed = cls_type(
			dim=self.model_channels,
			max_len=max_len
		)
		return pos_embed






operations = type("Ops", (), {
	"RMSNorm": RMSNorm,
	"Linear": nn.Linear,
	"sincos": SinCosEmbedding  # 🔁 alias matches checkpoint metadata
})

