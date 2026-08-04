"""
Evaluate compression ratio of the tokenizer.
"""

from angstromchat.dataset import parquets_iter_batched


news_text = r"""
(Geneva, Switzerland, August 2, 2026)—In a landmark decision, the International Energy Agency (IEA) announced that global renewable energy capacity has surpassed fossil fuels for the first time in history. The comprehensive report, released early Tuesday morning, details how solar and wind infrastructure expansions in Southeast Asia and Sub-Saharan Africa contributed to a 15% year-over-year growth. However, grid storage remains a critical bottleneck. "We are producing more clean electrons than ever before," stated IEA Director Maria Gonzalez. "The challenge for the next decade is capturing that energy and dispatching it when the sun sets and the wind stops blowing." The announcement triggered a surge in clean energy stocks across global markets, though analysts caution that thousands of miles of new transmission lines will be required to stabilize the grid.
""".strip()


korean_text = r"""
인공지능 모델의 크기가 커짐에 따라 효율적인 토큰화(Tokenization)와 압축률의 중요성이 그 어느 때보다 대두되고 있습니다.

대규모 언어 모델(LLM)은 텍스트를 고유한 토큰 단위로 쪼개어 처리하는데, 이 과정에서 한국어와 같은 교착어는 영어에 비해 상대적으로 많은 토큰을 소모하는 경향이 있습니다. 이는 모델의 추론 속도를 저하시키고 컴퓨팅 리소스 비용을 증가시키는 주요 원인으로 지목되어 왔습니다.

최근 오픈소스 커뮤니티에서는 이러한 비효율성을 극복하기 위해 다국어 코퍼스에 최적화된 BPE(Byte-Pair Encoding) 토크나이저를 자체적으로 구축하려는 시도가 활발히 이루어지고 있습니다. 문자(Character) 단위의 병합을 통해 자주 쓰이는 형태소나 단어를 하나의 서브워드(Subword)로 묶어내면, 컨텍스트 윈도우(Context Window)를 더욱 효율적으로 사용할 수 있기 때문입니다.
""".strip()


code_text = r"""
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # calculate attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        
        y = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
""".strip()


math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}

\begin{document}

\section*{The Gaussian Integral}

The Gaussian integral, also known as the Euler-Poisson integral, is the integral of the Gaussian function $e^{-x^2}$ over the entire real line. It is evaluated as:
\[
\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}
\]

\subsection*{Proof via Polar Coordinates}
Let $I$ be the integral in question:
\[
I = \int_{-\infty}^{\infty} e^{-x^2} \, dx
\]
Squaring $I$ allows us to express it as a double integral over $\mathbb{R}^2$:
\[
I^2 = \left( \int_{-\infty}^{\infty} e^{-x^2} \, dx \right) \left( \int_{-\infty}^{\infty} e^{-y^2} \, dy \right) = \iint_{\mathbb{R}^2} e^{-(x^2 + y^2)} \, dx \, dy
\]
Transforming to polar coordinates, where $x = r \cos \theta$, $y = r \sin \theta$, and the area element is $dx \, dy = r \, dr \, d\theta$:
\[
I^2 = \int_{0}^{2\pi} \int_{0}^{\infty} e^{-r^2} r \, dr \, d\theta
\]
The inner integral can be solved using substitution $u = r^2$, $du = 2r \, dr$:
\[
\int_{0}^{\infty} r e^{-r^2} \, dr = \frac{1}{2} \int_{0}^{\infty} e^{-u} \, du = \frac{1}{2} \left[ -e^{-u} \right]_0^\infty = \frac{1}{2}
\]
Since $e^{-x^2}$ is positive everywhere, $I$ must be positive. Thus, $I = \sqrt{\pi}$.

\end{document}
""".strip()


science_text = r"""
General relativity fundamentally altered our understanding of gravitation, conceptualizing it not as a classical force acting across a distance, but as a geometric consequence of spacetime curvature induced by mass and energy. The Einstein field equations mathematically articulate this relationship, equating the Einstein tensor—a construct of the metric tensor and its derivatives representing local spacetime geometry—to the stress-energy tensor, which encodes the density and flux of energy and momentum. One of the most profound predictions of this framework is the existence of black holes, regions where spacetime curvature becomes so extreme that the escape velocity exceeds the speed of light. The boundary of such a region, the event horizon, marks a surface of no return. Recent interferometric observations of supermassive black holes at the centers of galaxies have provided unprecedented direct empirical validation of these theoretical constructs, mapping the photon ring and the shadow cast by the event horizon against the backdrop of an accretion disk's hot, orbiting plasma.
""".strip()



train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("train", train_text),
]

if val_text:
    all_text.append(("val", val_text))


# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    if tokenizer_name == "gpt2":
        tokenizer = 
    elif tokenizer_name == "gpt4":
        tokenizer =
    else:
        tokenizer =
