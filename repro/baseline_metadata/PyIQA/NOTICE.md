# PyIQA MUSIQ adapter attribution

Required Notice: Copyright (c) 2022 Chaofeng Chen

`repro/musiq_chunked.py` follows `MultiHeadAttention.forward` in PyIQA 0.1.16,
`pyiqa/archs/musiq_arch.py`, reimplemented by Chaofeng Chen. The adaptation splits
queries while retaining all keys, the same finite mask and the same softmax.
It adds inference-mode enforcement and records source and adapter hashes.
The installed package and original evaluator snapshot are unchanged.

Source: <https://github.com/chaofengc/IQA-PyTorch>

The installed distribution ships the PolyForm Noncommercial License 1.0.0 in
`LICENSE` and an additional `LICENSE-S-Lab`. Both files are preserved verbatim
here. This notice does not assign a license to the rest of the SCA project.

MUSIQ: Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar and Feng Yang,
"MUSIQ: Multi-scale Image Quality Transformer," ICCV 2021.
