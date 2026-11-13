# One CKKS ciphertext multiplication: tensor, relinearise, rescale — an encrypted answer on a GPU

Answers `ciphertext_multiplication/ckks@1.0.0` over ciphertexts, with FIDESlib (CKKS) under OpenFHE's envelope.

Yours to write: **gpu.cu** — the circuit, on the card — and the packing in
**solve.cpp**. Everything else is generated from the signature and the config,
and `--update` rewrites it without touching yours.

| | |
| --- | --- |
| secret arguments | a, b — arrive in `run` as OpenFHE ciphertexts |
| public arguments | none — arrive as plain values |
| measured | the `run` call alone: the copy to the card, the circuit, the copy back |
| not measured | `gpu_init`: the device context and the keys, once per point |
| keys | generated and held by the envelope; `gpu_init` reads the evaluation keys out of the context's store |
| parameters | `config.jsonc`: the OpenFHE parameters; the device context is adapted from them |
| builds in | `fherma/fideslib:2.1.3` — FIDESlib 2.1.3 with a patched OpenFHE 1.5.1, CUDA 12.9, compiled for sm_89 and sm_120 |

The context is OpenFHE's, so `config.jsonc` reads exactly as it does for an
OpenFHE answer, and the security level in it is enforced by the library at
context creation. FIDESlib answers CKKS only. The first compile happens on a
machine with a card: the image has no CPU fallback.
