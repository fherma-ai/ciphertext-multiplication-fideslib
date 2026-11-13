# Ciphertext multiplication on FIDESlib

An answer to `ciphertext-multiplication/ckks@1.0.0` on FHERMA that finishes on
the card: the envelope makes the context, the keys and the ciphertexts with
OpenFHE, and the multiplication runs on the GPU through FIDESlib's interop
layer.

The point is the same, and so is everything the author chooses: these
directories carry the **same `config.jsonc`** as
[ciphertext-multiplication-openfhe](https://github.com/fherma-ai/ciphertext-multiplication-openfhe),
byte for byte. The ring, the scale, the first modulus, the key-switch digit
count, the scaling technique and the security level are therefore identical on
both sides, and what separates the two measurements is the machine and the
library.

| | `solution-s32768-d10` | `solution-s4096-d20` |
| --- | --- | --- |
| point | slots 32768, depth 10, precision 15 | slots 4096, depth 20, precision 20 |
| `ring_dimension` | 65536 | 65536 |
| `batch_size` / `mult_depth` | 32768 / 10 | 4096 / 20 |
| `scale_mod_size` | 50 | 48 |
| residue polynomials per ciphertext | 11 | 21 |

## Three answers per point

`solution-<point>` constructs its device ciphertexts per case;
`solution-<point>-reused` allocates them once in `gpu_init` and loads the
operands into them. The transfers are inside the measured call in both, so
the difference between the two rows is what allocating 2 × (level + 1)
residue polynomials on the device costs per multiplication. Nothing else
differs: same `config.jsonc`, same call, same point.

`solution-<point>-split` is `-reused` with the transfers moved out of the
measured call, into the two optional functions every loop calls around it
(fherma-lang 0.20.0 and later):

| Function | What it does | Reported as |
| --- | --- | --- |
| `fherma_h2d` → `gpu_h2d` | `GetRawCipherText` and `load` of both operands | `h2d_s` |
| `solve_run` → `gpu_mult` | `mult(y, rescale)`, then `cudaDeviceSynchronize` | `seconds`, the score |
| `fherma_d2h` → `gpu_d2h` | `store` and `GetOpenFHECipherText` of the answer | `d2h_s` |

The arithmetic is `-reused`'s, call for call. Its `seconds` is the
multiplication on the card; `seconds + h2d_s + d2h_s` is what `-reused`
measures.

## What is measured

`solve_run` calls `gpu_mult`, and that is one FIDESlib call plus the two
transfers:

```cpp
GpuCiphertext x(gpu, rawA);
GpuCiphertext y(gpu, rawB);
x.mult(y, /*rescale=*/true);   // tensor, key switch, rescale
```

`mult` with `rescale` set performs the rescale when the technique is
`FIXEDMANUAL`, which is what the config declares — so the measured call holds
the same three stages the CPU answer's `EvalMult` + `RescaleInPlace` does.

In `solution-<point>` and `-reused` both transfers are inside the window:
the operands go to the card and the answer comes back as an OpenFHE
ciphertext the envelope can decrypt. For a single multiplication that is a
large share of the measurement rather than a rounding error. `-split` clocks
them apart, so its score is the arithmetic alone; the keys were already on
the card from `gpu_init`.

## The one parameter that had to move

The image carries FIDESlib's own patched OpenFHE 1.5.1, and it refuses the
deep set at a 50-bit scale exactly as the stock 1.5.1 does: the first
measurement aborted with *EstimateLogP() failure: expected sizeP [7], but got
sizeP [6]* before a single case ran. 1.6.1, which the CPU rows are built
with, accepts that scale. So the deep point is measured at a 48-bit scale on
both sides rather than on one, and the pair keeps differing in the device
alone.

## Building it

The first build happens on the fleet: the image needs CUDA and the device link
is an LTO link against FIDESlib's static library, which carries its own
OpenFHE.

```sh
docker run --rm --gpus all -v "$PWD/solution-s32768-d10":/src:ro --entrypoint sh \
  fherma/fideslib:2.1.3 -c "cmake -S /src -B /tmp/b && cmake --build /tmp/b -j"
```
