# meeting-1 — three Pallas kernels

Three kernels to write on an H100, in [Pallas](https://docs.jax.dev/en/latest/pallas/).
Short, medium, large. Each one teaches something the last one didn't.

```
kernel/kernel1.py   rmsnorm     grid, BlockSpec, refs, reductions
kernel/kernel2.py   matmul      tiling over K, tensor cores, block sizes
kernel/kernel3.py   attention   fusing two matmuls, online softmax, carried state
```

Each file is an empty function. Write it.

## Running them

```sh
bench/run.sh 1 --m 8192 --n 4096
bench/run.sh 2 --m 4096 --k 4096 --n 4096
bench/run.sh 3 --b 4 --s 4096 --h 16 --d 128
```

That creates your H100 if you don't have one, copies your kernel up, checks it
against a reference, and times it. Pass anything else your kernel takes with
`--kw`:

```sh
bench/run.sh 2 --m 8192 --k 8192 --n 8192 --kw block_m=128 --kw block_k=64
```

Set `MLSYS_NODE` to use a node by a different name. Setup is in
[help/setup.md](help/setup.md).

## What you're aiming at

Measured on one H100 80GB, bf16, clocks locked. "first attempt" is a correct
but unconsidered implementation — it's what you should expect to beat.

| | first attempt | xla/cublas | floor | bound |
| --- | ---: | ---: | ---: | --- |
| rmsnorm `M=8192 N=4096` | 725 µs | 110 µs | 40 µs | memory |
| matmul `4096³` | 246 µs | 213 µs | 139 µs | compute |
| attention `B=4 S=4096 H=16 D=128` | 1825 µs | 3536 µs | 556 µs | compute |

Three different situations, on purpose:

- **rmsnorm** moves more bytes than it does arithmetic, so the only thing that
  matters is whether you're saturating memory bandwidth. A first attempt gets
  5% of the roofline and loses to XLA by 6x. Finding the 6x is the exercise.
- **matmul** is the opposite — enough arithmetic to be compute-bound, so this
  is about keeping the tensor cores fed. cuBLAS is a serious opponent and you
  can get close to it.
- **attention** is neither thing alone. The naive version builds the whole
  `S x S` score matrix; at `S=16384` that's 69 GB and it simply OOMs. Your
  kernel is the only thing that runs at all.

The floor is physics: 989 TFLOP/s bf16, 3.35 TB/s HBM3. Nobody reaches it.

## Layout

```
kernel/   the three files you edit
bench/    run.sh drives the H100, bench.py does the measuring
help/     setup, debugging, and the Pallas traps worth knowing
```
