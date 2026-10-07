# When it breaks

Run `help/debug.sh` first — it checks the whole chain and tells you which link
is broken. If every line says ok, it's your kernel, and it's one of these.

**`not written yet`** — the stub is still raising. That's the starting state.

**`exceeds available shared memory`** — your block doesn't fit in 232 KB.
Shrink it, or see [`pallas.md`](pallas.md).

**`Unimplemented primitive ... slice`** — you indexed an array with a scalar
inside the kernel. Use a reduction.

**`WRONG` with a small error** — usually accumulating in bf16 instead of
float32, or a boundary tile handled differently from the rest.

**`WRONG` with nonsense (inf, nan)** — in anything with an exponential,
subtract the max *before* exponentiating, not after.

**`WRONG SHAPE`** — the block specs already describe the layout. If you
reshaped inside the kernel, don't.

**Correct but slow** — that's the exercise, not a bug. Change the block sizes,
then `num_warps`, then `num_stages`. Pass them with `--kw`:

```sh
bench/run.sh 2 --m 4096 --k 4096 --n 4096 --kw block_m=64 --kw warps=4
```

**Worked yesterday, not today** — the machine stopped and woke. Files are
kept. `help/debug.sh` confirms it's awake.

**Something else** — paste the whole error in the group chat.
