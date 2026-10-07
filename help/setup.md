# Setting up your H100

Two steps. Do them before the meeting.

---

## 1. Install the CLI and sign in

We rent GPUs from Autoresearch. Everything goes through one tool, `autor`.

```sh
curl -fsSL https://autoresearch.sfcompute.com/cli.sh | bash
autor login
```

> **Use the same email you signed up to ML Systems with.**
>
> Your GPU time comes out of the group's shared credit, and you only reach
> that credit if your Autoresearch account matches the email we have for you.
> Sign up with a different address and you get an account with no credit,
> which looks exactly like everything working right up until you try to start
> a machine.
>
> Already registered with something else? Tell Alazar. Thirty second fix, but
> only if we know.

Check it:

```sh
autor whoami      # should print a workspace
autor billing     # should print the group's credit
```

If `autor billing` complains about an `org:read` scope, you're signed in but
not attached to the group yet. That's the email problem above.

---

## 2. Run something

```sh
bench/run.sh 2 --m 4096 --k 4096 --n 4096
```

The first run creates an H100 named `$(whoami)-pallas` with JAX already on it,
which takes a minute or two. After that it reuses the machine and wakes it if
it's asleep. Nothing to install.

Set `MLSYS_NODE` if you want a different name.

### Turn it off

```sh
autor node stop $(whoami)-pallas -y
```

**$0.066 a minute**, about four dollars an hour — the meter runs whenever the
machine is up, not just when something is executing. A 90 minute session is
roughly six dollars; a machine left on overnight is forty. Your files are kept
when you stop it, so there is no reason not to.

It also stops itself after 15 minutes idle, but don't rely on that.

---

## What's on the machine

Verified, not quoted from docs:

| | |
| --- | --- |
| GPU | H100 80GB HBM3, compute capability 9.0 |
| Driver | 595.91.07 |
| CUDA | 12.9 |
| Python | 3.12.3 |
| JAX | 0.11.1 |
| `ncu` | present, but counters need a node made with `--profiling`, which is 8x-shapes only |
| `nsys` | not installed |

Your home is `/home/dev` and you are user `dev`, not root.

A 4096³ bf16 matmul on this machine runs at **678 TFLOP/s**, about 69% of the
989 TFLOP/s dense peak. That's the realistic ceiling, not the theoretical one.
