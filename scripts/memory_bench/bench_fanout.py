"""Stage 3/4 fan-out simulation (no Neo4j / LLM needed).

Modes
-----
gather_all : what annotation/entity_extraction do TODAY. asyncio.gather() over every node; the
             per-node context/prompt/messages are built BEFORE the LLM semaphore is taken.
sem_first  : the FIX (WP2). The LLM semaphore is taken first; nothing heavy is built before it.
bounded    : reference only. A fixed pool of workers pulling from a queue.

Usage:  python scripts/memory_bench/bench_fanout.py <mode> <n_nodes>
Measured (macOS, N=10000): gather_all +701 MB | sem_first +15 MB | bounded +4 MB, same wall clock (+5%).
This script measures RSS with `ps`; it is a *relative* comparison between modes, not an absolute figure.
"""

import asyncio
import os
import resource
import subprocess
import sys
import time

N = int(sys.argv[2])
MODE = sys.argv[1]
CATALOG = "model catalog line with docs and fields. " * 700  # ~30 KB, like build_catalog_block()
TEMPLATE = "SYSTEM PROMPT ... {catalog_block} ... END"


def rss():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1024


async def fetch_ctx(i):  # Neo4j read: fast (I/O) compared to the LLM
    await asyncio.sleep(0.001)
    return {
        "info_units": [{"title": f"t{i}", "description": ("lorem ipsum " * 900) + str(i)}]
    }  # ~10 KB text


async def process(i, sem_neo, sem_llm):
    async with sem_neo:
        ctx = await fetch_ctx(i)
    system = TEMPLATE.format(
        catalog_block=CATALOG + str(i)
    )  # new ~30 KB str per node (as in _decide_model)
    human = (
        "<node_context>"
        + "".join(f"<d>{u['description']}</d>" for u in ctx["info_units"])
        + "</node_context>"
    )
    msgs = [system, human]  # noqa: F841 - kept alive on purpose: models memory held while waiting
    async with sem_llm:
        await asyncio.sleep(0.02)  # LLM latency
    return {"node_id": i, "error": None}


async def process_sem_first(i, sem_neo, sem_llm):
    async with sem_llm:  # take the LLM slot BEFORE fetching / building anything
        async with sem_neo:
            ctx = await fetch_ctx(i)
        system = TEMPLATE.format(catalog_block=CATALOG + str(i))
        human = (
            "<node_context>"
            + "".join(f"<d>{u['description']}</d>" for u in ctx["info_units"])
            + "</node_context>"
        )
        msgs = [system, human]  # noqa: F841 - kept alive on purpose: models memory held while waiting
        await asyncio.sleep(0.02)
    return {"node_id": i, "error": None}


async def sem_first():
    sem_neo, sem_llm = asyncio.Semaphore(10), asyncio.Semaphore(8)
    return await asyncio.gather(
        *[process_sem_first(i, sem_neo, sem_llm) for i in range(N)], return_exceptions=True
    )


async def gather_all():
    sem_neo, sem_llm = asyncio.Semaphore(10), asyncio.Semaphore(8)
    return await asyncio.gather(
        *[process(i, sem_neo, sem_llm) for i in range(N)], return_exceptions=True
    )


async def bounded():
    sem_neo, sem_llm = asyncio.Semaphore(10), asyncio.Semaphore(8)
    q = asyncio.Queue()
    for i in range(N):
        q.put_nowait(i)
    out = []

    async def worker():
        while True:
            try:
                i = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            out.append(await process(i, sem_neo, sem_llm))

    await asyncio.gather(*[worker() for _ in range(16)])  # 2x LLM concurrency
    return out


base = rss()
t = time.time()
asyncio.run({"gather_all": gather_all, "bounded": bounded, "sem_first": sem_first}[MODE]())
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f"{MODE:11} N={N:6}: peak RSS +{peak - base:6.0f} MB   ({time.time() - t:.1f}s)")
