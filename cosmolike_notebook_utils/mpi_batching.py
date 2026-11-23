"""Spread independent jobs over MPI ranks, and group bins into batches.

Two pieces of the MPI derivative drivers that do not depend on a project:
  - pair_coloring and bins_of_color: a greedy coloring of the tomographic
    bins, so that two bins that share a cross spectrum never sit in the same
    batch (one batch of bins is stepped together in one evaluation);
  - task_owners, run_owned_jobs and peak_rss_mb: a fixed assignment of jobs
    to ranks, the loop that runs a rank's jobs, and the peak memory report.

The communicator is passed in, so importing this module does not need mpi4py.
"""
import sys
import time

import numpy as np


def pair_coloring(nbins, pairs):
    """Return a color per bin: the smallest one its lower-numbered partners lack.

    The partners of a bin are the bins it forms a cross pair with. The bins
    are colored in bin order (greedy coloring): bin i takes the smallest
    color (0, 1, 2, ...) that none of its already colored partners j < i
    has. Two bins of one pair therefore never share a color. The order is
    fixed, so the coloring is reproducible.

    Arguments:
      nbins = the number of lens bins.
      pairs = the cross pairs, 0-based (i, j).

    Returns:
      list of int: the color of every bin, length nbins.
    """
    # partners[i] = the set of bins paired with bin i (both directions)
    partners = []
    for _ in range(nbins):
        partners.append(set())
    for a, b in pairs:
        partners[a].add(b)
        partners[b].add(a)

    color = []
    for i in range(nbins):
        # the colors taken by the partners that are already colored
        used = set()
        for j in partners[i]:
            if j < i:
                used.add(color[j])

        # the smallest free color
        c = 0
        while c in used:
            c += 1
        color.append(c)

    return color


def bins_of_color(color, c):
    """Return the 0-based bins of color c, as an int array.

    Arguments:
      color = the color of every bin (pair_coloring).
      c     = the color.

    Returns:
      int array of the bins with color c, in bin order.
    """
    members = []
    for i in range(len(color)):
        if color[i] == c:
            members.append(i)
    return np.array(members, dtype=int)


def task_owners(n_tasks, n_ranks):
    """Return the rank of each task: k mod n_ranks, the last one on the last rank.

    Round robin spreads the jobs evenly. The last job (the second fiducial
    evaluation) is moved to the last rank: with several ranks it then runs
    in another process than the first fiducial evaluation (rank 0), after
    that process has applied its own steps.

    Arguments:
      n_tasks = the number of jobs.
      n_ranks = the number of MPI ranks.

    Returns:
      list of int: the rank of every job.
    """
    owners = []
    for k in range(n_tasks):
        owners.append(k % n_ranks)
    owners[n_tasks - 1] = n_ranks - 1
    return owners


def peak_rss_mb():
    """Return this process's peak memory in MB (the run reports print it).

    Returns:
      float: the peak resident set size (the largest physical memory the
      process has used) in MB.
    """
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # macOS reports bytes, Linux kilobytes
    if sys.platform == "darwin":
        return peak/1.0e6
    return peak*1024.0/1.0e6


def run_owned_jobs(comm, jobs, evaluate):
    """Evaluate this rank's share of a job list and gather every result on rank 0.

    Job k runs on rank task_owners(k): k mod N, the last job on the last
    rank. Each rank evaluates its jobs in list order and prints the time of
    each; comm.gather then sends every rank's {job: result} dict to rank 0,
    which merges them. With one rank (plain python, or mpirun -n 1) this is
    the serial run in list order.

    Arguments:
      comm     = the MPI communicator (MPI.COMM_WORLD).
      jobs     = the job list; every rank must hold the same list. Jobs are
                 hashable (tuples), because they key the result dict.
      evaluate = a function job -> result, run on the owning rank.

    Returns:
      (results, timings) on rank 0: results = {job: result}; timings = one
      dict per rank, in rank order, {"jobs", "seconds", "peak_rss_mb"}.
      (None, None) on every other rank.

    Raises:
      RuntimeError on rank 0 when a job has no result after the gather.
    """
    rank = comm.Get_rank()
    owners = task_owners(n_tasks=len(jobs), n_ranks=comm.Get_size())
    evaluation_start = time.time()

    # mine maps each job of this rank to its result
    mine = {}
    for k in range(len(jobs)):
        if owners[k] == rank:
            job_start = time.time()
            mine[jobs[k]] = evaluate(jobs[k])
            print(f"[rank {rank}] {k + 1}/{len(jobs)} {jobs[k]}: "
                  f"{time.time() - job_start:.1f} s", flush=True)

    timing = {
        "jobs": len(mine),
        "seconds": time.time() - evaluation_start,
        "peak_rss_mb": peak_rss_mb(),
    }

    # gather returns a list of every rank's value on rank 0, None elsewhere
    gathered = comm.gather(mine, root=0)
    timings = comm.gather(timing, root=0)
    if rank != 0:
        return None, None

    results = {}
    for part in gathered:
        results.update(part)

    missing = []
    for job in jobs:
        if job not in results:
            missing.append(job)
    if missing:
        raise RuntimeError(f"no rank returned the jobs {missing}")
    return results, timings
