import numpy as np
import random, time
import matplotlib.pyplot as plt

N = 10000                       # number of devices
ALARM_WINDOW_S = 1.0            # devices wake up in 1s window ###TODO HUH
SLOT_MS = 10                    # slot time (ms)
SLOT_S = SLOT_MS / 1000.0       # slot time (s)
MAX_SIM_S = 3600.0              # at most 1 hour
NUM_SLOTS = MAX_SIM_S / SLOT_S  # number of possible slots to simulate
MAX_ATTEMPTS = 20               # max re-tx attempts
NUM_PREAMBLES = 48              # number of unique preambles TODO is all rach slotted?
BACKOFF_MAX = 2                # max time for backoff in event of collision

# init devices
alarm_times = np.random.rand(N) * ALARM_WINDOW_S
next_attempt = alarm_times.copy()
attempts = np.zeros(N)
success_time = np.full(N, float('inf'))
state = np.zeros(N, dtype=np.int8)  # 0=pending/idle, 1=succeeded, -1=failed

# get indices attempting in current slot 
def indices_attempting(t):
    # return indices where not yet success/failed and next_attempt <= t
    mask = (state == 0) & (next_attempt <= t + 1e-12)
    return np.nonzero(mask)[0]

# reset global arrays with device states
def reset_arrays():
    next_attempt = alarm_times.copy()
    attempts = np.zeros(N, dtype=np.int32)
    success_time = np.full(N, np.nan)
    state = np.zeros(N, dtype=np.int8)

# RACH no modifications 
def simulate_slotted_rach():
    reset_arrays()
    t = 0.0
    slot = 0
    successes_by_slot = []
    collisions_by_slot = []
    succeeded = 0

    while slot < NUM_SLOTS and succeeded < N:
        print(slot)
        t = slot * SLOT_S
        idxs = indices_attempting(t)
        num_attempts = idxs.size
        if num_attempts == 0:
            slot += 1
            continue

        # each attempting device randomly picks a preamble
        preambles = np.random.randint(0,NUM_PREAMBLES-1,size=num_attempts)

        #collisions if count for a single preamble > 1
        # For each preamble value, find indices
        successes_this_slot = 0
        collisions_this_slot = 0
        for p in range(NUM_PREAMBLES):
            if not np.any(preambles == p):
                continue

            # get indicies of attempting nodes using preamble
            preamble_index = idxs[preambles == p]   
            num_nodes_in_preamble = preamble_index.size
            attempts[preamble_index] += 1
            if num_nodes_in_preamble == 1:
                # successful tx
                i = preamble_index[0]
                state[i] = 1
                success_time[i] = t + SLOT_S
                successes_this_slot += 1
                succeeded += 1
            else:
                # collision
                # schedule backoff for each node
                collisions_this_slot += NUM_PREAMBLES
                # fail if max attempts, else backoff based on uniform rv
                timeout = attempts[preamble_index] >= MAX_ATTEMPTS
                if np.any(timeout):
                    state[preamble_index[timeout]] = -1
                retry = preamble_index[~timeout]
                if retry.size > 0:
                    backoff = np.random.rand(retry.size) * BACKOFF_MAX
                    next_attempt[retry] = t + backoff

        successes_by_slot.append(successes_this_slot)
        collisions_by_slot.append(collisions_this_slot)

        slot += 1

    # gather statistics
    succeeded_mask = (state == 1)
    succ_times = success_time[succeeded_mask]
    failed = np.sum(state == -1)
    pending = np.sum(state == 0)
    results = {
        'succeeded': succeeded,
        'failed': int(failed),
        'pending': int(pending),
        'collision_prob': sum(collisions_by_slot) / sum(np.array(successes_by_slot)+np.array(collisions_by_slot)) if sum(successes_by_slot)+sum(collisions_by_slot)>0 else 0.0,
        'succ_times': succ_times,
        'successes_by_slot': successes_by_slot,
        'collisions_by_slot': collisions_by_slot,
        't_end': slot * SLOT_S
    }
    return results

# TODO: same as base rach rn
# RACH with ACB 
def simulate_acb_rach():
    reset_arrays()
    t = 0.0
    slot = 0
    successes_by_slot = []
    collisions_by_slot = []
    succeeded = 0

    while slot < NUM_SLOTS and succeeded < N:
        print(slot)
        t = slot * SLOT_S
        idxs = indices_attempting(t)
        num_attempts = idxs.size
        if num_attempts == 0:
            slot += 1
            continue

        # each attempting device randomly picks a preamble
        preambles = np.random.randint(0,NUM_PREAMBLES-1,size=num_attempts)

        #collisions if count for a single preamble > 1
        # For each preamble value, find indices
        successes_this_slot = 0
        collisions_this_slot = 0
        for p in range(NUM_PREAMBLES):
            if not np.any(preambles == p):
                continue

            # get indicies of attempting nodes using preamble
            preamble_index = idxs[preambles == p]   
            num_nodes_in_preamble = preamble_index.size
            attempts[preamble_index] += 1
            if num_nodes_in_preamble == 1:
                # successful tx
                i = preamble_index[0]
                state[i] = 1
                success_time[i] = t + SLOT_S
                successes_this_slot += 1
                succeeded += 1
            else:
                # collision
                # schedule backoff for each node
                collisions_this_slot += NUM_PREAMBLES
                # fail if max attempts, else backoff based on uniform rv
                timeout = attempts[preamble_index] >= MAX_ATTEMPTS
                if np.any(timeout):
                    state[preamble_index[timeout]] = -1
                retry = preamble_index[~timeout]
                if retry.size > 0:
                    backoff = np.random.rand(retry.size) * BACKOFF_MAX
                    next_attempt[retry] = t + backoff

        successes_by_slot.append(successes_this_slot)
        collisions_by_slot.append(collisions_this_slot)

        slot += 1

    # gather statistics
    succeeded_mask = (state == 1)
    succ_times = success_time[succeeded_mask]
    failed = np.sum(state == -1)
    pending = np.sum(state == 0)
    results = {
        'succeeded': succeeded,
        'failed': int(failed),
        'pending': int(pending),
        'collision_prob': sum(collisions_by_slot) / sum(np.array(successes_by_slot)+np.array(collisions_by_slot)) if sum(successes_by_slot)+sum(collisions_by_slot)>0 else 0.0,
        'succ_times': succ_times,
        'successes_by_slot': successes_by_slot,
        'collisions_by_slot': collisions_by_slot,
        't_end': slot * SLOT_S
    }
    return results

# summarize results
def print_summary(results):
    print("Succeeded:", results['succeeded'])
    print("Penging:", results['pending'])
    print("Failed:", results['failed'])
    print("Collision Probability:", results['collision_prob'])

def plot_cumulative(results, title):
    s = np.array(results['successes_by_slot'])
    cum = np.cumsum(s)
    t = np.arange(len(s)) * SLOT_S
    plt.figure()
    plt.plot(t, cum)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative successes")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, min(200, t[-1]))

def plot_collision_frac(results, title):
    coll = np.array(results['collisions_by_slot'])
    succ = np.array(results['successes_by_slot'])
    attempts = coll + succ
    frac = np.where(attempts>0, coll/attempts, 0.0)
    t = np.arange(len(frac)) * SLOT_S
    plt.figure()
    plt.plot(t, frac)
    plt.xlabel("Time (s)")
    plt.ylabel("Collision fraction")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, min(200, t[-1]))

def plot_latency_hist(results, title):
    sts = np.array(results['succ_times'])
    if sts.size == 0:
        print(title, "- no successes")
        return
    plt.figure()
    plt.hist(sts, bins=100)
    plt.xlabel("Success time (s)")
    plt.ylabel("Count")
    plt.title(title)
    plt.xlim(0, min(200, sts.max()))

########## run ##########
rach = simulate_slotted_rach()
print_summary(rach)

# print results
plot_cumulative(rach, "RACH Cumulative Success")
plot_collision_frac(rach, "RACH Collision Fraction")
plot_latency_hist(rach, "RACH Latency Histogram")

plt.show()