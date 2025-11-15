import numpy as np
import random, time
import matplotlib.pyplot as plt

N = 10000                       # number of devices
ALARM_WINDOW_S = 1.0            # devices wake up uniformly within 1 second
SLOT_MS = 10                    # slot time (ms)
SLOT_S = SLOT_MS / 1000.0       # slot time (s)
MAX_SIM_S = 600.0              # at most 1 hour
NUM_SLOTS = MAX_SIM_S / SLOT_S  # number of possible slots to simulate
MAX_ATTEMPTS = 20               # max re-tx attempts
NUM_PREAMBLES = 48              # number of unique preambles TODO is all rach slotted?
BACKOFF_MAX = 2                # max time for backoff in event of collision

# RACH no modifications 
def simulate_slotted_rach():
    alarm_times = np.random.rand(N) * ALARM_WINDOW_S
    next_attempt = alarm_times.copy()
    attempts = np.zeros(N, dtype=np.int32)
    success_time = np.full(N, np.nan)
    state = np.zeros(N, dtype=np.int8)

    t = 0.0
    slot = 0
    successes_by_slot = []
    collisions_by_slot = []
    succeeded = 0

    while slot < NUM_SLOTS and succeeded < N:
        print(slot)
        t = slot * SLOT_S
        mask = (state == 0) & (next_attempt <= t + 1e-12)
        idxs = np.nonzero(mask)[0]
        num_attempts = idxs.size
        if num_attempts == 0:
            slot += 1
            continue

        # each attempting device randomly picks a preamble
        preambles = np.random.randint(0,NUM_PREAMBLES,size=num_attempts)

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
    acb_barring_time = 10.0# TBD
    acb_detection_window = 1.0
    acb_collision_treshold = 0.30   # ACB if collision frac > 0.30
    acb_allowed_digit = 0           # IDs ending with are can tx in barring time window

    alarm_times = np.random.rand(N) * ALARM_WINDOW_S
    next_attempt = alarm_times.copy()
    attempts = np.zeros(N, dtype=np.int32)
    success_time = np.full(N, np.nan)
    state = np.zeros(N, dtype=np.int8)

    t = 0.0
    slot = 0
    successes_by_slot = []
    collisions_by_slot = []
    succeeded = 0

    acb_until = -1.0                # initialize ACB end time to negative
    allowed_ids = np.array([ (i % 10) == acb_allowed_digit for i in range(N) ])
    recent_collisions = np.zeros(int(max(1, acb_detection_window / SLOT_S)), dtype=np.int32)
    recent_attempts = np.zeros(int(max(1, acb_detection_window / SLOT_S)), dtype=np.int32)

    while slot < NUM_SLOTS and succeeded < N:
        print(slot)
        t = slot * SLOT_S
        mask = (state == 0) & (next_attempt <= t + 1e-12)
        idxs = np.nonzero(mask)[0]
        num_attempts = idxs.size
        if num_attempts == 0:
            slot += 1
            continue
        
        if t < acb_until:
            # nodes without allowed_ids get delayed
            barred_mask = ~allowed_ids[idxs]
            if barred_mask.any():
                barred_idxs = idxs[barred_mask]
                next_attempt[barred_idxs] = t + acb_barring_time + np.random.rand(barred_idxs.size)
                attempts[barred_idxs] += 1
                # remove from current attempt set
                idxs = idxs[~barred_mask]
                num_attempts = idxs.size
                if num_attempts == 0:
                    # no nodes are txing
                    successes_by_slot.append(0)
                    collisions_by_slot.append(0)
                    slot += 1
                    continue

        # each attempting device randomly picks a preamble
        preambles = np.random.randint(0,NUM_PREAMBLES,size=num_attempts)

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

        recent_attempts[slot % len(recent_attempts)] = successes_this_slot + collisions_this_slot
        recent_collisions[slot % len(recent_collisions)] = collisions_this_slot
        total_recent_attempts = recent_attempts.sum()
        total_recent_collisions = recent_collisions.sum()
        collision_frac = (total_recent_collisions / total_recent_attempts) if total_recent_attempts>0 else 0.0

        if (collision_frac > acb_collision_treshold) and (t > acb_until):
            acb_until = t + acb_barring_time

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
    plt.xlim(0, t[-1])

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
    plt.xlim(0, t[-1])

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
    plt.xlim(0, sts.max())

########## run ##########
random.seed(42)
np.random.seed(42)
rach_base = simulate_slotted_rach()
print_summary(rach_base)
plot_cumulative(rach_base, "RACH Cumulative Success")
plot_collision_frac(rach_base, "RACH Collision Fraction")
plot_latency_hist(rach_base, "RACH Latency Histogram")

random.seed(42)
np.random.seed(42)
rach_acb = simulate_acb_rach()
print_summary(rach_acb)
plot_cumulative(rach_acb, "RACH Cumulative Success")
plot_collision_frac(rach_acb, "RACH Collision Fraction")
plot_latency_hist(rach_acb, "RACH Latency Histogram")

plt.show()