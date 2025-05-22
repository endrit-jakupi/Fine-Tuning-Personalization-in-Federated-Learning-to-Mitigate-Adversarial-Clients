import torch, random, misc
import math
from itertools import combinations


def average(_, vectors):
    return torch.stack(vectors).mean(dim=0)


def trmean(aggregator, vectors):
    if aggregator.nb_byz == 0:
        return torch.stack(vectors).mean(dim=0)
    return torch.stack(vectors).sort(dim=0).values[aggregator.nb_byz:-aggregator.nb_byz].mean(dim=0)


def median(_, vectors):
    return torch.stack(vectors).quantile(q=0.5, dim=0)
    #return torch.stack(vectors).median(dim=0)[0]


def geometric_median_old(aggregator, vectors):
    return misc.smoothed_weiszfeld(aggregator.nb_workers, vectors)


def geometric_median(aggregator, vectors):
    return misc.smoothed_weiszfeld2(aggregator.nb_workers, vectors)


def krum(aggregator, vectors):
    #JS: Compute all pairwise distances
    distances = misc.compute_distances(vectors)
    #JS: return the vector with smallest score
    return misc.get_vector_best_score(vectors, aggregator.nb_byz, distances)

def krum_old(aggregator, vectors): ##old krum
    #JS: Compute all pairwise distances
    distances = compute_distances(vectors)
    #JS: return the vector with smallest score
    result = get_vector_best_score(vectors, aggregator.nb_byz, distances)
    return result


#JS: used for Krum, Multi-Krum , and MDA aggregators
def compute_distances(vectors):
    """ Compute all pairwise distances between vectors"""
    distances = dict()
    all_pairs = list(combinations(range(len(vectors)), 2))
    for (x,y) in all_pairs:
        dist = vectors[x].sub(vectors[y]).norm().item()
        if not math.isfinite(dist):
            dist = float('inf')
        distances[(x,y)] = dist
    return distances

#JS: used for Krum aggregator
def get_vector_best_score(vectors, nb_byz, distances):
    """ Get the vector with the smallest score."""

    #JS: compute the scores of all vectors
    min_score = float('inf')

    for worker_id in range(len(vectors)):
        distances_squared_to_vector = list()

        #JS: Compute the distances of all other vectors to vectors[worker_id]
        for neighbour in range(len(vectors)):
            if neighbour != worker_id:
                dist = distances.get((min(worker_id, neighbour), max(worker_id, neighbour)), 0)
                distances_squared_to_vector.append(dist**2)

        distances_squared_to_vector.sort()
        score = sum(distances_squared_to_vector[:len(vectors) - nb_byz - 1])

        #JS: update min score
        if score < min_score:
            min_score, min_index = score, worker_id

    #JS: return the vector with smallest score
    return vectors[min_index]



def multi_krum(aggregator, vectors):
    #JS: k is the number of vectors to average in the end
    k = aggregator.nb_workers - aggregator.nb_byz
    #JS: Compute all pairwise distances
    distances = misc.compute_distances(vectors)
    #JS: get scores of vectors, sorted in increasing order
    scores = misc.get_vector_scores(vectors, aggregator.nb_byz, distances)
    best_vectors = [vectors[worker_id] for _, worker_id in scores[:k]]
    #JS: return the average of the k vectors with lowest scores
    return torch.stack(best_vectors).mean(dim=0)


def nearest_neighbor_mixing(aggregator, vectors, numb_iter=1):
    vectors = torch.stack(vectors)
    for _ in range(numb_iter):
        # SY: Replace every vector by the average of its nearest neighbors
        vectors = misc.average_nearest_neighbors(vectors, aggregator.nb_byz)

    return robust_aggregators[aggregator.second_aggregator](aggregator, torch.unbind(vectors))


def server_clip(aggregator, vectors):
    magnitudes = [(vector.norm().item(), vector_id) for vector_id, vector in enumerate(vectors)]
    magnitudes.sort(key=lambda x:x[0])
    if aggregator.nb_byz < int(aggregator.nb_workers / 3):
        cut_off_value = aggregator.nb_workers - int(aggregator.nb_byz * (1 + aggregator.nb_byz / (aggregator.nb_workers - 2 * aggregator.nb_byz)))
    else:
        cut_off_value = aggregator.nb_workers - aggregator.nb_byz

    f_largest = magnitudes[cut_off_value:]
    clipping_threshold = magnitudes[cut_off_value - 1][0]
    for _, vector_id in f_largest:
        vectors[vector_id] = misc.clip_vector(vectors[vector_id], clipping_threshold)
    return robust_aggregators[aggregator.aggregator_name](aggregator, vectors)

def nearest_neighbor_mixing_old(aggregator, vectors, numb_iter=1):
    for _ in range(numb_iter):
        mixed_vectors = list()
        for vector in vectors:
            #JS: Replace every vector by the average of its nearest neighbors
            mixed_vectors.append(average_nearest_neighbors(vectors, aggregator.nb_byz, vector))
        vectors = mixed_vectors

    return robust_aggregators[aggregator.second_aggregator](aggregator, vectors)

#JS: Compute the average of the n-f closest vectors to pivot
def average_nearest_neighbors(vectors, f, pivot):
    vector_scores = list()
    
    for i in range(len(vectors)):
        #JS: compute distance to pivot
        distance = vectors[i].sub(pivot).norm().item()
        vector_scores.append((i, distance))
    
    #JS: sort vector_scores by increasing distance to pivot
    vector_scores.sort(key=lambda x: x[1])
    
    #JS: Return the average of the n-f closest vectors to pivot
    closest_vectors = [vectors[vector_scores[j][0]] for j in range(len(vectors) -f)]
    return torch.stack(closest_vectors).mean(dim=0)

def bucketing(aggregator, vectors):
    def round_up(n):
        if n == int(n):
            # n is integer
            return int(n)
        else:
            # If n is positive
            return int(n) + 1

    random.shuffle(vectors)
    number_buckets = round_up(aggregator.nb_workers / aggregator.bucket_size)
    avg_buckets = list()
    for i in range(number_buckets):
        start_index = i * aggregator.bucket_size
        end_index = min((i + 1) * aggregator.bucket_size, aggregator.nb_workers)
        bucket = vectors[start_index:end_index]
        avg_buckets.append(torch.stack(bucket).mean(dim=0))
    return robust_aggregators[aggregator.second_aggregator](aggregator, avg_buckets)


def pseudo_multi_krum(aggregator, vectors):
    #JS: k is the number of vectors to average in the end
    k = aggregator.nb_workers - aggregator.nb_byz
    k_vectors = list()

    #JS: dictionary to hold pairwise distances
    distances = dict()
    indices = range(aggregator.nb_workers)

    #JS: Run Pseudo Krum k times, and store result in list then average
    for _ in range(k):
        #JS: choose (f+1) vectors at random, and compute their pseudo-scores
        random_indices = random.sample(indices, aggregator.nb_byz + 1)
        #JS: compute the pseudo-scores of only these random vectors
        #JS: a pseudo-score is the same as a normal score, but computed only over a random set of (n-f) neighbors
        min_score = float('inf')

        for index in random_indices:
            #JS: vectors[index] is one of the candidates to be outputted by pseudo-Krum
            random_neighbors = random.sample(indices, k)
            score = 0
            for neighbor in random_neighbors:

                #JS: if index = neighbour, distance = 0 and score is unchanged
                if index == neighbor:
                    continue

                #JS: fetch the distance between vector and neighbor from dictionary (if found)
                #otherwise calculate it and store it in dictionary
                key = (min(index, neighbor), max(index, neighbor))

                if key in distances:
                    dist = distances[key]
                else:
                    dist = vectors[index].sub(vectors[neighbor]).norm().item()
                    distances[key] = dist

                score += dist**2

            if score < min_score:
                min_score, min_index = score, index

        #JS: append the vector with the smallest score (among the considered f+1) to the list
        k_vectors.append(vectors[min_index])

    #JS: return the average of the k vectors
    return torch.stack(k_vectors).mean(dim=0)


def centered_clipping(aggregator, vectors, L_iter=3, clip_thresh=1):
    #JS: v is the returned vector, as per the algorithm of CC
    v = aggregator.prev_momentum
    avg_dist = torch.zeros_like(vectors[0]) # SY: pre-allocate the tensor for summing
    for _ in range(L_iter):
        avg_dist.zero_() # SY: clear the previous sum
        for vector in vectors:
            # SY: compute and clip distance
            distance = vector.sub(v)
            distance = misc.clip_vector(distance, clip_thresh)
            avg_dist.add_(distance) # SY: add to the sum
        avg_dist.div_(aggregator.nb_workers)
        v.add_(avg_dist)
    return v


def minimum_diameter_averaging(aggregator, vectors):
    selected_subset = misc.compute_min_diameter_subset(vectors, aggregator.nb_workers, aggregator.nb_byz)
    selected_vectors = [vectors[j] for j in selected_subset]
    return torch.stack(selected_vectors).mean(dim=0)


def minimum_variance_averaging(aggregator, vectors):
    selected_subset = misc.compute_min_variance_subset(vectors, aggregator.nb_workers, aggregator.nb_byz)
    selected_vectors = [vectors[j] for j in selected_subset]
    return torch.stack(selected_vectors).mean(dim=0)


def monna(aggregator, vectors):
    #JS: Compute n-f closest vectors to vectors[-1] and average them
    return misc.compute_closest_vectors_and_mean(vectors, aggregator.nb_workers, aggregator.nb_byz)


def meamed(aggregator, vectors):
    vectors_stacked = torch.stack(vectors)
    median_vector = robust_aggregators["median"](aggregator, vectors)
    nb_workers, dimension = vectors_stacked.shape
    nb_honest = nb_workers - aggregator.nb_byz
    #JS: compute and aggregate (n-f) vectors closest to median (per dimension)
    bottom_indices = vectors_stacked.sub(median_vector).abs().topk(nb_honest, dim=0, largest=False, sorted=False).indices
    bottom_indices.mul_(dimension).add_(torch.arange(0, dimension, dtype=bottom_indices.dtype, device=bottom_indices.device))
    return vectors_stacked.take(bottom_indices).mean(dim=0)


#JS: Dictionary mapping every aggregator to its corresponding function
robust_aggregators = {"average": average, "trmean": trmean, "median": median, "geometric_median_old": geometric_median_old, "geometric_median": geometric_median,
                      "krum": krum, "multi_krum": multi_krum,
                      "server_clip": server_clip,
                      "nnm": nearest_neighbor_mixing, "bucketing": bucketing, "pmk": pseudo_multi_krum, "cc": centered_clipping, "mda": minimum_diameter_averaging,
                      "mva": minimum_variance_averaging, "monna": monna, "meamed": meamed,
                      "krum_old": krum_old, "nnm_old": nearest_neighbor_mixing_old}

class RobustAggregator(object):

    def __init__(self, aggregator_name, second_aggregator, server_clip, nb_workers, nb_byz, bucket_size, model_size, device):
        self.aggregator_name = aggregator_name
        self.second_aggregator = second_aggregator
        self.server_clip = server_clip
        self.nb_workers = nb_workers
        self.nb_byz = nb_byz
        #JS: bucket size for bucketing aggregator
        self.bucket_size = bucket_size
        #JS; previous value of aggregated momentum, used for example for CC
        self.prev_momentum = torch.zeros(model_size, device=device)

    def aggregate(self, vectors):
        if self.server_clip:
            aggregate_vector = robust_aggregators["server_clip"](self, vectors)
        else:
            aggregate_vector = robust_aggregators[self.aggregator_name](self, vectors)
        #JS: Update the value of the previous momentum (e.g., for Centered Clipping aggregator)
        self.prev_momentum = aggregate_vector
        return aggregate_vector