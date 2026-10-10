import numpy as np
import pandas as pd
import time
from pathlib import Path
 
 
# ============================================================================
# PROFILE GENERATION
# ============================================================================
 
def generate_full_profile(weights_dict, output_path=None):
    """
    Generates a full ranking profile for all features in a weights dictionary.
 
    Converts raw importance weights to absolute values, groups near-identical
    scores into the same equivalence class, then assigns average ranks to each
    class so that tied (or near-tied) features share one rank value.
 
    Near-tie threshold
    ------------------
    Two features are considered to have the same score when:
        |score_A - score_B| <= (max_score - min_score) / 100
 
    Binning is done with a single greedy pass over the scores sorted in
    descending order.  A new bin starts only when the gap from the current
    bin's first (highest) score exceeds the threshold.  This guarantees
    transitivity: if A ~ B and B ~ C (each pair within threshold) they all
    land in the same bin, rather than being split by a non-transitive
    pairwise comparison.
 
    Ranking
    -------
    After binning, each feature is assigned the integer bin label (1 = most
    important bin).  Average ranks are then computed from those bin labels
    using pd.Series.rank(method='average'), so all features in the same bin
    receive the same fractional rank — the midpoint of the positions they
    collectively occupy.
 
    Normalized score
    ----------------
    Alongside rank/position (which only capture ORDER), each feature also
    gets a min-max normalized score that captures MAGNITUDE:
        normalized_score(i) = (abs_weight(i) - score_min) / (score_max - score_min)
    so the most important feature scores 1.0, the least important scores 0.0,
    and everything else falls proportionally in between. When every feature
    has the same weight (score_range == 0) there is no magnitude information
    to normalize, so every feature is assigned 1.0.

    Args:
        weights_dict (dict): Mapping of feature name -> importance weight
                             (can be positive or negative; absolute value is used).
        output_path (str | Path | None): If provided, the full ranking is saved
                                         to a CSV at this path with columns
                                         ['feature', 'abs_weight', 'normalized_score',
                                         'bin', 'rank', 'position'].
                                         Parent directories are created automatically.

    Returns:
        pd.Series: Series of average ranks indexed by feature name, where
                   rank 1 (or the lowest fractional rank) corresponds to the
                   most important feature.
    """
    abs_weights = pd.Series(weights_dict).abs()
    _profile_start_time = time.time()

    score_min = abs_weights.min()
    score_max = abs_weights.max()
    score_range = score_max - score_min

    # When all scores are identical the threshold is 0; every feature ties.
    threshold = score_range / 100.0 if score_range > 0 else 0.0

    # --- min-max normalized score (magnitude, independent of rank/bin) ---
    if score_range > 0:
        normalized_scores = (abs_weights - score_min) / score_range
    else:
        # No spread at all -> nothing to normalize; treat every feature as
        # equally (fully) important rather than dividing by zero.
        normalized_scores = pd.Series(1.0, index=abs_weights.index)
 
    # --- greedy near-tie binning (descending pass) ---
    # Primary sort: descending score.
    # Secondary sort: ascending feature name — deterministic, alphabetical
    # tiebreaker so all consumers (fidelity, stability, plots) see the same
    # canonical order for features with identical importance scores.
    sorted_weights = abs_weights.sort_values(ascending=False)
    sort_keys = pd.DataFrame({'score': sorted_weights, 'feature': sorted_weights.index})
    sort_keys = sort_keys.sort_values(['score', 'feature'], ascending=[False, True])
    sorted_weights = abs_weights[sort_keys.index]

    bin_labels = pd.Series(index=sorted_weights.index, dtype=int)
    current_bin = 1
    bin_anchor = sorted_weights.iloc[0]  # highest score in the current bin

    for feat, score in sorted_weights.items():
        if (bin_anchor - score) > threshold:
            # Gap too large → start a new bin anchored at this score
            current_bin += 1
            bin_anchor = score
        bin_labels[feat] = current_bin

    # --- average ranks from bin labels ---
    # bin_labels are 1-based integers; rank(ascending=True) gives the correct
    # average position because bin 1 (best) should map to the smallest rank number.
    ranks = bin_labels.rank(method='average', ascending=True)

    # Enforce canonical order: primary = rank (ascending),
    # secondary = feature name A→Z within the same rank value.
    ranks = ranks.sort_index()            # A-Z within same rank
    ranks = ranks.sort_values(kind='stable')  # then by rank value
    
    positions = pd.Series(range(1, len(ranks) + 1), index=ranks.index)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile_df = pd.DataFrame({
            'feature':         ranks.index,
            'abs_weight':      abs_weights[ranks.index].values,
            'normalized_score': normalized_scores[ranks.index].values,
            'bin':             bin_labels[ranks.index].values,
            'rank':            ranks.values,
            'position':        positions.values,
        })
        profile_df.to_csv(output_path, index=False)

    _profile_elapsed = time.time() - _profile_start_time
    print(f"⏱  [generate_full_profile | {len(ranks)} features] completed in "
          f"{_profile_elapsed:.2f}s ({_profile_elapsed*1000:.1f} ms)")

    return ranks
 
 
# ============================================================================
# F_prof  (Spearman's Footrule on Profiles — Fagin et al. 2003)
# ============================================================================
 
def compute_f_prof_distance(full_ranks1, full_ranks2, full_positions1, full_positions2, k, total_features):
    """
    Computes the normalized F_prof distance between two full ranking profiles.

    Each feature i is mapped to a projected rank:
        f(i) = rank(i)  if position(i) ≤ k
        f(i) = ℓ        otherwise

    The top-k membership is determined by *position* (1-based ordinal slot),
    so features ranked 2.5 (a tie spanning positions 2 and 3) are both
    included when k ≥ 2, matching the intent of the caller.

    Args:
        full_ranks1 (pd.Series):     Rank values for method 1, indexed by feature name.
        full_ranks2 (pd.Series):     Rank values for method 2, indexed by feature name.
        full_positions1 (pd.Series): Position values for method 1, indexed by feature name.
        full_positions2 (pd.Series): Position values for method 2, indexed by feature name.
        k (int):                     The top-k cut-off to evaluate.
        total_features (int):        Total number of features |D|.

    Returns:
        dict: {
            'normalized_distance': float,
            'raw_distance':        float,
            'location_param':      float,
            'max_distance':        float,
        }
    """
    l_param = (total_features + k + 1) / 2.0
    max_distance = 2 * k * l_param - k * (k + 1)

    all_features = full_ranks1.index.union(full_ranks2.index)

    def project(ranks, positions, features, l):
        projected = {}
        for feat in features:
            r = ranks.get(feat, np.nan)
            pos = positions.get(feat, np.nan)
            if pd.isna(r) or pd.isna(pos):
                projected[feat] = l
            else:
                projected[feat] = r if pos <= k else l
        return projected

    proj1 = project(full_ranks1, full_positions1, all_features, l_param)
    proj2 = project(full_ranks2, full_positions2, all_features, l_param)

    raw_distance = sum(abs(proj1[f] - proj2[f]) for f in all_features)
    normalized = raw_distance / max_distance if max_distance > 0 else 0.0

    return {
        'normalized_distance': round(normalized, 4),
        'similarity':          round(1 - normalized, 4),
        'raw_distance':        round(raw_distance, 4),
        'location_param':      round(l_param, 4),
        'max_distance':        round(max_distance, 4),
    }


# ============================================================================
# Score-based distance  (magnitude comparison, using normalized_score)
# ============================================================================

def compute_score_distance(full_scores1, full_scores2, full_positions1, full_positions2, k):
    """
    Computes a normalized magnitude-based distance between two ranking
    profiles, using each feature's min-max normalized_score instead of its
    rank. This is the score-based counterpart to compute_f_prof_distance.

    Each feature i is mapped to a projected score:
        g(i) = normalized_score(i)   if position(i) ≤ k   (i.e. in the top-k)
        g(i) = avg_tail_score        otherwise
    """
    # 1. --- Calculate average normalized_score of each method's own out-of-top-k tail ---
    tail1_mask = full_positions1 > k
    tail2_mask = full_positions2 > k
    
    avg_tail_1 = full_scores1[tail1_mask].mean()
    avg_tail_2 = full_scores2[tail2_mask].mean()
    
    # Fallback to 0.0 if there is no tail (e.g., k >= total_features)
    avg_tail_1 = 0.0 if pd.isna(avg_tail_1) else float(avg_tail_1)
    avg_tail_2 = 0.0 if pd.isna(avg_tail_2) else float(avg_tail_2)

    # 2. --- Define the evaluation universe (Union of Top-K features) ---
    top1_idx = full_positions1[full_positions1 <= k].index
    top2_idx = full_positions2[full_positions2 <= k].index
    all_features = sorted(top1_idx.union(top2_idx))
    
    # If no features made it into the top-k in either profile, distance is zero
    if not all_features:
        return {
            'normalized_distance': 0.0, 'similarity': 1.0, 'raw_distance': 0.0,
            'max_distance': 0.0, 'avg_tail_score_1': avg_tail_1, 'avg_tail_score_2': avg_tail_2
        }

    # 3. --- Vectorized Projection onto the Union Set ---
    # Reindex aligns features; any feature missing or outside top-k gets filled with the tail average.
    # We use np.inf as a placeholder for missing positions to safely trigger the '>' condition.
    
    # Profile 1 Projection
    s1_proj = full_scores1.reindex(all_features).fillna(avg_tail_1).to_numpy()
    pos1_proj = full_positions1.reindex(all_features).fillna(np.inf).to_numpy()
    s1_proj = np.where(pos1_proj > k, avg_tail_1, s1_proj)

    # Profile 2 Projection
    s2_proj = full_scores2.reindex(all_features).fillna(avg_tail_2).to_numpy()
    pos2_proj = full_positions2.reindex(all_features).fillna(np.inf).to_numpy()
    s2_proj = np.where(pos2_proj > k, avg_tail_2, s2_proj)

    # 4. --- Distance Calculation & Normalization ---
    raw_distance = np.sum(np.abs(s1_proj - s2_proj))
    
    # The theoretical worst-case occurs when the top-k sets are completely disjoint.
    # In this scenario, Profile 1's top-k features (score 1.0) drop to Profile 2's tail (avg_tail_2), 
    # and Profile 2's top-k features (score 1.0) drop to Profile 1's tail (avg_tail_1).
    max_distance = k * (1.0 - avg_tail_2) + k * (1.0 - avg_tail_1)

    # Prevent division by zero and normalize
    normalized = raw_distance / max_distance if max_distance > 0 else 0.0
    
    # Clip to ensure structural boundaries [0.0, 1.0] despite minor numerical/overlap float variances
    normalized = min(max(normalized, 0.0), 1.0) 

    return {
        'normalized_distance': round(normalized, 4),
        'similarity':          round(1 - normalized, 4),
        'raw_distance':        round(float(raw_distance), 4),
        'max_distance':        round(float(max_distance), 4),
        'avg_tail_score_1':    round(avg_tail_1, 4),
        'avg_tail_score_2':    round(avg_tail_2, 4),
    }

# ============================================================================
# TOP-K LIST HELPERS  (Footrule breakdown, Jaccard, Overlap@K)
# ============================================================================

def compute_f_distance_analysis(method1_ranks, method2_ranks, k):
    """
    Computes the normalized Spearman's Footrule distance F^(k+1) 
    between two top-k lists and breaks it down into components
    according to the paper "Comparing top k lists".
    
    Args:
        method1_ranks (dict): Mapping of feature name to its rank (1 to k).
        method2_ranks (dict): Mapping of feature name to its rank (1 to k).
        k (int): The size of the top-k list.
        
    Returns:
        dict: Breakdown of the disagreement between the two methods.
    """
    
    # Define parameters according to Lemma 3.5
    l_param = k + 1  # Location parameter for missing elements
    d1_elements = set(method1_ranks.keys())
    d2_elements = set(method2_ranks.keys())
    
    # Define groups
    z_set = d1_elements.intersection(d2_elements)  # Shared features
    s_set = d1_elements - d2_elements              # Features only in method 1
    t_set = d2_elements - d1_elements              # Features only in method 2
    
    z = len(z_set)
    
    # 1. Ordering Disagreement: 
    # Sum of rank differences for shared elements (Z)
    ordering_dist = sum(abs(method1_ranks[col] - method2_ranks[col]) for col in z_set)
    
    # 2. Identity Disagreement: 
    # Penalty for features present in one list but missing in the other (S and T)
    # Using the derived formula from Lemma 3.5
    identity_dist = (2 * (k - z) * l_param) - sum(method1_ranks[col] for col in s_set) - sum(method2_ranks[col] for col in t_set)
    
    # 3. Total Distance
    total_f_dist = ordering_dist + identity_dist
    
    # 4. Normalization
    # The maximum possible distance F^(k+1) is k*(k+1) when lists are disjoint
    max_possible_dist = k * (k + 1)
    normalized_total = total_f_dist / max_possible_dist
    
    return {
        "normalized_distance": round(normalized_total, 4),
        "identity_component": round(identity_dist / max_possible_dist, 4),
        "ordering_component": round(ordering_dist / max_possible_dist, 4),
        "overlap_count": z,
        "disjoint_count": k - z
    }
    
def compute_jaccard_similarity(list1_keys, list2_keys):
    """
    Computes the Jaccard Similarity coefficient between two sets of features.
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    set1 = set(list1_keys)
    set2 = set(list2_keys)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return round(intersection / union, 4) if union > 0 else 0.0

def compute_overlap_at_k(list1_keys, list2_keys, k):
    """
    Computes the Overlap@K (Intersection over K) between two lists of features.
    Overlap(A, B) = |A ∩ B| / k
    Note: list1_keys and list2_keys should already be the Top-K features.
    """
    set1 = set(list1_keys)
    set2 = set(list2_keys)
    
    intersection = len(set1.intersection(set2))
    
    return round(intersection / k, 4) if k > 0 else 0.0