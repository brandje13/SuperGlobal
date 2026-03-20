from collections import Counter


def merge_results(cfg, models, mode):
    results = {}

    # Dynamic threshold: 2 for 3 models, 3 for 4 or 5 models, etc.
    min_votes = (len(models) // 2) + 1

    for query in cfg['qimlist']:
        if mode == 'majority':
            # 1. Pool all votes for this query across all models
            all_votes = []
            for model in models:
                all_votes.extend(model[1][query]['top_k'])

            # 2. Count them and apply the threshold
            vote_counts = Counter(all_votes)
            results[query] = [img for img, count in vote_counts.items() if count >= min_votes]

        else:
            # Union and Intersection (Rolling updates)
            joined_set = set()
            for i, model in enumerate(models):
                top_k_set = set(model[1][query]['top_k'])

                if i == 0:
                    joined_set = top_k_set.copy()
                elif mode == 'union':
                    joined_set.update(top_k_set)
                elif mode == 'intersection':
                    joined_set.intersection_update(top_k_set)

            results[query] = list(joined_set)

    return results