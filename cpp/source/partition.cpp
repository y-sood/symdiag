#include "partition.hpp"
#include <cstdlib>

void partition_stats(int N, int* nr_groups, int* group_size) {
	int n = N % 2 == 0 ? N : N + 1; // if N is odd we add a dummy index

	*nr_groups = n - 1;
	*group_size = n / 2;
}

void set_pair(PQPair** pairs, int N, int nr_groups, int group_size, int group, int group_i, int p, int q, int* dummy_count) {
	if (p == N || q == N) {
		*dummy_count = (*dummy_count) + 1;
		return;
	}

	PQPair* pair = pairs[(group * group_size) + group_i - (*dummy_count)];
	pair->p = p > q ? q : p;
	pair->q = q > p ? q : p;
}

int wrap(int n, int i) { return ((i - 1) % (n - 1)) + 1; }

void partition(PQPair** pairs, int N, int* nr_groups, int* group_size) {
	int dummy_count = 0;

	// Outer loop pairs the fixed index (0), with all other indices.
	for (int i = 1; i <= (*nr_groups); i++) {
		set_pair(pairs, N, (*nr_groups), (*group_size), i - 1, 0, 0, i, &dummy_count);

		// Inner loop pairs the remaining indices with each other symmetrically
		for (int j = 1; j <= ((*nr_groups) - 1) / 2; j++) {
			int p = wrap((*nr_groups) + 1, i + j);
			int q = wrap((*nr_groups) + 1, p + ((*nr_groups) - 1) - (2 * j - 1));

			set_pair(pairs, N, (*nr_groups), (*group_size), i - 1, j, p, q, &dummy_count);
		}
	}

	if (dummy_count) { *group_size = (*group_size) - 1; } // Decrease group size by one, since one dummy pair has been removed from each group
}

void cleanup_partition(JacobiPartition* partition){
    for (int i = 0; i < partition->nr_groups * partition->group_size; i++) {
        free(partition->pairs[i]);
    }
    free(partition->pairs);
}