#pragma once

struct PQPair {
    int p;
    int q;
};

struct JacobiPartition {
    int nr_groups;
    int group_size;
    PQPair** pairs;
};

void partition_stats(int N, int* nr_groups, int* group_size);
void partition(PQPair** pairs, int N, int* nr_groups, int* group_size);
void cleanup_partition(JacobiPartition* partition);

