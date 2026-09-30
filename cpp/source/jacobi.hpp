#pragma once
//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include "jacobi_config.hpp"
#include "partition.hpp"

// Builds the input tensor for config (odeco test tensor, or random symmetric with
// --generic). Deterministic in config.seed, so it can be regenerated after the run
// instead of keeping a copy alive through the sweeps.
void make_input_tensor(const JacobiConfig& config, dim_t tSize[FLA_MAX_ORDER], FLA_Obj* T);
void setup_jacobi(FLA_Obj* T, FLA_Obj* F, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]);
void jacobi_diagonalization(FLA_Obj* T, FLA_Obj* F, const JacobiConfig& config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]);
void cleanup_jacobi(FLA_Obj* T, FLA_Obj* F);
