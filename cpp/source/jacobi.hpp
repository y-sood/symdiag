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

void setup_jacobi(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O, JacobiConfig config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER], int seed);
void jacobi_diagonalization(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O, JacobiConfig config, JacobiPartition* vpartition, dim_t tSize[FLA_MAX_ORDER]);
void cleanup_jacobi(FLA_Obj* T, FLA_Obj* F, FLA_Obj* F_temp, FLA_Obj* O);