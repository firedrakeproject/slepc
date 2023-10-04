/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Private header for SLP
*/

#pragma once

typedef struct {
  EPS       eps;      /* linear eigensolver for T*z = mu*Tp*z */
  EPS       epsts;    /* linear eigensolver for T'*z = mu*Tp'*z */
  KSP       ksp;
  PetscReal deftol;   /* tolerance for the deflation (threshold) */
} NEP_SLP;

SLEPC_INTERN PetscErrorCode NEPSolve_SLP(NEP);
SLEPC_INTERN PetscErrorCode NEPSolve_SLP_Twosided(NEP);
