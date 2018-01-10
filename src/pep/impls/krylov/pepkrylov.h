/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Private header for TOAR and STOAR
*/

#if !defined(__TOAR_H)
#define __TOAR_H

typedef struct {
  PetscReal   keep;         /* restart parameter */
  PetscBool   lock;         /* locking/non-locking variant */
  PetscScalar *qB;          /* auxiliary matrices */
  BV          V;            /* tensor basis vectors object for the linearization */
} PEP_TOAR;

#endif

PETSC_INTERN PetscErrorCode PEPExtractVectors_TOAR(PEP);

