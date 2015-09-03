/*
   Private header for PEPJD.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__PJDP_H)
#define __PJDP_H

typedef struct {
  PetscReal   keep;          /* restart parameter */
  PetscReal   mtol;          /* tolerance for eigenvalue multiplicity */
  PetscReal   htol;          /* tolerance for harmonic JD */
  PetscReal   stol;          /* tolerance for harmonic shift */
  PetscInt    fnini;         /* first initial search space */
  PetscBool   randini;       /* use random initial search space */
  PetscBool   custpc;        /* use custom correction equation preconditioner */
  PetscBool   flglk;         /* whether in locking step */
  PetscBool   flgre;         /* whether in restarting step */
  BV          *W;            /* work basis vectors to store A_i*V */
  PC          pcshell;       /* preconditioner including basic precond+projector */
} PEP_JD;

typedef struct {
  PC          pc;            /* basic preconditioner */
  Vec         Bp;            /* preconditioned residual of derivative polynomial, B\p */
  Vec         u;             /* Ritz vector */
  PetscScalar gamma;         /* precomputed scalar u'*B\p */
} PEP_JD_PCSHELL;

PETSC_INTERN PetscErrorCode PEPView_JD(PEP,PetscViewer);
PETSC_INTERN PetscErrorCode PEPSetFromOptions_JD(PetscOptions*,PEP);
PETSC_INTERN PetscErrorCode PEPJDSetRestart_JD(PEP,PetscReal);
PETSC_INTERN PetscErrorCode PEPJDGetRestart_JD(PEP,PetscReal*);
PETSC_INTERN PetscErrorCode PEPJDSetTolerances_JD(PEP,PetscReal,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode PEPJDGetTolerances_JD(PEP,PetscReal*,PetscReal*,PetscReal*);

#endif
