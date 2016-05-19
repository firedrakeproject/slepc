/*
   Private header for PEPJD.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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
  BV          V;             /* work basis vectors to store the search space */
  BV          W;             /* work basis vectors to store the test space */
  BV          *TV;           /* work basis vectors to store T*V (each TV[i] is the coefficient for \lambda^i of T*V for the extended T) */
  BV          *AX;           /* work basis vectors to store A_i*X for locked eigenvectors */
  BV          X;             /* locked eigenvectors */
  PetscScalar *T;            /* matrix of the invariant pair */
  PetscScalar *Tj;           /* matrix containing the powers of the invariant pair matrix */
  PetscScalar *XpX;          /* X^H*X */
  PC          pcshell;       /* preconditioner including basic precond+projector */
  Mat         Pshell;        /* auxiliary shell matrix */
  PetscInt    nconv;         /* number of locked vectors in the invariant pair */
} PEP_JD;

typedef struct {
  PC          pc;            /* basic preconditioner */
  Vec         Bp;            /* preconditioned residual of derivative polynomial, B\p */
  Vec         u;             /* Ritz vector */
  PetscScalar gamma;         /* precomputed scalar u'*B\p */
  PetscScalar *M;
  PetscScalar *ps;
  PetscInt    ld;
  Vec         *work;
  BV          X;
  PetscInt    n;
} PEP_JD_PCSHELL;

typedef struct {
  Mat         P;             /*  */
  PEP         pep;
  Vec         *work;
  PetscScalar theta;
} PEP_JD_MATSHELL;

PETSC_INTERN PetscErrorCode PEPView_JD(PEP,PetscViewer);
PETSC_INTERN PetscErrorCode PEPSetFromOptions_JD(PetscOptionItems*,PEP);
PETSC_INTERN PetscErrorCode PEPJDSetRestart_JD(PEP,PetscReal);
PETSC_INTERN PetscErrorCode PEPJDGetRestart_JD(PEP,PetscReal*);

#endif
