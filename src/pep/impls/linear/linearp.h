/*
   Private header for PEPLINEAR.

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

#if !defined(__LINEARP_H)
#define __LINEARP_H

typedef struct {
  PetscBool  explicitmatrix;
  PEP        pep;
  PetscInt   cform;            /* companion form */
  PetscReal  sfactor,dsfactor; /* scaling factors */
  Mat        A,B;              /* matrices of generalized eigenproblem */
  EPS        eps;              /* linear eigensolver for Az=lBz */
  Mat        M,C,K;            /* copy of PEP coefficient matrices */
  Vec        w[6];             /* work vectors */
  PetscBool  setfromoptionscalled;
} PEP_LINEAR;

/* General case for implicit matrices of degree d */
PETSC_INTERN PetscErrorCode MatMult_Linear(Mat,Vec,Vec);

/* N1 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_N1A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_N1B(MPI_Comm,PEP_LINEAR*,Mat*);

/* N2 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_N2A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_N2B(MPI_Comm,PEP_LINEAR*,Mat*);

/* S1 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_S1A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_S1B(MPI_Comm,PEP_LINEAR*,Mat*);

/* S2 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_S2A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_S2B(MPI_Comm,PEP_LINEAR*,Mat*);

/* H1 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_H1A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_H1B(MPI_Comm,PEP_LINEAR*,Mat*);

/* H2 */
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_H2A(MPI_Comm,PEP_LINEAR*,Mat*);
PETSC_INTERN PetscErrorCode MatCreateExplicit_Linear_H2B(MPI_Comm,PEP_LINEAR*,Mat*);

#endif
