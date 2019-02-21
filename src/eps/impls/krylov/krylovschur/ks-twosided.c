/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "krylovschur"

   Method: Two-sided Arnoldi with Krylov-Schur restart (for left eigenvectors)

   References:

       [1] I.N. Zwaan and M.E. Hochstenbach, "Krylov-Schur-type restarts
           for the two-sided Arnoldi method", SIAM J. Matrix Anal. Appl.
           38(2):297-321, 2017.

*/

#include <slepc/private/epsimpl.h>
#include "krylovschur.h"

PetscErrorCode EPSSolve_KrylovSchur_TwoSided(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        ld;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  SETERRQ(PetscObjectComm((PetscObject)eps),1,"Not implemented yet");
  PetscFunctionReturn(0);
}

