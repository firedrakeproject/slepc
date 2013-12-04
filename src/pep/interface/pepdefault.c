/*
     This file contains some simple default routines for common PEP operations.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/pepimpl.h>     /*I "slepcpep.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "PEPReset_Default"
PetscErrorCode PEPReset_Default(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(pep->nwork,&pep->work);CHKERRQ(ierr);
  pep->nwork = 0;
  ierr = PEPFreeSolution(pep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetWorkVecs"
/*@
   PEPSetWorkVecs - Sets a number of work vectors into a PEP object

   Collective on PEP

   Input Parameters:
+  pep - polynomial eigensolver context
-  nw  - number of work vectors to allocate

   Developers Note:
   This is PETSC_EXTERN because it may be required by user plugin PEP
   implementations.

   Level: developer
@*/
PetscErrorCode PEPSetWorkVecs(PEP pep,PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pep->nwork != nw) {
    ierr = VecDestroyVecs(pep->nwork,&pep->work);CHKERRQ(ierr);
    pep->nwork = nw;
    ierr = VecDuplicateVecs(pep->t,nw,&pep->work);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(pep,nw,pep->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPConvergedEigRelative"
/*
  PEPConvergedEigRelative - Checks convergence relative to the eigenvalue.
*/
PetscErrorCode PEPConvergedEigRelative(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal w;

  PetscFunctionBegin;
  w = SlepcAbsEigenvalue(eigr,eigi);
  *errest = res/w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPConvergedAbsolute"
/*
  PEPConvergedAbsolute - Checks convergence absolutely.
*/
PetscErrorCode PEPConvergedAbsolute(PEP pep,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = res;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors_Schur"
PetscErrorCode PEPComputeVectors_Schur(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       n,ld,i;
  PetscScalar    *Z;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    tmp;
  PetscReal      norm,normi;
#endif

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(pep->ds,&n,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /* right eigenvectors */
  ierr = DSVectors(pep->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);

  /* AV = V * Z */
  ierr = DSGetArray(pep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);
  ierr = SlepcUpdateVectors(n,pep->V,0,n,Z,ld,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DSRestoreArray(pep->ds,DS_MAT_X,&Z);CHKERRQ(ierr);

  /* normalization */
  for (i=0;i<n;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (pep->eigi[i] != 0.0) {
      ierr = VecNorm(pep->V[i],NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecNorm(pep->V[i+1],NORM_2,&normi);CHKERRQ(ierr);
      tmp = 1.0 / SlepcAbsEigenvalue(norm,normi);
      ierr = VecScale(pep->V[i],tmp);CHKERRQ(ierr);
      ierr = VecScale(pep->V[i+1],tmp);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      ierr = VecNormalize(pep->V[i],NULL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PEPKrylovConvergence"
/*
   PEPKrylovConvergence - This is the analogue to EPSKrylovConvergence, but
   for polynomial Krylov methods.

   Differences:
   - Always non-symmetric
   - Does not check for STSHIFT
   - No correction factor
   - No support for true residual
*/
PetscErrorCode PEPKrylovConvergence(PEP pep,PetscBool getall,PetscInt kini,PetscInt nits,PetscInt nv,PetscReal beta,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,ld;
  PetscScalar    re,im;
  PetscReal      resnorm;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  marker = -1;
  if (pep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = pep->eigr[k];
    im = pep->eigi[k];
    newk = k;
    ierr = DSVectors(pep->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    resnorm *= beta;
    /* error estimate */
    ierr = (*pep->converged)(pep,re,im,resnorm,&pep->errest[k],pep->convergedctx);CHKERRQ(ierr);
    if (marker==-1 && pep->errest[k] >= pep->tol) marker = k;
    if (newk==k+1) {
      pep->errest[k+1] = pep->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}

