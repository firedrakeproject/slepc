/*
     This file contains some simple default routines for common operations.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h"   /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_Default"
PetscErrorCode EPSDestroy_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);

  /* free work vectors */
  ierr = EPSDefaultFreeWork(eps);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_Default"
PetscErrorCode EPSBackTransform_Default(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STBackTransform(eps->OP,eps->nconv,eps->eigr,eps->eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Default"
/*
  EPSComputeVectors_Default - Compute eigenvectors from the vectors
  provided by the eigensolver. This version just copies the vectors
  and is intended for solvers such as power that provide the eigenvector.
 */
PetscErrorCode EPSComputeVectors_Default(EPS eps)
{
  PetscFunctionBegin;
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Hermitian"
/*
  EPSComputeVectors_Hermitian - Copies the Lanczos vectors as eigenvectors
  using purification for generalized eigenproblems.
 */
PetscErrorCode EPSComputeVectors_Hermitian(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      norm;
  Vec            w;

  PetscFunctionBegin;
  if (eps->isgeneralized) {
    /* Purify eigenvectors */
    ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->V[i],w);CHKERRQ(ierr);
      ierr = STApply(eps->OP,w,eps->V[i]);CHKERRQ(ierr);
      ierr = IPNorm(eps->ip,eps->V[i],&norm);CHKERRQ(ierr);
      ierr = VecScale(eps->V[i],1.0/norm);CHKERRQ(ierr);
    }
    ierr = VecDestroy(w);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_Schur"
/*
  EPSComputeVectors_Schur - Compute eigenvectors from the vectors
  provided by the eigensolver. This version is intended for solvers 
  that provide Schur vectors. Given the partial Schur decomposition
  OP*V=V*T, the following steps are performed:
      1) compute eigenvectors of T: T*Z=Z*D
      2) compute eigenvectors of OP: X=V*Z
  If left eigenvectors are required then also do Z'*Tl=D*Z', Y=W*Z
 */
PetscErrorCode EPSComputeVectors_Schur(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_TREVC)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   ncv,nconv,mout,info; 
  PetscScalar    *Z,*work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif
  PetscReal      norm;
  Vec            w;
  
  PetscFunctionBegin;
  ncv = PetscBLASIntCast(eps->ncv);
  nconv = PetscBLASIntCast(eps->nconv);
  if (eps->ishermitian) {
    ierr = EPSComputeVectors_Hermitian(eps);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (eps->ispositive) {
    ierr = VecDuplicate(eps->V[0],&w);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(nconv*nconv*sizeof(PetscScalar),&Z);CHKERRQ(ierr);
  ierr = PetscMalloc(3*nconv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(nconv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  /* right eigenvectors */
#if !defined(PETSC_USE_COMPLEX)
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,&info);
#else
  LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->T,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,rwork,&info);
#endif
  if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

  /* AV = V * Z */
  ierr = SlepcUpdateVectors(eps->nconv,eps->V,0,eps->nconv,Z,eps->nconv,PETSC_FALSE);CHKERRQ(ierr);
  if (eps->ispositive) {
    /* Purify eigenvectors */
    for (i=0;i<eps->nconv;i++) {
      ierr = VecCopy(eps->V[i],w);CHKERRQ(ierr); 
      ierr = STApply(eps->OP,w,eps->V[i]);CHKERRQ(ierr);
      ierr = VecNormalize(eps->V[i],&norm);CHKERRQ(ierr);
    }
  }
   
  /* left eigenvectors */
  if (eps->solverclass == EPS_TWO_SIDE) {
#if !defined(PETSC_USE_COMPLEX)
    LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->Tl,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,&info);
#else
    LAPACKtrevc_("R","A",PETSC_NULL,&nconv,eps->Tl,&ncv,PETSC_NULL,&nconv,Z,&nconv,&nconv,&mout,work,rwork,&info);
#endif
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

    /* AW = W * Z */
    ierr = SlepcUpdateVectors(eps->nconv,eps->W,0,eps->nconv,Z,eps->nconv,PETSC_FALSE);CHKERRQ(ierr);
  }
   
  ierr = PetscFree(Z);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  if (eps->ispositive) {
    ierr = VecDestroy(w);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
  PetscFunctionReturn(0);
#endif 
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultGetWork"
/*
  EPSDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
+ eps  - eigensolver context
- nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated.

 */
PetscErrorCode EPSDefaultGetWork(EPS eps, PetscInt nw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (eps->nwork != nw) {
    if (eps->nwork > 0) {
      ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
    }
    eps->nwork = nw;
    ierr = VecDuplicateVecs(eps->vec_initial,nw,&eps->work); CHKERRQ(ierr);
    ierr = PetscLogObjectParents(eps,nw,eps->work);
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultFreeWork"
/*
  EPSDefaultFreeWork - Free work vectors.

  Input Parameters:
. eps  - eigensolver context

 */
PetscErrorCode EPSDefaultFreeWork(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->work)  {
    ierr = VecDestroyVecs(eps->work,eps->nwork); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
