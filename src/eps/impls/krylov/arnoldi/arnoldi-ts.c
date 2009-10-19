/*                       

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi (two-sided)

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

#include "private/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_TS_ARNOLDI"
PetscErrorCode EPSSolve_TS_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,ncv=eps->ncv;
  Vec            fr=eps->work[0];
  Vec            fl=eps->work[1];
  Vec            *Qr=eps->V, *Ql=eps->W;
  PetscScalar    *Hr=eps->T,*Ur,*work;
  PetscScalar    *Hl=eps->Tl,*Ul;
  PetscReal      beta,g;
  PetscScalar    *eigr,*eigi,*aux;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(Hr,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(Hl,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Ur);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Ul);CHKERRQ(ierr);
  ierr = PetscMalloc((ncv+4)*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&eigr);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&eigi);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscScalar),&aux);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,eps->its,Qr[0],PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSGetLeftStartVector(eps,eps->its,Ql[0]);CHKERRQ(ierr);
  
  /* Restart loop */
  while (eps->its<eps->max_it) {
    eps->its++;

    /* Compute an ncv-step Arnoldi factorization for both A and A' */
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,Hr,ncv,Qr,eps->nconv,&ncv,fr,&beta,&breakdown);CHKERRQ(ierr);
    ierr = EPSBasicArnoldi(eps,PETSC_TRUE,Hl,ncv,Ql,eps->nconv,&ncv,fl,&g,&breakdown);CHKERRQ(ierr);

    ierr = IPBiOrthogonalize(eps->ip,ncv,Qr,Ql,fr,aux,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<ncv;i++) {
      Hr[ncv*(ncv-1)+i] += beta * aux[i];
    }
    ierr = IPBiOrthogonalize(eps->ip,ncv,Ql,Qr,fl,aux,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<ncv;i++) {
      Hl[ncv*(ncv-1)+i] += g * aux[i];
    }

    /* Reduce H to (quasi-)triangular form, H <- U H U' */
    ierr = PetscMemzero(Ur,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<ncv;i++) { Ur[i*(ncv+1)] = 1.0; }
    ierr = EPSDenseSchur(ncv,eps->nconv,Hr,ncv,Ur,eps->eigr,eps->eigi);CHKERRQ(ierr);

    ierr = PetscMemzero(Ul,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<ncv;i++) { Ul[i*(ncv+1)] = 1.0; }
    ierr = EPSDenseSchur(ncv,eps->nconv,Hl,ncv,Ul,eigr,eigi);CHKERRQ(ierr);

    /* Sort the remaining columns of the Schur form */
    ierr = EPSSortDenseSchur(eps,ncv,eps->nconv,Hr,ncv,Ur,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = EPSSortDenseSchur(eps,ncv,eps->nconv,Hl,ncv,Ul,eigr,eigi);CHKERRQ(ierr);

    /* Compute residual norm estimates */
    ierr = ArnoldiResiduals(Hr,ncv,Ur,PETSC_NULL,beta,eps->nconv,ncv,eps->eigr,eps->eigi,eps->errest,work);CHKERRQ(ierr);
    ierr = ArnoldiResiduals(Hl,ncv,Ul,PETSC_NULL,g,eps->nconv,ncv,eigr,eigi,eps->errest_left,work);CHKERRQ(ierr);

    /* Lock converged eigenpairs and update the corresponding vectors,
       including the restart vector: V(:,idx) = V*U(:,idx) */
    k = eps->nconv;
    while (k<ncv && eps->errest[k]<eps->tol && eps->errest_left[k]<eps->tol) k++;
    ierr = SlepcUpdateVectors(ncv,Qr,eps->nconv,PetscMin(k+1,ncv),Ur,ncv,PETSC_FALSE);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(ncv,Ql,eps->nconv,PetscMin(k+1,ncv),Ul,ncv,PETSC_FALSE);CHKERRQ(ierr);
    eps->nconv = k;

    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest_left,ncv);
    if (eps->nconv >= eps->nev) break;
  }
  
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  ierr = PetscFree(Ur);CHKERRQ(ierr);
  ierr = PetscFree(Ul);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(eigr);CHKERRQ(ierr);
  ierr = PetscFree(eigi);CHKERRQ(ierr);
  ierr = PetscFree(aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

