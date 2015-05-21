/*

   SLEPc matrix function solver: "krylov"

   Method: Arnoldi

   Algorithm:

       Single-vector Arnoldi method to build a Krylov subspace, then
       compute f(B) on the projected matrix B.

   References:

       [1] R. Sidje, "Expokit: a software package for computing matrix
           exponentials", ACM Trans. Math. Softw. 24(1):130-156, 1998.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/mfnimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MFNSetUp_Krylov"
PetscErrorCode MFNSetUp_Krylov(MFN mfn)
{
  PetscErrorCode  ierr;
  PetscInt        N;

  PetscFunctionBegin;
  ierr = MatGetSize(mfn->A,&N,NULL);CHKERRQ(ierr);
  if (!mfn->ncv) mfn->ncv = PetscMin(30,N);
  if (!mfn->max_it) mfn->max_it = PetscMax(100,2*N/mfn->ncv);
  ierr = MFNAllocateSolution(mfn,2);CHKERRQ(ierr);
  ierr = DSAllocate(mfn->ds,mfn->ncv+2);CHKERRQ(ierr);
  ierr = DSSetType(mfn->ds,DSNHEP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNBasicArnoldi"
static PetscErrorCode MFNBasicArnoldi(BV V, Mat A,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  PetscReal      norm;
  Vec            vj,vj1;

  PetscFunctionBegin;
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(V,j+1,&vj1);CHKERRQ(ierr);
    ierr = MatMult(A,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(V,j+1,H+ldh*j,&norm,breakdown);CHKERRQ(ierr);
    H[j+1+ldh*j] = norm;
    if (*breakdown) {
      *M = j+1;
      PetscFunctionReturn(0);
    } else {
      ierr = BVScaleColumn(V,j+1,1/norm);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSolve_Krylov"
PetscErrorCode MFNSolve_Krylov(MFN mfn,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       mxstep,mxrej,m,mb,ld,i,j,ireject,mx,k1;
  Vec            v,r;
  PetscScalar    *H,*B,*F,*betaF,t,sgn;
  PetscReal      anorm,normb,avnorm,tol,err_loc,rndoff;
  PetscReal      t_out,t_new,t_now,t_step;
  PetscReal      xm,fact,s,p1,p2;
  PetscReal      beta,gamma,delta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  m   = mfn->ncv;
  tol = mfn->tol;
  mxstep = mfn->max_it;
  mxrej = 10;
  gamma = 0.9;
  delta = 1.2;
  mb    = m;
  t     = mfn->sfactor;
  t_out = PetscAbsScalar(t);
  t_new = 0.0;
  t_now = 0.0;
  ierr = MatNorm(mfn->A,NORM_INFINITY,&anorm);CHKERRQ(ierr);
  rndoff = anorm*PETSC_MACHINE_EPSILON;

  k1 = 2;
  xm = 1.0/(PetscReal)m;
  ierr = VecNorm(b,NORM_2,&normb);CHKERRQ(ierr);
  if (!normb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot pass a zero b vector to MFNSolve()");
  beta = normb;
  fact = PetscPowRealInt((m+1)/2.72,m+1)*PetscSqrtReal(2*PETSC_PI*(m+1));
  t_new = (1.0/anorm)*PetscPowReal((fact*tol)/(4.0*beta*anorm),xm);
  s = PetscPowReal(10.0,PetscFloorReal(PetscLog10Real(t_new))-1);
  t_new = PetscCeilReal(t_new/s)*s;
  sgn = t/PetscAbsScalar(t);

  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(mfn->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc2(m+1,&betaF,ld*ld,&B);CHKERRQ(ierr);

  while (mfn->reason == MFN_CONVERGED_ITERATING) {
    mfn->its++;
    if (PetscIsInfOrNanReal(t_new)) t_new = PETSC_MAX_REAL;
    t_step = PetscMin(t_out-t_now,t_new);
    ierr = BVInsertVec(mfn->V,0,x);CHKERRQ(ierr);
    ierr = BVScaleColumn(mfn->V,0,1.0/beta);CHKERRQ(ierr);
    ierr = DSGetArray(mfn->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = MFNBasicArnoldi(mfn->V,mfn->A,H,ld,0,&mb,&breakdown);CHKERRQ(ierr);
    if (breakdown) {
      k1 = 0;
      t_step = t_out-t_now;
    }
    if (k1!=0) {
      H[m+1+ld*m] = 1.0;
      ierr = BVGetColumn(mfn->V,m,&v);CHKERRQ(ierr);
      ierr = BVGetColumn(mfn->V,m+1,&r);CHKERRQ(ierr);
      ierr = MatMult(mfn->A,v,r);CHKERRQ(ierr);
      ierr = BVRestoreColumn(mfn->V,m,&v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(mfn->V,m+1,&r);CHKERRQ(ierr);
      ierr = BVNormColumn(mfn->V,m+1,NORM_2,&avnorm);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(B,H,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = DSRestoreArray(mfn->ds,DS_MAT_A,&H);CHKERRQ(ierr);

    ireject = 0;
    while (ireject <= mxrej) {
      mx = mb + k1;
      ierr = DSSetDimensions(mfn->ds,mx,0,0,0);CHKERRQ(ierr);
      ierr = DSGetArray(mfn->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      for (i=0;i<mx;i++) {
        for (j=0;j<mx;j++) {
          H[i+j*ld] = sgn*B[i+j*ld]*t_step;
        }
      }
      ierr = DSRestoreArray(mfn->ds,DS_MAT_A,&H);CHKERRQ(ierr);
      ierr = DSComputeFunction(mfn->ds,mfn->function);CHKERRQ(ierr);

      if (k1==0) {
        err_loc = tol;
        break;
      } else {
        ierr = DSGetArray(mfn->ds,DS_MAT_F,&F);CHKERRQ(ierr);
        p1 = PetscAbsScalar(beta*F[m]);
        p2 = PetscAbsScalar(beta*F[m+1]*avnorm);
        ierr = DSRestoreArray(mfn->ds,DS_MAT_F,&F);CHKERRQ(ierr);
        if (p1 > 10*p2) {
          err_loc = p2;
          xm = 1.0/(PetscReal)m;
        } else if (p1 > p2) {
          err_loc = (p1*p2)/(p1-p2);
          xm = 1.0/(PetscReal)m;
        } else {
          err_loc = p1;
          xm = 1.0/(PetscReal)(m-1);
        }
      }
      if (err_loc <= delta*t_step*tol) break;
      else {
        t_step = gamma*t_step*PetscPowReal(t_step*tol/err_loc,xm);
        s = PetscPowReal(10.0,PetscFloorReal(PetscLog10Real(t_step))-1);
        t_step = PetscCeilReal(t_step/s)*s;
        ireject = ireject+1;
      }
    }

    mx = mb + PetscMax(0,k1-1);
    ierr = DSGetArray(mfn->ds,DS_MAT_F,&F);CHKERRQ(ierr);
    for (j=0;j<mx;j++) betaF[j] = beta*F[j];
    ierr = DSRestoreArray(mfn->ds,DS_MAT_F,&F);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(mfn->V,0,mx);CHKERRQ(ierr);
    ierr = BVMultVec(mfn->V,1.0,0.0,x,betaF);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&beta);CHKERRQ(ierr);

    t_now = t_now+t_step;
    if (t_now>=t_out) mfn->reason = MFN_CONVERGED_TOL;
    else {
      t_new = gamma*t_step*PetscPowReal((t_step*tol)/err_loc,xm);
      s = PetscPowReal(10.0,PetscFloorReal(PetscLog10Real(t_new))-1);
      t_new = PetscCeilReal(t_new/s)*s;
    }
    err_loc = PetscMax(err_loc,rndoff);
    if (mfn->its==mxstep) mfn->reason = MFN_DIVERGED_ITS;
    ierr = MFNMonitor(mfn,mfn->its,t_now);CHKERRQ(ierr);
  }

  ierr = PetscFree2(betaF,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNCreate_Krylov"
PETSC_EXTERN PetscErrorCode MFNCreate_Krylov(MFN mfn)
{
  PetscFunctionBegin;
  mfn->ops->solve          = MFNSolve_Krylov;
  mfn->ops->setup          = MFNSetUp_Krylov;
  PetscFunctionReturn(0);
}
