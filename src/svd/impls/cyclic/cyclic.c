/*                       

   SLEPc singular value solver: "cyclic"

   Method: Uses a Hermitian eigensolver for H(A) = [ 0  A ; A^T 0 ]

   Last update: Jun 2007

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

#include "private/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepceps.h"

typedef struct {
  PetscTruth explicitmatrix;
  EPS        eps;
  PetscTruth setfromoptionscalled;
  Mat        mat;
  Vec        x1,x2,y1,y2;
} SVD_CYCLIC;

#undef __FUNCT__  
#define __FUNCT__ "ShellMatMult_CYCLIC"
PetscErrorCode ShellMatMult_CYCLIC(Mat B,Vec x, Vec y)
{
  PetscErrorCode ierr;
  SVD            svd;
  SVD_CYCLIC     *cyclic;
  PetscScalar    *px,*py;
  PetscInt       m;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  cyclic = (SVD_CYCLIC *)svd->data;
  ierr = SVDMatGetLocalSize(svd,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->x2,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(cyclic->y2,py+m);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_FALSE,cyclic->x2,cyclic->y1);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_TRUE,cyclic->x1,cyclic->y2);CHKERRQ(ierr);        
  ierr = VecResetArray(cyclic->x1);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->x2);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->y1);CHKERRQ(ierr);
  ierr = VecResetArray(cyclic->y2);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ShellMatGetDiagonal_CYCLIC"
PetscErrorCode ShellMatGetDiagonal_CYCLIC(Mat B,Vec diag)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(diag,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_CYCLIC"
PetscErrorCode SVDSetUp_CYCLIC(SVD svd)
{
  PetscErrorCode    ierr;
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC *)svd->data;
  PetscInt          M,N,m,n,i,j,start,end,ncols,*pos,nloc,isl;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscScalar       *pU;
  PetscTruth        trackall;
  Vec               v;
  PetscScalar       *isa,*va;

  PetscFunctionBegin;
  
  if (cyclic->mat) { 
    ierr = MatDestroy(cyclic->mat);CHKERRQ(ierr);
  }
  if (cyclic->x1) { 
    ierr = VecDestroy(cyclic->x1);CHKERRQ(ierr); 
    ierr = VecDestroy(cyclic->x2);CHKERRQ(ierr); 
    ierr = VecDestroy(cyclic->y1);CHKERRQ(ierr); 
    ierr = VecDestroy(cyclic->y2);CHKERRQ(ierr); 
  }

  ierr = SVDMatGetSize(svd,&M,&N);CHKERRQ(ierr);
  ierr = SVDMatGetLocalSize(svd,&m,&n);CHKERRQ(ierr);
  if (cyclic->explicitmatrix) {
    cyclic->x1 = cyclic->x2 = cyclic->y1 = cyclic->y2 = PETSC_NULL;
    ierr = MatCreate(((PetscObject)svd)->comm,&cyclic->mat);CHKERRQ(ierr);
    ierr = MatSetSizes(cyclic->mat,m+n,m+n,M+N,M+N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(cyclic->mat);CHKERRQ(ierr);
    if (svd->AT) {
      ierr = MatGetOwnershipRange(svd->AT,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(svd->AT,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        j = i + M;
        ierr = MatSetValues(cyclic->mat,1,&j,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(cyclic->mat,ncols,cols,1,&j,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(svd->AT,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscMalloc(sizeof(PetscInt)*n,&pos);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(svd->A,&start,&end);CHKERRQ(ierr);
      for (i=start;i<end;i++) {
        ierr = MatGetRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) 
          pos[j] = cols[j] + M;
        ierr = MatSetValues(cyclic->mat,1,&i,ncols,pos,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(cyclic->mat,ncols,pos,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(svd->A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      }
      ierr = PetscFree(pos);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(cyclic->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(cyclic->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,m,M,PETSC_NULL,&cyclic->x1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,n,N,PETSC_NULL,&cyclic->x2);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,m,M,PETSC_NULL,&cyclic->y1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,n,N,PETSC_NULL,&cyclic->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(((PetscObject)svd)->comm,m+n,m+n,M+N,M+N,svd,&cyclic->mat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(cyclic->mat,MATOP_MULT,(void(*)(void))ShellMatMult_CYCLIC);CHKERRQ(ierr);  
    ierr = MatShellSetOperation(cyclic->mat,MATOP_GET_DIAGONAL,(void(*)(void))ShellMatGetDiagonal_CYCLIC);CHKERRQ(ierr);  
  }

  ierr = EPSSetOperators(cyclic->eps,cyclic->mat,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(cyclic->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(cyclic->eps,svd->which == SVD_LARGEST ? EPS_LARGEST_REAL : EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSSetDimensions(cyclic->eps,svd->nsv,svd->ncv,svd->mpd);CHKERRQ(ierr);
  ierr = EPSSetTolerances(cyclic->eps,svd->tol,svd->max_it);CHKERRQ(ierr);
  /* Transfer the trackall option from svd to eps */
  ierr = SVDGetTrackAll(svd,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(cyclic->eps,trackall);CHKERRQ(ierr);
  /* Transfer the initial subspace from svd to eps */
  if (svd->nini < 0) {
    for (i=0; i<-svd->nini; i++) {
      ierr = MatGetVecs(cyclic->mat,&v,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecGetArray(v,&va);CHKERRQ(ierr);
      ierr = VecGetArray(svd->IS[i],&isa);CHKERRQ(ierr);
      ierr = VecGetSize(svd->IS[i],&isl);CHKERRQ(ierr);
      if (isl == m) {
        ierr = PetscMemcpy(va,isa,sizeof(PetscScalar)*m);CHKERRQ(ierr);
        ierr = PetscMemzero(&va[m],sizeof(PetscScalar)*n);CHKERRQ(ierr);
      } else if (isl == n) {
        ierr = PetscMemzero(va,sizeof(PetscScalar)*m);CHKERRQ(ierr);
        ierr = PetscMemcpy(&va[m],isa,sizeof(PetscScalar)*n);CHKERRQ(ierr);
      } else {
        SETERRQ(PETSC_ERR_SUP,"Size of the initial subspace vectors should match to some dimension of A");
      }
      ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
      ierr = VecRestoreArray(svd->IS[i],&isa);CHKERRQ(ierr);
      ierr = VecDestroy(svd->IS[i]);CHKERRQ(ierr);
      svd->IS[i] = v;
    }
    ierr = EPSSetInitialSpace(cyclic->eps,-svd->nini,svd->IS);CHKERRQ(ierr);
    for (i=0; i<-svd->nini; i++) {
      ierr = VecDestroy(svd->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->IS);CHKERRQ(ierr);
    svd->IS = PETSC_NULL;
    svd->nini = 0;
  }
  if (cyclic->setfromoptionscalled == PETSC_TRUE) {
    ierr = EPSSetFromOptions(cyclic->eps);CHKERRQ(ierr);
    cyclic->setfromoptionscalled = PETSC_FALSE;
  }
  ierr = EPSSetUp(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cyclic->eps,PETSC_NULL,&svd->ncv,&svd->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(cyclic->eps,&svd->tol,&svd->max_it);CHKERRQ(ierr);

  if (svd->ncv != svd->n) {
    if (svd->U) {  
      ierr = VecGetArray(svd->U[0],&pU);CHKERRQ(ierr);
      for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
      ierr = PetscFree(pU);CHKERRQ(ierr);
      ierr = PetscFree(svd->U);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,&nloc,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*nloc*sizeof(PetscScalar),&pU);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) {
      ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,pU+i*nloc,&svd->U[i]);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_CYCLIC"
PetscErrorCode SVDSolve_CYCLIC(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC *)svd->data;
  PetscInt       i,j,M,m,idx,start,end;
  PetscScalar    sigma,*px;
  Vec            x;
  IS             isU,isV;
  VecScatter     vsU,vsV;
  
  PetscFunctionBegin;
  ierr = EPSSolve(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cyclic->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cyclic->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);

  ierr = MatGetVecs(cyclic->mat,&x,PETSC_NULL);CHKERRQ(ierr);
  if (cyclic->explicitmatrix) {
    ierr = EPSGetOperationCounters(cyclic->eps,&svd->matvecs,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = SVDMatGetSize(svd,&M,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(svd->U[0],&start,&end);CHKERRQ(ierr);
    ierr = ISCreateBlock(((PetscObject)svd)->comm,end-start,1,&start,&isU);CHKERRQ(ierr);      
    ierr = VecScatterCreate(x,isU,svd->U[0],PETSC_NULL,&vsU);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(svd->V[0],&start,&end);CHKERRQ(ierr);
    idx = start + M;
    ierr = ISCreateBlock(((PetscObject)svd)->comm,end-start,1,&idx,&isV);CHKERRQ(ierr);      
    ierr = VecScatterCreate(x,isV,svd->V[0],PETSC_NULL,&vsV);CHKERRQ(ierr);

    for (i=0,j=0;i<svd->nconv;i++) {
      ierr = EPSGetEigenpair(cyclic->eps,i,&sigma,PETSC_NULL,x,PETSC_NULL);CHKERRQ(ierr);
      if (PetscRealPart(sigma) > 0.0) {
        svd->sigma[j] = PetscRealPart(sigma);
        ierr = VecScatterBegin(vsU,x,svd->U[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterBegin(vsV,x,svd->V[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(vsU,x,svd->U[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(vsV,x,svd->V[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScale(svd->U[j],1.0/sqrt(2.0));CHKERRQ(ierr);
        ierr = VecScale(svd->V[j],1.0/sqrt(2.0));CHKERRQ(ierr);	  	  
        j++;
      }
    }
      
    ierr = ISDestroy(isU);CHKERRQ(ierr);
    ierr = VecScatterDestroy(vsU);CHKERRQ(ierr);
    ierr = ISDestroy(isV);CHKERRQ(ierr);
    ierr = VecScatterDestroy(vsV);CHKERRQ(ierr);
  } else {
    ierr = SVDMatGetLocalSize(svd,&m,PETSC_NULL);CHKERRQ(ierr);
    for (i=0,j=0;i<svd->nconv;i++) {
      ierr = EPSGetEigenpair(cyclic->eps,i,&sigma,PETSC_NULL,x,PETSC_NULL);CHKERRQ(ierr);
      if (PetscRealPart(sigma) > 0.0) {
        svd->sigma[j] = PetscRealPart(sigma);
        ierr = VecGetArray(x,&px);CHKERRQ(ierr);
        ierr = VecPlaceArray(cyclic->x1,px);CHKERRQ(ierr);
        ierr = VecPlaceArray(cyclic->x2,px+m);CHKERRQ(ierr);
        
        ierr = VecCopy(cyclic->x1,svd->U[j]);CHKERRQ(ierr);
        ierr = VecScale(svd->U[j],1.0/sqrt(2.0));CHKERRQ(ierr);

        ierr = VecCopy(cyclic->x2,svd->V[j]);CHKERRQ(ierr);
        ierr = VecScale(svd->V[j],1.0/sqrt(2.0));CHKERRQ(ierr);	  
        
        ierr = VecResetArray(cyclic->x1);CHKERRQ(ierr);
        ierr = VecResetArray(cyclic->x2);CHKERRQ(ierr);
        ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
        j++;
      }
    }
  }
  svd->nconv = j;

  ierr = VecDestroy(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitor_CYCLIC"
PetscErrorCode SVDMonitor_CYCLIC(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,j;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0,j=0;i<nest;i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = STBackTransform(eps->OP, 1, &er, &ei); CHKERRQ(ierr);
    if (PetscRealPart(er) > 0.0) {
      svd->sigma[j] = PetscRealPart(er);
      svd->errest[j] = errest[i];
      if (errest[i] < svd->tol) nconv++;
      j++;
    }
  }
  nest = j;
  SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_CYCLIC"
PetscErrorCode SVDSetFromOptions_CYCLIC(SVD svd)
{
  PetscErrorCode ierr;
  PetscTruth     set,val;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC *)svd->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)svd)->comm,((PetscObject)svd)->prefix,"CYCLIC Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",cyclic->explicitmatrix,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDCyclicSetExplicitMatrix(svd,val);CHKERRQ(ierr);
  }
  if (cyclic->explicitmatrix) {
    /* don't build the transpose */
    if (svd->transmode == PETSC_DECIDE)
      svd->transmode = SVD_TRANSPOSE_IMPLICIT;
  } else {
    /* use as default an ST with shell matrix and Jacobi */ 
    ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,ST_MATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  cyclic->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicSetExplicitMatrix_CYCLIC"
PetscErrorCode SVDCyclicSetExplicitMatrix_CYCLIC(SVD svd,PetscTruth explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  cyclic->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicSetExplicitMatrix"
/*@
   SVDCyclicSetExplicitMatrix - Indicate if the eigensolver operator 
   H(A) = [ 0  A ; A^T 0 ] must be computed explicitly.

   Collective on SVD

   Input Parameters:
+  svd      - singular value solver
-  explicit - boolean flag indicating if H(A) is built explicitly

   Options Database Key:
.  -svd_cyclic_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: SVDCyclicGetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicSetExplicitMatrix(SVD svd,PetscTruth explicitmatrix)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,explicitmatrix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicGetExplicitMatrix_CYCLIC"
PetscErrorCode SVDCyclicGetExplicitMatrix_CYCLIC(SVD svd,PetscTruth *explicitmatrix)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(explicitmatrix,2);
  *explicitmatrix = cyclic->explicitmatrix;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SVDCyclicGetExplicitMatrix"
/*@
   SVDCyclicGetExplicitMatrix - Returns the flag indicating if H(A) is built explicitly

   Not collective

   Input Parameter:
.  svd  - singular value solver

   Output Parameter:
.  explicit - the mode flag

   Level: advanced

.seealso: SVDCyclicSetExplicitMatrix()
@*/
PetscErrorCode SVDCyclicGetExplicitMatrix(SVD svd,PetscTruth *explicitmatrix)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,explicitmatrix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicSetEPS_CYCLIC"
PetscErrorCode SVDCyclicSetEPS_CYCLIC(SVD svd,EPS eps)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(cyclic->eps);CHKERRQ(ierr);  
  cyclic->eps = eps;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicSetEPS"
/*@
   SVDCyclicSetEPS - Associate an eigensolver object (EPS) to the
   singular value solver. 

   Collective on SVD

   Input Parameters:
+  svd - singular value solver
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicGetEPS()
@*/
PetscErrorCode SVDCyclicSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDCyclicSetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicGetEPS_CYCLIC"
PetscErrorCode SVDCyclicGetEPS_CYCLIC(SVD svd,EPS *eps)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(eps,2);
  *eps = cyclic->eps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDCyclicGetEPS"
/*@
   SVDCyclicGetEPS - Retrieve the eigensolver object (EPS) associated
   to the singular value solver.

   Not Collective

   Input Parameter:
.  svd - singular value solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: SVDCyclicSetEPS()
@*/
PetscErrorCode SVDCyclicGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS *eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDCyclicGetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_CYCLIC"
PetscErrorCode SVDView_CYCLIC(SVD svd,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  if (cyclic->explicitmatrix) {
    ierr = PetscViewerASCIIPrintf(viewer,"cyclic matrix: explicit\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"cyclic matrix: implicit\n");CHKERRQ(ierr);
  }
  ierr = EPSView(cyclic->eps,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_CYCLIC"
PetscErrorCode SVDDestroy_CYCLIC(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC *cyclic = (SVD_CYCLIC *)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(cyclic->eps);CHKERRQ(ierr);
  if (cyclic->mat) { ierr = MatDestroy(cyclic->mat);CHKERRQ(ierr); }
  if (cyclic->x1) { 
    ierr = VecDestroy(cyclic->x1);CHKERRQ(ierr);
    ierr = VecDestroy(cyclic->x2);CHKERRQ(ierr);
    ierr = VecDestroy(cyclic->y1);CHKERRQ(ierr);
    ierr = VecDestroy(cyclic->y2);CHKERRQ(ierr);
  }
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicSetEPS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicGetEPS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_CYCLIC"
PetscErrorCode SVDCreate_CYCLIC(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_CYCLIC *cyclic;
  
  PetscFunctionBegin;
  ierr = PetscNew(SVD_CYCLIC,&cyclic);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_CYCLIC));
  svd->data                      = (void *)cyclic;
  svd->ops->solve                = SVDSolve_CYCLIC;
  svd->ops->setup                = SVDSetUp_CYCLIC;
  svd->ops->setfromoptions       = SVDSetFromOptions_CYCLIC;
  svd->ops->destroy              = SVDDestroy_CYCLIC;
  svd->ops->view                 = SVDView_CYCLIC;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicSetEPS_C","SVDCyclicSetEPS_CYCLIC",SVDCyclicSetEPS_CYCLIC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicGetEPS_C","SVDCyclicGetEPS_CYCLIC",SVDCyclicGetEPS_CYCLIC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C","SVDCyclicSetExplicitMatrix_CYCLIC",SVDCyclicSetExplicitMatrix_CYCLIC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C","SVDCyclicGetExplicitMatrix_CYCLIC",SVDCyclicGetExplicitMatrix_CYCLIC);CHKERRQ(ierr);

  ierr = EPSCreate(((PetscObject)svd)->comm,&cyclic->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(cyclic->eps,((PetscObject)svd)->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(cyclic->eps,"svd_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)cyclic->eps,(PetscObject)svd,1);CHKERRQ(ierr);  
  PetscLogObjectParent(svd,cyclic->eps);
  ierr = EPSSetIP(cyclic->eps,svd->ip);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSMonitorSet(cyclic->eps,SVDMonitor_CYCLIC,svd,PETSC_NULL);CHKERRQ(ierr);
  cyclic->explicitmatrix = PETSC_FALSE;
  cyclic->mat = PETSC_NULL;
  cyclic->x1 = PETSC_NULL;
  cyclic->x2 = PETSC_NULL;
  cyclic->y1 = PETSC_NULL;
  cyclic->y2 = PETSC_NULL;
  cyclic->setfromoptionscalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
