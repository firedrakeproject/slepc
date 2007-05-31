/*                       

   SLEPc singular value solver: "cyclic"

   Method: Uses a Hermitian eigensolver for H(A) = [ 0  A ; A^T 0 ]

   Last update: Jan 2007

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepceps.h"

typedef struct {
  PetscTruth explicitmatrix;
  EPS        eps;
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
  PetscInt          M,N,m,n,i,j,start,end,ncols,*pos;
  const PetscInt    *cols;
  const PetscScalar *vals;

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
    ierr = MatCreate(svd->comm,&cyclic->mat);CHKERRQ(ierr);
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
    ierr = VecCreateMPIWithArray(svd->comm,m,M,PETSC_NULL,&cyclic->x1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(svd->comm,n,N,PETSC_NULL,&cyclic->x2);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(svd->comm,m,M,PETSC_NULL,&cyclic->y1);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(svd->comm,n,N,PETSC_NULL,&cyclic->y2);CHKERRQ(ierr);
    ierr = MatCreateShell(svd->comm,m+n,m+n,M+N,M+N,svd,&cyclic->mat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(cyclic->mat,MATOP_MULT,(void(*)(void))ShellMatMult_CYCLIC);CHKERRQ(ierr);  
    ierr = MatShellSetOperation(cyclic->mat,MATOP_GET_DIAGONAL,(void(*)(void))ShellMatGetDiagonal_CYCLIC);CHKERRQ(ierr);  
  }

  ierr = EPSSetOperators(cyclic->eps,cyclic->mat,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(cyclic->eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetDimensions(cyclic->eps,svd->nsv,svd->ncv);CHKERRQ(ierr);
  ierr = EPSSetTolerances(cyclic->eps,svd->tol,svd->max_it);CHKERRQ(ierr);
  ierr = EPSSetUp(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(cyclic->eps,PETSC_NULL,&svd->ncv);CHKERRQ(ierr);
  ierr = EPSGetTolerances(cyclic->eps,&svd->tol,&svd->max_it);CHKERRQ(ierr);

  if (svd->U) {  
    for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
  for (i=0;i<svd->ncv;i++) { ierr = SVDMatGetVecs(svd,PETSC_NULL,svd->U+i);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_CYCLIC"
PetscErrorCode SVDSolve_CYCLIC(SVD svd)
{
  PetscErrorCode ierr;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC *)svd->data;
  int            i,j;
  PetscInt       M,m,idx,start,end;
  PetscScalar    sigma,*px;
  Vec            x;
  IS             isU,isV;
  VecScatter     vsU,vsV;
  
  PetscFunctionBegin;
  ierr = EPSSetWhichEigenpairs(cyclic->eps,svd->which == SVD_LARGEST ? EPS_LARGEST_REAL : EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSSolve(cyclic->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(cyclic->eps,&svd->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(cyclic->eps,&svd->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason);CHKERRQ(ierr);

  ierr = MatGetVecs(cyclic->mat,&x,PETSC_NULL);CHKERRQ(ierr);
  if (cyclic->explicitmatrix) {
    ierr = EPSGetOperationCounters(cyclic->eps,&svd->matvecs,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = SVDMatGetSize(svd,&M,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(svd->U[0],&start,&end);CHKERRQ(ierr);
    ierr = ISCreateBlock(svd->comm,end-start,1,&start,&isU);CHKERRQ(ierr);      
    ierr = VecScatterCreate(x,isU,svd->U[0],PETSC_NULL,&vsU);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(svd->V[0],&start,&end);CHKERRQ(ierr);
    idx = start + M;
    ierr = ISCreateBlock(svd->comm,end-start,1,&idx,&isV);CHKERRQ(ierr);      
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
PetscErrorCode SVDMonitor_CYCLIC(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *ctx)
{
  int        i,j;
  SVD        svd = (SVD)ctx;

  PetscFunctionBegin;
  nconv = 0;
  for (i=0,j=0;i<nest;i++) {
    if (PetscRealPart(eigr[i]) > 0.0) {
      svd->sigma[j] = PetscRealPart(eigr[i]);
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
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC *)svd->data;
  ST             st;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"CYCLIC Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",PETSC_FALSE,&cyclic->explicitmatrix,PETSC_NULL);CHKERRQ(ierr);
  if (cyclic->explicitmatrix) {
    /* don't build the transpose */
    if (svd->transmode == PETSC_DECIDE)
      svd->transmode = SVD_TRANSPOSE_IMPLICIT;
  } else {
    /* use as default an ST with shell matrix and Jacobi */ 
    ierr = EPSGetST(cyclic->eps,&st);CHKERRQ(ierr);
    ierr = STSetMatMode(st,STMATMODE_SHELL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = EPSSetFromOptions(cyclic->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
/*@C
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
/*@C
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

  ierr = EPSCreate(svd->comm,&cyclic->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(cyclic->eps,svd->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(cyclic->eps,"svd_");CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}
EXTERN_C_END
