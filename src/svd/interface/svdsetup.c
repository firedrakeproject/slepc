/*
     SVD routines for setting up the solver.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/svdimpl.h"      /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSetOperator"
/*@
   SVDSetOperator - Set the matrix associated with the singular value problem.

   Collective on SVD and Mat

   Input Parameters:
+  svd - the singular value solver context
-  A  - the matrix associated with the singular value problem

   Level: beginner

.seealso: SVDSolve(), SVDGetOperator()
@*/
PetscErrorCode SVDSetOperator(SVD svd,Mat mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCheckSameComm(svd,1,mat,2);
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  if (svd->OP) {
    ierr = MatDestroy(svd->OP);CHKERRQ(ierr);
  }
  svd->OP = mat;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetOperator"
/*@
   SVDGetOperator - Get the matrix associated with the singular value problem.

   Not collective, though parallel Mats are returned if the SVD is parallel

   Input Parameter:
.  svd - the singular value solver context

   Output Parameters:
.  A    - the matrix associated with the singular value problem

   Level: advanced

.seealso: SVDSolve(), SVDSetOperator()
@*/
PetscErrorCode SVDGetOperator(SVD svd,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(A,2);
  *A = svd->OP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp"
/*@
   SVDSetUp - Sets up all the internal data structures necessary for the
   execution of the singular value solver.

   Collective on SVD

   Input Parameter:
.  svd   - singular value solver context

   Level: advanced

   Notes:
   This function need not be called explicitly in most cases, since SVDSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

.seealso: SVDCreate(), SVDSolve(), SVDDestroy()
@*/
PetscErrorCode SVDSetUp(SVD svd)
{
  PetscErrorCode ierr;
  PetscTruth     flg,lindep;
  PetscInt       i,k,M,N,nloc;
  PetscScalar    *pV;
  PetscReal      norm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!((PetscObject)svd)->type_name) {
    ierr = SVDSetType(svd,SVDCROSS);CHKERRQ(ierr);
  }

  /* check matrix */
  if (!svd->OP)
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSetOperator must be called first"); 
  
  /* determine how to build the transpose */
  if (svd->transmode == PETSC_DECIDE) {
    ierr = MatHasOperation(svd->OP,MATOP_TRANSPOSE,&flg);CHKERRQ(ierr);    
    if (flg) svd->transmode = SVD_TRANSPOSE_EXPLICIT;
    else svd->transmode = SVD_TRANSPOSE_IMPLICIT;
  }
  
  /* build transpose matrix */
  if (svd->A) { ierr = MatDestroy(svd->A);CHKERRQ(ierr); }
  if (svd->AT) { ierr = MatDestroy(svd->AT);CHKERRQ(ierr); }
  ierr = MatGetSize(svd->OP,&M,&N);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)svd->OP);CHKERRQ(ierr);
  switch (svd->transmode) {
    case SVD_TRANSPOSE_EXPLICIT:
      ierr = MatHasOperation(svd->OP,MATOP_TRANSPOSE,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(1,"Matrix has not defined the MatTranpose operation");
      if (M>=N) {
        svd->A = svd->OP;
        ierr = MatTranspose(svd->OP, MAT_INITIAL_MATRIX,&svd->AT);CHKERRQ(ierr);
      } else {
        ierr = MatTranspose(svd->OP, MAT_INITIAL_MATRIX,&svd->A);CHKERRQ(ierr);
        svd->AT = svd->OP;
      }
      break;
    case SVD_TRANSPOSE_IMPLICIT:
      ierr = MatHasOperation(svd->OP,MATOP_MULT_TRANSPOSE,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(1,"Matrix has not defined the MatMultTranpose operation");
      if (M>=N) {
        svd->A = svd->OP;
        svd->AT = PETSC_NULL;    
      } else {
        svd->A = PETSC_NULL;
        svd->AT = svd->OP;
      }
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid transpose mode"); 
  }

  /* initialize the random number generator */
  ierr = PetscRandomCreate(((PetscObject)svd)->comm,&svd->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(svd->rand);CHKERRQ(ierr);

  /* call specific solver setup */
  ierr = (*svd->ops->setup)(svd);CHKERRQ(ierr);

  if (svd->ncv > M || svd->ncv > N)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"ncv bigger than matrix dimensions");
  if (svd->nsv > svd->ncv)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nsv bigger than ncv");

  if (svd->ncv != svd->n) {
    /* free memory for previous solution  */
    if (svd->n) { 
      ierr = PetscFree(svd->sigma);CHKERRQ(ierr);
      ierr = PetscFree(svd->perm);CHKERRQ(ierr);
      ierr = PetscFree(svd->errest);CHKERRQ(ierr);
      ierr = VecGetArray(svd->V[0],&pV);CHKERRQ(ierr);
      for (i=0;i<svd->n;i++) {
        ierr = VecDestroy(svd->V[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(pV);CHKERRQ(ierr);
      ierr = PetscFree(svd->V);CHKERRQ(ierr);
    }
    /* allocate memory for next solution */
    ierr = PetscMalloc(svd->ncv*sizeof(PetscReal),&svd->sigma);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*sizeof(PetscInt),&svd->perm);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*sizeof(PetscReal),&svd->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(svd->ncv*sizeof(Vec),&svd->V);CHKERRQ(ierr);
    if (svd->A) {
      ierr = MatGetLocalSize(svd->A,PETSC_NULL,&nloc);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalSize(svd->AT,&nloc,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(svd->ncv*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) {
      ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,pV+i*nloc,&svd->V[i]);CHKERRQ(ierr);
    }
    svd->n = svd->ncv;
  }

  /* process initial vectors */
  if (svd->nini<0) {
    svd->nini = -svd->nini;
    if (svd->nini>svd->ncv) SETERRQ(1,"The number of initial vectors is larger than ncv")
    k = 0;
    for (i=0;i<svd->nini;i++) {
      ierr = VecCopy(svd->IS[i],svd->V[k]);CHKERRQ(ierr);
      ierr = VecDestroy(svd->IS[i]);CHKERRQ(ierr);
      ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,k,PETSC_NULL,svd->V,svd->V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
      if (norm==0.0 || lindep) PetscInfo(svd,"Linearly dependent initial vector found, removing...\n");
      else {
        ierr = VecScale(svd->V[k],1.0/norm);CHKERRQ(ierr);
        k++;
      }
    }
    svd->nini = k;
    ierr = PetscFree(svd->IS);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(SVD_SetUp,svd,0,0,0);CHKERRQ(ierr);
  svd->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetInitialSpace"
/*@
   SVDSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on SVD and Vec

   Input Parameter:
+  svd   - the singular value solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one SVDSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted singular space. Then, convergence may be faster.

   Level: intermediate
@*/
PetscErrorCode SVDSetInitialSpace(SVD svd,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (n<0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (svd->nini<0) {
    for (i=0;i<-svd->nini;i++) {
      ierr = VecDestroy(svd->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->IS);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&svd->IS);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    svd->IS[i] = is[i];
  }

  svd->nini = -n;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}

