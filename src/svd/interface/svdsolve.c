/*
      SVD routines related to the solution process.
*/
#include "src/svd/svdimpl.h"   /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve"
/*@
   SVDSolve - Solves the singular value problem.

   Collective on SVD

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Options Database:
.   -svd_view - print information about the solver used

   Level: beginner

.seealso: SVDCreate(), SVDSetUp(), SVDDestroy() 
@*/
PetscErrorCode SVDSolve(SVD svd) 
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);

  if (!svd->setupcalled) { ierr = SVDSetUp(svd);CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);
  ierr = (*svd->ops->solve)(svd);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SVD_Solve,svd,0,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(svd->prefix,"-svd_view",&flg);CHKERRQ(ierr); 
  if (flg && !PetscPreLoadingOn) { ierr = SVDView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetConverged"
/*@
   SVDGetConverged - Gets the number of converged singular values.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context
  
   Output Parameter:
.  nconv - number of converged singular values 

   Note:
   This function should be called after SVDSolve() has finished.

   Level: beginner

@*/
PetscErrorCode SVDGetConverged(SVD svd,int *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidIntPointer(nconv,2);
  if (svd->nconv < 0) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSolve must be called first"); 
  }
  *nconv = svd->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetSingularTriplet" 
/*@
   SVDGetSingularTriplet - Gets the i-th triplet of the singular value decomposition
   as computed by SVDSolve(). The solution consists in the singular value and its left 
   and right singular vectors.

   Not Collective

   Input Parameters:
+  svd - singular value solver context 
-  i   - index of the solution

   Output Parameters:
+  sigma - singular value
.  U     - left singular vector
-  V     - right singular vector

   The index i should be a value between 0 and nconv (see SVDGetConverged()).
   Both U or V can be PETSC_NULL if singular vectors are not required. 

   Level: beginner

.seealso: SVDSolve(),  SVDGetConverged()
@*/
PetscErrorCode SVDGetSingularTriplet(SVD svd, int i, PetscReal *sigma, Vec U, Vec V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  PetscValidPointer(sigma,3);
  if (svd->nconv < 0) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "SVDSolve must be called first"); 
  }
  if (i<0 || i>=svd->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  *sigma = svd->sigma[i];
  if (U) {
    PetscValidHeaderSpecific(U,VEC_COOKIE,4);
//    ierr = VecCopy(svd->U[i],U);CHKERRQ(ierr);
  }
  if (V) {
    PetscValidHeaderSpecific(V,VEC_COOKIE,5);   
//    ierr = VecCopy(svd->V[i],V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
