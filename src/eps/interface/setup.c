/*
      EPS routines related to problem setup.

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

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Level: advanced

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   This function sets a random initial vector if none has been provided.

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp()
@*/
PetscErrorCode EPSSetUp(EPS eps)
{
  PetscErrorCode ierr;
  Vec            v0,w0;  
  Mat            A,B; 
  PetscInt       N;
  PetscTruth     isCayley;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (eps->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!((PetscObject)eps)->type_name) {
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  }
  
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  /* Set default problem type */
  if (!eps->problem_type) {
    if (B==PETSC_NULL) {
      ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
    }
    else {
      ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
    }
  } else if ((B && !eps->isgeneralized) || (!B && eps->isgeneralized)) {
    SETERRQ(0,"Warning: Inconsistent EPS state"); 
  }
  
  if (eps->ispositive) {
    ierr = STGetBilinearForm(eps->OP,&B);CHKERRQ(ierr);
    ierr = IPSetBilinearForm(eps->ip,B,IPINNER_HERMITIAN);CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  } else {
    ierr = IPSetBilinearForm(eps->ip,PETSC_NULL,IPINNER_HERMITIAN);CHKERRQ(ierr);
  }
  
  /* Create random initial vectors if not set */
  /* right */
  ierr = EPSGetInitialVector(eps,&v0);CHKERRQ(ierr);
  if (!v0) {
    ierr = MatGetVecs(A,&v0,PETSC_NULL);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(v0);CHKERRQ(ierr);
    eps->vec_initial = v0;
  }
  /* left */
  ierr = EPSGetLeftInitialVector(eps,&w0);CHKERRQ(ierr);
  if (!w0) {
    ierr = MatGetVecs(A,PETSC_NULL,&w0);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(w0);CHKERRQ(ierr);
    eps->vec_initial_left = w0;
  }

  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv > N) eps->ncv = N;

  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);
  ierr = STSetUp(eps->OP); CHKERRQ(ierr); 
  
  ierr = PetscTypeCompare((PetscObject)eps->OP,STCAYLEY,&isCayley);CHKERRQ(ierr);
  if (isCayley && eps->problem_type == EPS_PGNHEP) {
    SETERRQ(PETSC_ERR_SUP,"Cayley spectral transformation is not compatible with PGNHEP");
  }

  if (eps->nds>0) {
    if (!eps->ds_ortho) {
      /* orthonormalize vectors in DS if necessary */
      ierr = IPQRDecomposition(eps->ip,eps->DS,0,eps->nds,PETSC_NULL,0);CHKERRQ(ierr);
    }
    ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,eps->nds,PETSC_NULL,eps->DS,eps->vec_initial,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
  }

  ierr = STCheckNullSpace(eps->OP,eps->nds,eps->DS);CHKERRQ(ierr);

  /* Build balancing matrix if required */
  if (!eps->balance) eps->balance = EPSBALANCE_NONE;
  if (!eps->ishermitian && (eps->balance==EPSBALANCE_ONESIDE || eps->balance==EPSBALANCE_TWOSIDE)) {
    if (!eps->D) {
      ierr = VecDuplicate(eps->vec_initial,&eps->D);CHKERRQ(ierr);
    }
    ierr = EPSBuildBalance_Krylov(eps);CHKERRQ(ierr);
    ierr = STSetBalanceMatrix(eps->OP,eps->D);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialVector"
/*@
   EPSSetInitialVector - Sets the initial vector from which the 
   eigensolver starts to iterate.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: EPSGetInitialVector(), EPSSetLeftInitialVector()

@*/
PetscErrorCode EPSSetInitialVector(EPS eps,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial); CHKERRQ(ierr);
  }
  eps->vec_initial = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInitialVector"
/*@
   EPSGetInitialVector - Gets the initial vector associated with the 
   eigensolver; if the vector was not set it will return a 0 pointer or
   a vector randomly generated by EPSSetUp().

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: EPSSetInitialVector(), EPSGetLeftInitialVector()

@*/
PetscErrorCode EPSGetInitialVector(EPS eps,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,2);
  *vec = eps->vec_initial;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetLeftInitialVector"
/*@
   EPSSetLeftInitialVector - Sets the initial vector from which the eigensolver 
   starts to iterate, corresponding to the left recurrence (two-sided solvers).

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: EPSGetLeftInitialVector(), EPSSetInitialVector()

@*/
PetscErrorCode EPSSetLeftInitialVector(EPS eps,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->vec_initial_left) {
    ierr = VecDestroy(eps->vec_initial_left); CHKERRQ(ierr);
  }
  eps->vec_initial_left = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetLeftInitialVector"
/*@
   EPSGetLeftInitialVector - Gets the left initial vector associated with the 
   eigensolver; if the vector was not set it will return a 0 pointer or
   a vector randomly generated by EPSSetUp().

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: EPSSetLeftInitialVector(), EPSGetLeftInitialVector()

@*/
PetscErrorCode EPSGetLeftInitialVector(EPS eps,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,2);
  *vec = eps->vec_initial_left;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOperators"
/*@
   EPSSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on EPS and Mat

   Input Parameters:
+  eps - the eigenproblem solver context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes: 
   To specify a standard eigenproblem, use PETSC_NULL for parameter B.

   Level: beginner

.seealso: EPSSolve(), EPSGetST(), STGetOperators()
@*/
PetscErrorCode EPSSetOperators(EPS eps,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(eps,1,A,2);
  if (B) PetscCheckSameComm(eps,1,B,3);

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"A is a non-square matrix"); }
  if (B) { 
    ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
    if (m!=n) { SETERRQ(1,"B is a non-square matrix"); }
  }

  ierr = STSetOperators(eps->OP,A,B);CHKERRQ(ierr);
  eps->setupcalled = 0;  /* so that next solve call will call setup */

  /* Destroy randomly generated initial vectors */
  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial);CHKERRQ(ierr);
    eps->vec_initial = PETSC_NULL;
  }
  if (eps->vec_initial_left) {
    ierr = VecDestroy(eps->vec_initial_left);CHKERRQ(ierr);
    eps->vec_initial_left = PETSC_NULL;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetOperators"
/*@
   EPSGetOperators - Gets the matrices associated with the eigensystem.

   Collective on EPS and Mat

   Input Parameter:
.  eps - the EPS context

   Output Parameters:
+  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Level: intermediate

.seealso: EPSSolve(), EPSGetST(), STGetOperators(), STSetOperators()
@*/
PetscErrorCode EPSGetOperators(EPS eps, Mat *A, Mat *B)
{
  PetscErrorCode ierr;
  ST             st;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (A) PetscValidPointer(A,2);
  if (B) PetscValidPointer(B,3);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetOperators(st,A,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSAttachDeflationSpace"
/*@
   EPSAttachDeflationSpace - Add vectors to the basis of the deflation space.

   Not Collective

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors to add
.  ds    - set of basis vectors of the deflation space
-  ortho - PETSC_TRUE if basis vectors of deflation space are orthonormal

   Notes:
   When a deflation space is given, the eigensolver seeks the eigensolution
   in the restriction of the problem to the orthogonal complement of this
   space. This can be used for instance in the case that an invariant 
   subspace is known beforehand (such as the nullspace of the matrix).

   The basis vectors can be provided all at once or incrementally with
   several calls to EPSAttachDeflationSpace().

   Use a value of PETSC_TRUE for parameter ortho if all the vectors passed
   in are known to be mutually orthonormal.

   Level: intermediate

.seealso: EPSRemoveDeflationSpace()
@*/
PetscErrorCode EPSAttachDeflationSpace(EPS eps,PetscInt n,Vec *ds,PetscTruth ortho)
{
  PetscErrorCode ierr;
  PetscInt       i,nloc;
  Vec            *newds;
  PetscScalar    *pV;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (n<=0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  /* allocate space for previous and new vectors */
  ierr = PetscMalloc((n+eps->nds)*sizeof(Vec), &newds);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ds[0],&nloc);CHKERRQ(ierr);
  ierr = PetscMalloc((n+eps->nds)*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
  for (i=0;i<n+eps->nds;i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,nloc,PETSC_DECIDE,pV+i*nloc,&newds[i]);CHKERRQ(ierr);
  }
  /* copy and free previous vectors */
  if (eps->nds > 0) {
    ierr = VecGetArray(eps->DS[0],&pV);CHKERRQ(ierr);
    for (i=0;i<eps->nds;i++) {
      ierr = VecCopy(eps->DS[i],newds[i]);CHKERRQ(ierr);
      ierr = VecDestroy(eps->DS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(eps->DS);CHKERRQ(ierr);
  }
  /* copy new vectors */
  eps->DS = newds;
  for (i=0; i<n; i++) {
    ierr = VecCopy(ds[i],eps->DS[i + eps->nds]);CHKERRQ(ierr);
  }
  eps->nds += n;
  if (!ortho) eps->ds_ortho = PETSC_FALSE;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRemoveDeflationSpace"
/*@
   EPSRemoveDeflationSpace - Removes the deflation space.

   Not Collective

   Input Parameter:
.  eps   - the eigenproblem solver context

   Level: intermediate

.seealso: EPSAttachDeflationSpace()
@*/
PetscErrorCode EPSRemoveDeflationSpace(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *pV;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->nds > 0) {
    ierr = VecGetArray(eps->DS[0],&pV);CHKERRQ(ierr);
    for (i=0;i<eps->nds;i++) {
      ierr = VecDestroy(eps->DS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(eps->DS);CHKERRQ(ierr);
  }
  eps->ds_ortho = PETSC_TRUE;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}
