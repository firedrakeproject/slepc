/*
      QEP routines related to problem setup.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>       /*I "slepcqep.h" I*/
#include <slepc-private/ipimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp"
/*@
   QEPSetUp - Sets up all the internal data structures necessary for the
   execution of the QEP solver.

   Collective on QEP

   Input Parameter:
.  qep   - solver context

   Notes:
   This function need not be called explicitly in most cases, since QEPSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   Level: advanced

.seealso: QEPCreate(), QEPSolve(), QEPDestroy()
@*/
PetscErrorCode QEPSetUp(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  PetscBool      khas,mhas,lindep,islinear,flg;
  PetscReal      knorm,mnorm,norm;
  Mat            mat[3];
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (qep->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(QEP_SetUp,qep,0,0,0);CHKERRQ(ierr);

  /* reset the convergence flag from the previous solves */
  qep->reason = QEP_CONVERGED_ITERATING;

  /* Set default solver type (QEPSetFromOptions was not called) */
  if (!((PetscObject)qep)->type_name) {
    ierr = QEPSetType(qep,QEPLINEAR);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)qep,QEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    if (!qep->st) { ierr = QEPGetST(qep,&qep->st);CHKERRQ(ierr); }
    if (!((PetscObject)qep->st)->type_name) {
      ierr = STSetType(qep->st,STSHIFT);CHKERRQ(ierr);
    }
  }
  if (!qep->ip) { ierr = QEPGetIP(qep,&qep->ip);CHKERRQ(ierr); }
  if (!((PetscObject)qep->ip)->type_name) {
    ierr = IPSetDefaultType_Private(qep->ip);CHKERRQ(ierr);
  }
  if (!qep->ds) { ierr = QEPGetDS(qep,&qep->ds);CHKERRQ(ierr); }
  ierr = DSReset(qep->ds);CHKERRQ(ierr);
  if (!((PetscObject)qep->rand)->type_name) {
    ierr = PetscRandomSetFromOptions(qep->rand);CHKERRQ(ierr);
  }

  /* Check matrices, transfer them to ST */
  if (!qep->M || !qep->C || !qep->K) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONGSTATE,"QEPSetOperators must be called first"); 
  if (!islinear) {
    mat[0] = qep->K;
    mat[1] = qep->C;
    mat[2] = qep->M;
    ierr = STSetOperators(qep->st,3,mat);CHKERRQ(ierr);
  }
  
  /* Set problem dimensions */
  ierr = MatGetSize(qep->M,&qep->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(qep->M,&qep->nloc,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&qep->t);CHKERRQ(ierr);
  ierr = SlepcMatGetVecsTemplate(qep->M,&qep->t,PETSC_NULL);CHKERRQ(ierr);

  /* Set default problem type */
  if (!qep->problem_type) {
    ierr = QEPSetProblemType(qep,QEP_GENERAL);CHKERRQ(ierr);
  }

  /* Compute scaling factor if not set by user */
  if (qep->sfactor==0.0) {
    ierr = MatHasOperation(qep->K,MATOP_NORM,&khas);CHKERRQ(ierr);
    ierr = MatHasOperation(qep->M,MATOP_NORM,&mhas);CHKERRQ(ierr);
    if (khas && mhas) {
      ierr = MatNorm(qep->K,NORM_INFINITY,&knorm);CHKERRQ(ierr);
      ierr = MatNorm(qep->M,NORM_INFINITY,&mnorm);CHKERRQ(ierr);
      qep->sfactor = PetscSqrtReal(knorm/mnorm);
    } else qep->sfactor = 1.0;
  }

  /* Call specific solver setup */
  ierr = (*qep->ops->setup)(qep);CHKERRQ(ierr);

  /* set tolerance if not yet set */
  if (qep->tol==PETSC_DEFAULT) qep->tol = SLEPC_DEFAULT_TOL;

  /* set eigenvalue comparison */
  switch (qep->which) {
    case QEP_LARGEST_MAGNITUDE:
      qep->which_func = SlepcCompareLargestMagnitude;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_SMALLEST_MAGNITUDE:
      qep->which_func = SlepcCompareSmallestMagnitude;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_LARGEST_REAL:
      qep->which_func = SlepcCompareLargestReal;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_SMALLEST_REAL:
      qep->which_func = SlepcCompareSmallestReal;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_LARGEST_IMAGINARY:
      qep->which_func = SlepcCompareLargestImaginary;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_SMALLEST_IMAGINARY:
      qep->which_func = SlepcCompareSmallestImaginary;
      qep->which_ctx  = PETSC_NULL;
      break;
    case QEP_TARGET_MAGNITUDE:
      qep->which_func = SlepcCompareTargetMagnitude;
      qep->which_ctx  = &qep->target;
      break;
    case QEP_TARGET_REAL:
      qep->which_func = SlepcCompareTargetReal;
      qep->which_ctx  = &qep->target;
      break;
    case QEP_TARGET_IMAGINARY:
      qep->which_func = SlepcCompareTargetImaginary;
      qep->which_ctx  = &qep->target;
      break;
  }

  if (qep->ncv > 2*qep->n) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"ncv must be twice the problem size at most");
  if (qep->nev > qep->ncv) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"nev bigger than ncv");

  /* Setup ST */
  if (!islinear) {
    ierr = PetscObjectTypeCompareAny((PetscObject)qep->st,&flg,STSHIFT,STSINVERT,"");CHKERRQ(ierr);
    if (!flg) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_SUP,"Only STSHIFT and STSINVERT spectral transformations can be used in QEP");
    ierr = STSetUp(qep->st);CHKERRQ(ierr);
  }

  /* process initial vectors */
  if (qep->nini<0) {
    qep->nini = -qep->nini;
    if (qep->nini>qep->ncv) SETERRQ(((PetscObject)qep)->comm,1,"The number of initial vectors is larger than ncv");
    k = 0;
    for (i=0;i<qep->nini;i++) {
      ierr = VecCopy(qep->IS[i],qep->V[k]);CHKERRQ(ierr);
      ierr = VecDestroy(&qep->IS[i]);CHKERRQ(ierr);
      ierr = IPOrthogonalize(qep->ip,0,PETSC_NULL,k,PETSC_NULL,qep->V,qep->V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
      if (norm==0.0 || lindep) { ierr = PetscInfo(qep,"Linearly dependent initial vector found, removing...\n");CHKERRQ(ierr); }
      else {
        ierr = VecScale(qep->V[k],1.0/norm);CHKERRQ(ierr);
        k++;
      }
    }
    qep->nini = k;
    ierr = PetscFree(qep->IS);CHKERRQ(ierr);
  }
  if (qep->ninil<0) {
    if (!qep->leftvecs) { ierr = PetscInfo(qep,"Ignoring initial left vectors\n");CHKERRQ(ierr); }
    else {
      qep->ninil = -qep->ninil;
      if (qep->ninil>qep->ncv) SETERRQ(((PetscObject)qep)->comm,1,"The number of initial left vectors is larger than ncv");
      k = 0;
      for (i=0;i<qep->ninil;i++) {
        ierr = VecCopy(qep->ISL[i],qep->W[k]);CHKERRQ(ierr);
        ierr = VecDestroy(&qep->ISL[i]);CHKERRQ(ierr);
        ierr = IPOrthogonalize(qep->ip,0,PETSC_NULL,k,PETSC_NULL,qep->W,qep->W[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
        if (norm==0.0 || lindep) { ierr = PetscInfo(qep,"Linearly dependent initial left vector found, removing...\n");CHKERRQ(ierr); }
        else {
          ierr = VecScale(qep->W[k],1.0/norm);CHKERRQ(ierr);
          k++;
        }
      }
      qep->ninil = k;
      ierr = PetscFree(qep->ISL);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(QEP_SetUp,qep,0,0,0);CHKERRQ(ierr);
  qep->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetOperators"
/*@
   QEPSetOperators - Sets the matrices associated with the quadratic eigenvalue problem.

   Collective on QEP and Mat

   Input Parameters:
+  qep - the eigenproblem solver context
.  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Notes: 
   The quadratic eigenproblem is defined as (l^2*M + l*C + K)*x = 0, where l is
   the eigenvalue and x is the eigenvector.

   Level: beginner

.seealso: QEPSolve(), QEPGetOperators()
@*/
PetscErrorCode QEPSetOperators(QEP qep,Mat M,Mat C,Mat K)
{
  PetscErrorCode ierr;
  PetscInt       m,n,m0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidHeaderSpecific(M,MAT_CLASSID,2);
  PetscValidHeaderSpecific(C,MAT_CLASSID,3);
  PetscValidHeaderSpecific(K,MAT_CLASSID,4);
  PetscCheckSameComm(qep,1,M,2);
  PetscCheckSameComm(qep,1,C,3);
  PetscCheckSameComm(qep,1,K,4);

  /* Check for square matrices */
  ierr = MatGetSize(M,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONG,"M is a non-square matrix");
  m0=m;
  ierr = MatGetSize(C,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONG,"C is a non-square matrix");
  if (m!=m0) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of M and C do not match");
  ierr = MatGetSize(K,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONG,"K is a non-square matrix");
  if (m!=m0) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of M and K do not match");

  /* Store a copy of the matrices */
  if (qep->setupcalled) { ierr = QEPReset(qep);CHKERRQ(ierr); }
  ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
  ierr = MatDestroy(&qep->M);CHKERRQ(ierr);
  qep->M = M;
  ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  ierr = MatDestroy(&qep->C);CHKERRQ(ierr);
  qep->C = C;
  ierr = PetscObjectReference((PetscObject)K);CHKERRQ(ierr);
  ierr = MatDestroy(&qep->K);CHKERRQ(ierr);
  qep->K = K;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetOperators"
/*@
   QEPGetOperators - Gets the matrices associated with the quadratic eigensystem.

   Collective on QEP and Mat

   Input Parameter:
.  qep - the QEP context

   Output Parameters:
+  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Level: intermediate

.seealso: QEPSolve(), QEPSetOperators()
@*/
PetscErrorCode QEPGetOperators(QEP qep,Mat *M,Mat *C,Mat *K)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (M) { PetscValidPointer(M,2); *M = qep->M; }
  if (C) { PetscValidPointer(C,3); *C = qep->C; }
  if (K) { PetscValidPointer(K,4); *K = qep->K; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetInitialSpace"
/*@
   QEPSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on QEP and Vec

   Input Parameter:
+  qep   - the quadratic eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one QEPSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: QEPSetInitialSpaceLeft()
@*/
PetscErrorCode QEPSetInitialSpace(QEP qep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(qep,n,2);
  if (n<0) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (qep->nini<0) {
    for (i=0;i<-qep->nini;i++) {
      ierr = VecDestroy(&qep->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(qep->IS);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&qep->IS);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    qep->IS[i] = is[i];
  }

  qep->nini = -n;
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetInitialSpaceLeft"
/*@
   QEPSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, the subspace from which the solver starts to iterate for
   building the left subspace (in methods that work with two subspaces).

   Collective on QEP and Vec

   Input Parameter:
+  qep   - the quadratic eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial left space

   Notes:
   Some solvers start to iterate on a single vector (initial left vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one QEPSolve() call to the other, so the
   initial left space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted left eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: QEPSetInitialSpace()
@*/
PetscErrorCode QEPSetInitialSpaceLeft(QEP qep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(qep,n,2);
  if (n<0) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (qep->ninil<0) {
    for (i=0;i<-qep->ninil;i++) {
      ierr = VecDestroy(&qep->ISL[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(qep->ISL);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&qep->ISL);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    qep->ISL[i] = is[i];
  }

  qep->ninil = -n;
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPAllocateSolution"
/*
  QEPAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. All vectors in V (and W) share a
  contiguous chunk of memory.
*/
PetscErrorCode QEPAllocateSolution(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       newc,cnt;
  
  PetscFunctionBegin;
  if (qep->allocated_ncv != qep->ncv) {
    newc = PetscMax(0,qep->ncv-qep->allocated_ncv);
    ierr = QEPFreeSolution(qep);CHKERRQ(ierr);
    cnt = 0;
    ierr = PetscMalloc(qep->ncv*sizeof(PetscScalar),&qep->eigr);CHKERRQ(ierr);
    ierr = PetscMalloc(qep->ncv*sizeof(PetscScalar),&qep->eigi);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscScalar);
    ierr = PetscMalloc(qep->ncv*sizeof(PetscReal),&qep->errest);CHKERRQ(ierr);
    ierr = PetscMalloc(qep->ncv*sizeof(PetscInt),&qep->perm);CHKERRQ(ierr);
    cnt += 2*newc*sizeof(PetscReal);
    ierr = PetscLogObjectMemory(qep,cnt);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(qep->t,qep->ncv,&qep->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(qep,qep->ncv,qep->V);CHKERRQ(ierr);
    qep->allocated_ncv = qep->ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPFreeSolution"
/*
  QEPFreeSolution - Free memory storage. This routine is related to 
  QEPAllocateSolution().
*/
PetscErrorCode QEPFreeSolution(QEP qep)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (qep->allocated_ncv > 0) {
    ierr = PetscFree(qep->eigr);CHKERRQ(ierr);
    ierr = PetscFree(qep->eigi);CHKERRQ(ierr);
    ierr = PetscFree(qep->errest);CHKERRQ(ierr); 
    ierr = PetscFree(qep->perm);CHKERRQ(ierr); 
    ierr = VecDestroyVecs(qep->allocated_ncv,&qep->V);CHKERRQ(ierr);
    qep->allocated_ncv = 0;
  }
  PetscFunctionReturn(0);
}

