/*
      EPS routines related to problem setup.

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

#include <private/epsimpl.h>       /*I "slepceps.h" I*/
#include <private/ipimpl.h>        /*I "slepcip.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   Level: advanced

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp(), EPSSetInitialSpace()
@*/
PetscErrorCode EPSSetUp(EPS eps)
{
  PetscErrorCode ierr;
  Mat            A,B; 
  PetscInt       i,k;
  PetscBool      flg,lindep;
  Vec            *newDS;
  PetscReal      norm;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    sigma;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (eps->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* Set default solver type (EPSSetFromOptions was not called) */
  if (!((PetscObject)eps)->type_name) {
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  }
  if (!eps->OP) { ierr = EPSGetST(eps,&eps->OP);CHKERRQ(ierr); }
  if (!((PetscObject)eps->OP)->type_name) {
      ierr = STSetType(eps->OP,STSHIFT);CHKERRQ(ierr);
  }
  if (!eps->ip) { ierr = EPSGetIP(eps,&eps->ip);CHKERRQ(ierr); }
  if (!((PetscObject)eps->ip)->type_name) {
    ierr = IPSetDefaultType_Private(eps->ip);CHKERRQ(ierr);
  }
  if (!((PetscObject)eps->rand)->type_name) {
    ierr = PetscRandomSetFromOptions(eps->rand);CHKERRQ(ierr);
  }
  
  /* Set problem dimensions */
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  if (!A) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_WRONGSTATE,"EPSSetOperators must be called first"); 
  ierr = MatGetSize(A,&eps->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&eps->nloc,PETSC_NULL);CHKERRQ(ierr);
  ierr = SlepcMatGetVecsTemplate(A,&eps->t,PETSC_NULL);CHKERRQ(ierr);

  /* Set default problem type */
  if (!eps->problem_type) {
    if (B==PETSC_NULL) {
      ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
    } else {
      ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
    }
  } else if (!B && eps->isgeneralized) {
    ierr = PetscInfo(eps,"Eigenproblem set as generalized but no matrix B was provided; reverting to a standard eigenproblem\n");CHKERRQ(ierr);
    eps->isgeneralized = PETSC_FALSE;
    eps->problem_type = eps->ishermitian? EPS_HEP: EPS_NHEP;
  } else if (B && !eps->isgeneralized) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_INCOMP,"Inconsistent EPS state"); 
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);
  if (eps->ishermitian && PetscImaginaryPart(sigma) != 0.0)
    SETERRQ(((PetscObject)eps)->comm,1,"Hermitian problems are not compatible with complex shifts");
#endif
  if (eps->ishermitian && eps->leftvecs)
    SETERRQ(((PetscObject)eps)->comm,1,"Requesting left eigenvectors not allowed in Hermitian problems");
  
  if (eps->ispositive) {
    ierr = STGetBilinearForm(eps->OP,&B);CHKERRQ(ierr);
    ierr = IPSetMatrix(eps->ip,B);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = IPSetMatrix(eps->ip,PETSC_NULL);CHKERRQ(ierr);
  }
  
  if (eps->nev > eps->n) eps->nev = eps->n;
  if (eps->ncv > eps->n) eps->ncv = eps->n;

  /* initialization of matrix norms */
  if (eps->nrma == PETSC_DETERMINE) {
    ierr = MatHasOperation(A,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (flg) { ierr = MatNorm(A,NORM_INFINITY,&eps->nrma);CHKERRQ(ierr); }
    else eps->nrma = 1.0;
  }
  if (eps->nrmb == PETSC_DETERMINE) {
    ierr = MatHasOperation(B,MATOP_NORM,&flg);CHKERRQ(ierr);
    if (flg) { ierr = MatNorm(B,NORM_INFINITY,&eps->nrmb);CHKERRQ(ierr); }
    else eps->nrmb = 1.0;
  }

  /* call specific solver setup */
  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);

  /* Build balancing matrix if required */
  if (!eps->balance) eps->balance = EPS_BALANCE_NONE;
  if (!eps->ishermitian && (eps->balance==EPS_BALANCE_ONESIDE || eps->balance==EPS_BALANCE_TWOSIDE)) {
    if (!eps->D) {
      ierr = VecDuplicate(eps->V[0],&eps->D);CHKERRQ(ierr);
    } else {
      ierr = VecSet(eps->D,1.0);CHKERRQ(ierr);
    }
    ierr = EPSBuildBalance_Krylov(eps);CHKERRQ(ierr);
    ierr = STSetBalanceMatrix(eps->OP,eps->D);CHKERRQ(ierr);
  }

  /* Setup ST */
  ierr = STSetUp(eps->OP);CHKERRQ(ierr); 
  
  ierr = PetscTypeCompare((PetscObject)eps->OP,STCAYLEY,&flg);CHKERRQ(ierr);
  if (flg && eps->problem_type == EPS_PGNHEP)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Cayley spectral transformation is not compatible with PGNHEP");

  ierr = PetscTypeCompare((PetscObject)eps->OP,STFOLD,&flg);CHKERRQ(ierr);
  if (flg && !eps->ishermitian)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Fold spectral transformation requires a Hermitian problem");

  if (eps->nds>0) {
    if (!eps->ds_ortho) {
      /* allocate memory and copy deflation basis vectors into DS */
      ierr = VecDuplicateVecs(eps->t,eps->nds,&newDS);CHKERRQ(ierr);
      for (i=0;i<eps->nds;i++) {
        ierr = VecCopy(eps->DS[i],newDS[i]);CHKERRQ(ierr);
        ierr = VecDestroy(&eps->DS[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(eps->DS);CHKERRQ(ierr);
      eps->DS = newDS;
      /* orthonormalize vectors in DS */
      k = 0;
      for (i=0;i<eps->nds;i++) {
        ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,k,PETSC_NULL,eps->DS,eps->DS[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
        if (norm==0.0 || lindep) {
          ierr = PetscInfo(eps,"Linearly dependent deflation vector found, removing...\n");CHKERRQ(ierr);
        } else {
          ierr = VecScale(eps->DS[k],1.0/norm);CHKERRQ(ierr);
          k++;
        }
      }
      for (i=k;i<eps->nds;i++) { ierr = VecDestroy(&eps->DS[i]);CHKERRQ(ierr); }
      eps->nds = k;
      eps->ds_ortho = PETSC_TRUE;
    }
  }
  ierr = STCheckNullSpace(eps->OP,eps->nds,eps->DS);CHKERRQ(ierr);

  /* process initial vectors */
  if (eps->nini<0) {
    eps->nini = -eps->nini;
    if (eps->nini>eps->ncv) SETERRQ(((PetscObject)eps)->comm,1,"The number of initial vectors is larger than ncv");
    k = 0;
    for (i=0;i<eps->nini;i++) {
      ierr = VecCopy(eps->IS[i],eps->V[k]);CHKERRQ(ierr);
      ierr = VecDestroy(&eps->IS[i]);CHKERRQ(ierr);
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,k,PETSC_NULL,eps->V,eps->V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
      if (norm==0.0 || lindep) {
        ierr = PetscInfo(eps,"Linearly dependent initial vector found, removing...\n");CHKERRQ(ierr);
      } else {
        ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
        k++;
      }
    }
    eps->nini = k;
    ierr = PetscFree(eps->IS);CHKERRQ(ierr);
  }
  if (eps->ninil<0) {
    if (!eps->leftvecs) {
      ierr = PetscInfo(eps,"Ignoring initial left vectors\n");CHKERRQ(ierr);
    } else {
      eps->ninil = -eps->ninil;
      if (eps->ninil>eps->ncv) SETERRQ(((PetscObject)eps)->comm,1,"The number of initial left vectors is larger than ncv");
      k = 0;
      for (i=0;i<eps->ninil;i++) {
        ierr = VecCopy(eps->ISL[i],eps->W[k]);CHKERRQ(ierr);
        ierr = VecDestroy(&eps->ISL[i]);CHKERRQ(ierr);
        ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,k,PETSC_NULL,eps->W,eps->W[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
        if (norm==0.0 || lindep) {
          ierr = PetscInfo(eps,"Linearly dependent initial left vector found, removing...\n");CHKERRQ(ierr);
        } else {
          ierr = VecScale(eps->W[k],1.0/norm);CHKERRQ(ierr);
          k++;
        }
      }
      eps->ninil = k;
      ierr = PetscFree(eps->ISL);CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
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

   It must be called after EPSSetUp(). If it is called again after EPSSetUp() then
   the EPS object is reset.

   Level: beginner

.seealso: EPSSolve(), EPSSetUp(), EPSReset(), EPSGetST(), STGetOperators()
@*/
PetscErrorCode EPSSetOperators(EPS eps,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n,m0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscCheckSameComm(eps,1,A,2);
  if (B) PetscCheckSameComm(eps,1,B,3);

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ(((PetscObject)eps)->comm,1,"A is a non-square matrix");
  if (B) { 
    ierr = MatGetSize(B,&m0,&n);CHKERRQ(ierr);
    if (m0!=n) SETERRQ(((PetscObject)eps)->comm,1,"B is a non-square matrix");
    if (m!=m0) SETERRQ(((PetscObject)eps)->comm,1,"Dimensions of A and B do not match");
  }

  if (eps->setupcalled) { ierr = EPSReset(eps);CHKERRQ(ierr); }
  if (!eps->OP) { ierr = EPSGetST(eps,&eps->OP);CHKERRQ(ierr); }
  ierr = STSetOperators(eps->OP,A,B);CHKERRQ(ierr);
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
PetscErrorCode EPSGetOperators(EPS eps,Mat *A,Mat *B)
{
  PetscErrorCode ierr;
  ST             st;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (A) PetscValidPointer(A,2);
  if (B) PetscValidPointer(B,3);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetOperators(st,A,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDeflationSpace"
/*@
   EPSSetDeflationSpace - Specify a basis of vectors that constitute
   the deflation space.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  ds    - set of basis vectors of the deflation space

   Notes:
   When a deflation space is given, the eigensolver seeks the eigensolution
   in the restriction of the problem to the orthogonal complement of this
   space. This can be used for instance in the case that an invariant 
   subspace is known beforehand (such as the nullspace of the matrix).

   Basis vectors set by a previous call to EPSSetDeflationSpace() are
   replaced.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   These vectors persist from one EPSSolve() call to the other, use
   EPSRemoveDeflationSpace() to eliminate them.

   Level: intermediate

.seealso: EPSRemoveDeflationSpace()
@*/
PetscErrorCode EPSSetDeflationSpace(EPS eps,PetscInt n,Vec *ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument n out of range"); 

  /* free previous vectors */
  ierr = EPSRemoveDeflationSpace(eps);CHKERRQ(ierr);

  /* get references of passed vectors */
  if (n>0) {
    ierr = PetscMalloc(n*sizeof(Vec),&eps->DS);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)ds[i]);CHKERRQ(ierr);
      eps->DS[i] = ds[i];
    }
    eps->setupcalled = 0;
    eps->ds_ortho = PETSC_FALSE;
  }

  eps->nds = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRemoveDeflationSpace"
/*@
   EPSRemoveDeflationSpace - Removes the deflation space.

   Collective on EPS

   Input Parameter:
.  eps   - the eigenproblem solver context

   Level: intermediate

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode EPSRemoveDeflationSpace(EPS eps)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = VecDestroyVecs(eps->nds,&eps->DS);CHKERRQ(ierr);
  eps->nds = 0;
  eps->setupcalled = 0;
  eps->ds_ortho = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialSpace"
/*@
   EPSSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   In contrast to EPSSetDeflationSpace(), these vectors do not persist from one
   EPSSolve() call to the other, so the initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: EPSSetInitialSpaceLeft(), EPSSetDeflationSpace()
@*/
PetscErrorCode EPSSetInitialSpace(EPS eps,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (eps->nini<0) {
    for (i=0;i<-eps->nini;i++) {
      ierr = VecDestroy(&eps->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(eps->IS);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  if (n>0) {
    ierr = PetscMalloc(n*sizeof(Vec),&eps->IS);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
      eps->IS[i] = is[i];
    }
    eps->setupcalled = 0;
  }

  eps->nini = -n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialSpaceLeft"
/*@
   EPSSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, the subspace from which the solver starts to iterate for
   building the left subspace (in methods that work with two subspaces).

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial left space

   Notes:
   Some solvers start to iterate on a single vector (initial left vector). In that case,
   the other vectors are ignored.

   In contrast to EPSSetDeflationSpace(), these vectors do not persist from one
   EPSSolve() call to the other, so the initial left space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted left eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: EPSSetInitialSpace(), EPSSetDeflationSpace()
@*/
PetscErrorCode EPSSetInitialSpaceLeft(EPS eps,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,n,2);
  if (n<0) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (eps->ninil<0) {
    for (i=0;i<-eps->ninil;i++) {
      ierr = VecDestroy(&eps->ISL[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(eps->ISL);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  if (n>0) {
    ierr = PetscMalloc(n*sizeof(Vec),&eps->ISL);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
      eps->ISL[i] = is[i];
    }
    eps->setupcalled = 0;
  }

  eps->ninil = -n;
  PetscFunctionReturn(0);
}

