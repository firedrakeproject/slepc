
/*                       
       This file implements a wrapper to the LAPACK eigenvalue subroutines.
       Currently, only LAPACK routines for standard problems are used.
       Generalized problems are transformed to standard ones.
*/
#include "src/eps/impls/lapack/lapackp.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LAPACK"
static int EPSSetUp_LAPACK(EPS eps)
{
  int         ierr,i,size,rank,n,m,row,nz,*cols,dummy;
  PetscScalar *vals;
  EPS_LAPACK *la = (EPS_LAPACK *)eps->data;
  MPI_Comm    comm = eps->comm;

  PetscFunctionBegin;
  ierr = EPSComputeExplicitOperator(eps,&la->BA);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr     = MatGetSize(la->BA,&n,&n);CHKERRQ(ierr);
  if (size > 1) { /* assemble matrix on first processor */
    if (!rank) {
      ierr = MatCreateMPIDense(comm,n,n,n,n,PETSC_NULL,&la->A);CHKERRQ(ierr);
    }
    else {
      ierr = MatCreateMPIDense(comm,0,n,n,n,PETSC_NULL,&la->A);CHKERRQ(ierr);
    }
    PetscLogObjectParent(la->BA,la->A);

    ierr = MatGetOwnershipRange(la->BA,&row,&dummy);CHKERRQ(ierr);
    ierr = MatGetLocalSize(la->BA,&m,&dummy);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = MatGetRow(la->BA,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(la->A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(la->BA,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    } 

    ierr = MatAssemblyBegin(la->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(la->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDefaults_LAPACK"
static int EPSSetDefaults_LAPACK(EPS eps)
{
  int         ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<1 || eps->ncv>N) SETERRQ(1,"Wrong value of ncv"); 
  }
  else eps->ncv = eps->nev;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LAPACK"
static int  EPSSolve_LAPACK(EPS eps,int *its)
{
  int         ierr,n,size,rank;
  PetscScalar *array,*pV;
  EPS_LAPACK *la = (EPS_LAPACK *)eps->data;
  MPI_Comm    comm = eps->comm;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size>1) {
    ierr = MatGetArray(la->A,&array);CHKERRQ(ierr);
  } else {
    ierr = MatGetArray(la->BA,&array);CHKERRQ(ierr);
  }
  ierr = MatGetSize(la->BA,&n,&n);CHKERRQ(ierr);

  if (eps->dropvectors) pV = PETSC_NULL;
  else {
    ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);
  }

  ierr = EPSDenseNHEPSorted(n,array,eps->eigr,eps->eigi,pV,eps->ncv,eps->which);CHKERRQ(ierr);

  if (!eps->dropvectors) {
    ierr = VecRestoreArray(eps->V[0],&pV);CHKERRQ(ierr);
  }

  if (size > 1) {
    ierr = MatRestoreArray(la->A,&array);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreArray(la->BA,&array);CHKERRQ(ierr);
  }

  eps->nconv = eps->ncv;
  eps->its   = 1;
  *its       = eps->its;
  
  eps->reason = EPS_CONVERGED_TOL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_LAPACK"
/*
  EPSDestroy_LAPACK - Destroys the context variable for LAPACK.

  Input Parameter: 
. eps - the iterative context
*/
int EPSDestroy_LAPACK(EPS eps)
{
  int         ierr,size;
  EPS_LAPACK *la = (EPS_LAPACK *)eps->data;
  MPI_Comm    comm = eps->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatDestroy(la->A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(la->BA);CHKERRQ(ierr);
  ierr = EPSDefaultDestroy(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LAPACK"
int EPSCreate_LAPACK(EPS eps)
{
  EPS_LAPACK *la;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LAPACK,&la);CHKERRQ(ierr);
  PetscMemzero(la,sizeof(EPS_LAPACK));
  PetscLogObjectMemory(eps,sizeof(EPS_LAPACK));
  eps->data                      = (void *) la;
  eps->ops->setup                = EPSSetUp_LAPACK;
  eps->ops->setdefaults          = EPSSetDefaults_LAPACK;
  eps->ops->solve                = EPSSolve_LAPACK;
  eps->ops->destroy              = EPSDestroy_LAPACK;
  eps->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
