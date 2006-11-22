/*                       

   SLEPc singular value solver: "eigen"

   Method: Uses an Hermitian eigensolver for A^T*A, A*A^T or H(A)

   Last update: Nov 2006

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepceps.h"

typedef struct {
  EPS eps;
  Mat mat;
  Vec w;
} SVD_EIGENSOLVER;

#undef __FUNCT__  
#define __FUNCT__ "ShellMatMult_EIGENSOLVER"
PetscErrorCode ShellMatMult_EIGENSOLVER(Mat B,Vec x, Vec y)
{
  PetscErrorCode  ierr;
  SVD             svd;
  SVD_EIGENSOLVER *eigen;

  PetscFunctionBegin;
  ierr = MatShellGetContext(B,(void**)&svd);CHKERRQ(ierr);
  eigen = (SVD_EIGENSOLVER *)svd->data;
  ierr = MatMult(svd->A,x,eigen->w);CHKERRQ(ierr);
  ierr = MatMultTranspose(svd->A,eigen->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_EIGENSOLVER"
PetscErrorCode SVDSetUp_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  PetscInt        m,n,M,N;
  Vec             x;

  PetscFunctionBegin;
  
  if (eigen->w) { ierr = VecDestroy(eigen->w);CHKERRQ(ierr); } 
  ierr = MatGetVecs(svd->A,&x,&eigen->w);CHKERRQ(ierr);
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);

  if (eigen->mat) { ierr = MatDestroy(eigen->mat);CHKERRQ(ierr); }
  ierr = MatCreateShell(svd->comm,n,n,N,N,svd,&eigen->mat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(eigen->mat,MATOP_MULT,(void(*)(void))ShellMatMult_EIGENSOLVER);CHKERRQ(ierr);  

  ierr = EPSSetOperators(eigen->eps,eigen->mat,PETSC_NULL);CHKERRQ(ierr);
  /* EPSSetProblemType deberia estar en SVDSetFromOptions
     PENDIENTE: ARREGLAR EL PROBLEMA EN EPS */
  ierr = EPSSetProblemType(eigen->eps,EPS_NHEP);CHKERRQ(ierr);
  ierr = EPSSetUp(eigen->eps);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_EIGENSOLVER"
PetscErrorCode SVDSolve_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;
  int             i;
  PetscScalar     sigma;
  
  PetscFunctionBegin;
  ierr = EPSSolve(eigen->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eigen->eps,&svd->nconv);CHKERRQ(ierr);

  if (svd->sigma) { ierr = PetscFree(svd->sigma);CHKERRQ(ierr); }
  if (svd->U) {
    for (i=0;i<svd->nconv;i++) {
      ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  if (svd->V) {
    for (i=0;i<svd->nconv;i++) {
      ierr = VecDestroy(svd->V[i]);CHKERRQ(ierr); 
    }
    ierr = PetscFree(svd->V);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc(svd->nconv*sizeof(PetscReal),&svd->sigma);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->nconv*sizeof(Vec),&svd->U);CHKERRQ(ierr);
  ierr = PetscMalloc(svd->nconv*sizeof(Vec),&svd->V);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    ierr = MatGetVecs(svd->A,svd->V+i,svd->U+i);CHKERRQ(ierr);
    ierr = EPSGetEigenpair(eigen->eps,i,&sigma,PETSC_NULL,svd->V[i],PETSC_NULL);CHKERRQ(ierr);
    svd->sigma[i] = sqrt(PetscRealPart(sigma));
    ierr = MatMult(svd->A,svd->V[i],svd->U[i]);CHKERRQ(ierr);
    ierr = VecScale(svd->U[i],1.0/svd->sigma[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_EIGENSOLVER"
PetscErrorCode SVDSetFromOptions_EIGENSOLVER(SVD svd)
{
  PetscErrorCode ierr;
  SVD_EIGENSOLVER      *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EIGEN options");CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eigen->eps);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigenSetEPS_EIGENSOLVER"
PetscErrorCode SVDEigenSetEPS_EIGENSOLVER(SVD svd,EPS eps)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,2);
  PetscCheckSameComm(svd,1,eps,2);
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(eigen->eps);CHKERRQ(ierr);  
  eigen->eps = eps;
  svd->setupcalled = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDEigenSetEPS"
/*@
   SVDEigenSetEPS - Associates a eigensolver object to the
   singular value solver. 

   Collective on SVD

   Input Parameters:
+  svd - singular value solver context obtained from SVDCreate()
-  eps - the eigensolver object

   Level: advanced

.seealso: SVDEigenGetEPS()
@*/
PetscErrorCode SVDEigenSetEPS(SVD svd,EPS eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigenSetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDEigenGetEPS_EIGENSOLVER"
PetscErrorCode SVDEigenGetEPS_EIGENSOLVER(SVD svd,EPS *eps)
{
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  PetscValidPointer(eps,2);
  *eps = eigen->eps;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDEigenGetEPS"
/*@
   SVDEigenGetEPS - Obtain the eigensolver (EPS) object associated
   to the singular value solver object.

   Not Collective

   Input Parameters:
.  svd - singular value solver context obtained from SVDCreate()

   Output Parameter:
.  eps - the eigensolver object

   Level: beginner
@*/
PetscErrorCode SVDEigenGetEPS(SVD svd,EPS *eps)
{
  PetscErrorCode ierr, (*f)(SVD,EPS *eps);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDEigenGetEPS_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_EIGENSOLVER"
PetscErrorCode SVDView_EIGENSOLVER(SVD svd,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  ierr = EPSView(eigen->eps,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_EIGENSOLVER"
PetscErrorCode SVDDestroy_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen = (SVD_EIGENSOLVER *)svd->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(eigen->eps);CHKERRQ(ierr);
  if (eigen->mat) { ierr = MatDestroy(eigen->mat);CHKERRQ(ierr); }
  if (eigen->w) { ierr = VecDestroy(eigen->w);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_EIGENSOLVER"
PetscErrorCode SVDCreate_EIGENSOLVER(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_EIGENSOLVER *eigen;
  
  PetscFunctionBegin;
  ierr = PetscNew(SVD_EIGENSOLVER,&eigen);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_EIGENSOLVER));
  svd->data                      = (void *)eigen;
  svd->ops->solve                = SVDSolve_EIGENSOLVER;
  svd->ops->setup                = SVDSetUp_EIGENSOLVER;
  svd->ops->setfromoptions       = SVDSetFromOptions_EIGENSOLVER;
  svd->ops->destroy              = SVDDestroy_EIGENSOLVER;
  svd->ops->view                 = SVDView_EIGENSOLVER;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigenSetEPS_C","SVDEigenSetEPS_EIGENSOLVER",SVDEigenSetEPS_EIGENSOLVER);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDEigenGetEPS_C","SVDEigenGetEPS_EIGENSOLVER",SVDEigenGetEPS_EIGENSOLVER);CHKERRQ(ierr);

  ierr = EPSCreate(svd->comm,&eigen->eps);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eigen->eps,svd->prefix);CHKERRQ(ierr);
  ierr = EPSAppendOptionsPrefix(eigen->eps,"svd_");CHKERRQ(ierr);
  PetscLogObjectParent(svd,eigen->eps);
  eigen->mat = PETSC_NULL;
  eigen->w = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

