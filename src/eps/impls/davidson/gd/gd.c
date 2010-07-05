/*
  SLEPc eigensolver: "davidson"

  Method: General Davidson Method

  References:
    - Ernest R. Davidson. Super-matrix methods. Computer Physics Communications,
      53:49–60, May 1989.
*/

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "private/stimpl.h"
#include "../src/eps/impls/davidson/common/davidson.h"
#include "slepcblaslapack.h"

PetscErrorCode EPSSetUp_GD(EPS eps);
PetscErrorCode EPSDestroy_GD(EPS eps);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_GD"
PetscErrorCode EPSSetFromOptions_GD(EPS eps)
{
  PetscErrorCode  ierr;
  PetscTruth      flg,op;
  PetscInt        opi,opi0;

  PetscFunctionBegin;
  
  ierr = PetscOptionsHead("GD options");CHKERRQ(ierr);

  ierr = EPSGDGetKrylovStart(eps, &op); CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-eps_gd_krylov_start","Start the searching subspace with a krylov basis","EPSGDSetKrylovStart",op,&op,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSGDSetKrylovStart(eps, op); CHKERRQ(ierr); }
 
  ierr = EPSGDGetBlockSize(eps, &opi); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_blocksize","Number vectors add to the searching subspace (if 0, nev employed)","EPSGDSetBlockSize",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSGDSetBlockSize(eps, opi); CHKERRQ(ierr); }

  ierr = EPSGDGetRestart(eps, &opi, &opi0); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_minv","Set the size of the searching subspace after restarting (if 0, eps_gd_bs is employed)","EPSGDSetRestart",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSGDSetRestart(eps, opi, opi0); CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_gd_plusk","Set the number of saved eigenvectors from the previous iteration when restarting","EPSGDSetRestart",opi0,&opi0,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSGDSetRestart(eps, opi, opi0); CHKERRQ(ierr); }

  ierr = EPSGDGetInitialSize(eps, &opi); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_initial_size","Set the initial size of the searching subspace","EPSGDSetInitialSize",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSGDSetInitialSize(eps, opi); CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}  
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_GD"
PetscErrorCode EPSSetUp_GD(EPS eps)
{
  PetscErrorCode  ierr;
  PetscTruth      t;
  KSP             ksp;

  PetscFunctionBegin;

  /* Check some constraints */ 
  ierr = STSetUp(eps->OP); CHKERRQ(ierr);
  ierr = STGetKSP(eps->OP, &ksp); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)ksp, KSPPREONLY, &t); CHKERRQ(ierr);
  if (t == PETSC_FALSE) SETERRQ(PETSC_ERR_SUP, "gd only works with preonly ksp of the spectral transformation");

  /* Setup common for all davidson solvers */
  ierr = EPSSetUp_DAVIDSON(eps);

  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_GD"
PetscErrorCode EPSCreate_GD(EPS eps) {
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Load the DAVIDSON solver */
  ierr = EPSCreate_DAVIDSON(eps); CHKERRQ(ierr);

  /* Overload the GD properties */
  eps->ops->setfromoptions       = EPSSetFromOptions_GD;
  eps->ops->setup                = EPSSetUp_GD;
  eps->ops->destroy              = EPSDestroy_GD;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetKrylovStart_C","EPSDAVIDSONSetKrylovStart_DAVIDSON",EPSDAVIDSONSetKrylovStart_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetKrylovStart_C","EPSDAVIDSONGetKrylovStart_DAVIDSON",EPSDAVIDSONGetKrylovStart_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBlockSize_C","EPSDAVIDSONSetBlockSize_DAVIDSON",EPSDAVIDSONSetBlockSize_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBlockSize_C","EPSDAVIDSONGetBlockSize_DAVIDSON",EPSDAVIDSONGetBlockSize_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetRestart_C","EPSDAVIDSONSetRestart_DAVIDSON",EPSDAVIDSONSetRestart_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetRestart_C","EPSDAVIDSONGetRestart_DAVIDSON",EPSDAVIDSONGetRestart_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetInitialSize_C","EPSDAVIDSONSetInitialSize_DAVIDSON",EPSDAVIDSONSetInitialSize_DAVIDSON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetInitialSize_C","EPSDAVIDSONGetInitialSize_DAVIDSON",EPSDAVIDSONGetInitialSize_DAVIDSON);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_GD"
PetscErrorCode EPSDestroy_GD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetKrylovStart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetKrylovStart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetRestart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetRestart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetInitialSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetInitialSize_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = EPSDestroy_DAVIDSON(eps);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetKrylovStart"
/*@
   EPSGDSetKrylovStart - Activates or deactivates starting the searching
   subspace with a Krylov basis. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  krylovstart - boolean flag

   Options Database Key:
.  -eps_gd_krylovstart - Activates starting the searching subspace with a
    Krylov basis
   
   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDSetKrylovStart(EPS eps,PetscTruth krylovstart)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDSetKrylovStart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,krylovstart);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetKrylovStart"
/*@
   EPSGDGetKrylovStart - Gets if the searching subspace is started with a
   Krylov basis.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  krylovstart - boolean flag indicating if starting the searching subspace
   with a Krylov basis is enabled.

   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDGetKrylovStart(EPS eps,PetscTruth *krylovstart)
{
  PetscErrorCode ierr, (*f)(EPS,PetscTruth*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDGetKrylovStart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,krylovstart);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetBlockSize"
/*@
   EPSGDSetBlockSize - Sets the number of vectors added to the searching space
   every iteration.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  blocksize - non-zero positive integer

   Options Database Key:
.  -eps_gd_blocksize - integer indicating the number of vectors added to the
   searching space every iteration. 
   
   Level: advanced

.seealso: EPSGDSetKrylovStart()
@*/
PetscErrorCode EPSGDSetBlockSize(EPS eps,PetscInt blocksize)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDSetBlockSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,blocksize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetBlockSize"
/*@
   EPSGDGetBlockSize - Gets the number of vectors added to the searching space
   every iteration.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  blocksize - integer indicating the number of vectors added to the searching
   space every iteration.

   Level: advanced

.seealso: EPSGDSetBlockSize()
@*/
PetscErrorCode EPSGDGetBlockSize(EPS eps,PetscInt *blocksize)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDGetBlockSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,blocksize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetRestart"
/*@
   EPSGDGetRestart - Gets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  minv - non-zero positive integer indicating the number of vectors of the
   searching subspace after restarting
-  plusk - positive integer indicating the number of vectors saved from the
   previous iteration   

   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt*,PetscInt*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDGetRestart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,minv,plusk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetRestart"
/*@
   EPSGDSetRestart - Sets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  minv - non-zero positive integer indicating the number of vectors of the
   searching subspace after restarting
-  plusk - positive integer indicating the number of vectors saved from the
   previous iteration   

   Options Database Key:
+  -eps_gd_minv - non-zero positive integer indicating the number of vectors
    of the searching subspace after restarting
-  -eps_gd_plusk - positive integer indicating the number of vectors saved
    from the previous iteration   
   
   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDSetRestart(EPS eps,PetscInt minv,PetscInt plusk)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDSetRestart_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,minv,plusk);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetInitialSize"
/*@
   EPSGDGetInitialSize - Gets the initial size of the searching space. In the 
   case of EPSGetKrylovStart is PETSC_FALSE and the user provides vectors by
   EPSSetInitialSpace, up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGetKrylovStart is PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  initialsize - non-zero positive integer indicating the number of vectors of
   the initial searching subspace

   Level: advanced

.seealso: EPSGDSetInitialSize(), EPSGetKrylovStart()
@*/
PetscErrorCode EPSGDGetInitialSize(EPS eps,PetscInt *initialsize)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDGetInitialSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,initialsize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetInitialSize"
/*@
   EPSGDSetInitialSize - Sets the initial size of the searching space. In the 
   case of EPSGetKrylovStart is PETSC_FALSE and the user provides vectors by
   EPSSetInitialSpace, up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGetKrylovStart is PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  initialsize - non-zero positive integer indicating the number of vectors of
   the initial searching subspace

   Options Database Key:
.  -eps_gd_initial_size - non-zero positive integer indicating the number of
    vectors of the initial searching subspace
   
   Level: advanced

.seealso: EPSGDGetInitialSize(), EPSGetKrylovStart()
@*/
PetscErrorCode EPSGDSetInitialSize(EPS eps,PetscInt initialsize)
{
  PetscErrorCode ierr, (*f)(EPS,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSGDSetInitialSize_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,initialsize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

