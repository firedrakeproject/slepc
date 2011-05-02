/*
  SLEPc eigensolver: "jd"

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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <private/stimpl.h>                 /*I "slepcst.h" I*/
#include <../src/eps/impls/davidson/common/davidson.h>
#include <slepcblaslapack.h>

PetscErrorCode EPSSetUp_JD(EPS eps);
PetscErrorCode EPSDestroy_JD(EPS eps);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_JD"
PetscErrorCode EPSSetFromOptions_JD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg,op;
  PetscInt       opi,opi0;
  PetscReal      opf;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)eps)->comm,((PetscObject)eps)->prefix,"JD Options","EPS");CHKERRQ(ierr);

  ierr = EPSJDGetKrylovStart(eps, &op); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_jd_krylov_start","Start the searching subspace with a krylov basis","EPSJDSetKrylovStart",op,&op,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetKrylovStart(eps, op); CHKERRQ(ierr); }
 
  ierr = EPSJDGetBlockSize(eps, &opi); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_blocksize","Number vectors add to the searching subspace","EPSJDSetBlockSize",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetBlockSize(eps, opi); CHKERRQ(ierr); }

  ierr = EPSJDGetRestart(eps, &opi, &opi0); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_minv","Set the size of the searching subspace after restarting","EPSJDSetRestart",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetRestart(eps, opi, opi0); CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_jd_plusk","Set the number of saved eigenvectors from the previous iteration when restarting","EPSJDSetRestart",opi0,&opi0,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetRestart(eps, opi, opi0); CHKERRQ(ierr); }

  ierr = EPSJDGetInitialSize(eps, &opi); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_initial_size","Set the initial size of the searching subspace","EPSJDSetInitialSize",opi,&opi,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetInitialSize(eps, opi); CHKERRQ(ierr); }

  ierr = EPSJDGetFix(eps, &opf); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_jd_fix","Set the tolerance for changing the target in the correction equation","EPSJDSetFix",opf,&opf,&flg); CHKERRQ(ierr);
  if(flg) { ierr = EPSJDSetFix(eps, opf); CHKERRQ(ierr); }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_JD"
PetscErrorCode EPSSetUp_JD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      t;
  KSP            ksp;

  PetscFunctionBegin;
  /* Setup common for all davidson solvers */
  ierr = EPSSetUp_Davidson(eps); CHKERRQ(ierr);

  /* Check some constraints */ 
  ierr = STSetUp(eps->OP); CHKERRQ(ierr);
  ierr = STGetKSP(eps->OP, &ksp); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)ksp, KSPPREONLY, &t); CHKERRQ(ierr);
  if (t) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP, "EPSJD does not work with KSPPREONLY");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_JD"
PetscErrorCode EPSCreate_JD(EPS eps)
{
  PetscErrorCode ierr;
  KSP            ksp;

  PetscFunctionBegin;
  /* Load the Davidson solver */
  ierr = EPSCreate_Davidson(eps); CHKERRQ(ierr);

  /* Set the default ksp of the st to gmres */
  ierr = STGetKSP(eps->OP, &ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-3, 1e-10, PETSC_DEFAULT, 90); CHKERRQ(ierr);

  /* Overload the JD properties */
  eps->ops->setfromoptions       = EPSSetFromOptions_JD;
  eps->ops->setup                = EPSSetUp_JD;
  eps->ops->destroy              = EPSDestroy_JD;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetKrylovStart_C","EPSDavidsonSetKrylovStart_Davidson",EPSDavidsonSetKrylovStart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetKrylovStart_C","EPSDavidsonGetKrylovStart_Davidson",EPSDavidsonGetKrylovStart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetBlockSize_C","EPSDavidsonSetBlockSize_Davidson",EPSDavidsonSetBlockSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetBlockSize_C","EPSDavidsonGetBlockSize_Davidson",EPSDavidsonGetBlockSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetRestart_C","EPSDavidsonSetRestart_Davidson",EPSDavidsonSetRestart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetRestart_C","EPSDavidsonGetRestart_Davidson",EPSDavidsonGetRestart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetInitialSize_C","EPSDavidsonSetInitialSize_Davidson",EPSDavidsonSetInitialSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetInitialSize_C","EPSDavidsonGetInitialSize_Davidson",EPSDavidsonGetInitialSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetFix_C","EPSDavidsonSetFix_Davidson",EPSDavidsonSetFix_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetFix_C","EPSDavidsonGetFix_Davidson",EPSDavidsonGetFix_Davidson);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_JD"
PetscErrorCode EPSDestroy_JD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetKrylovStart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetKrylovStart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetBlockSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetRestart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetRestart_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetInitialSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetInitialSize_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDSetFix_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSJDGetFix_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSDestroy_Davidson(eps);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDSetKrylovStart"
/*@
   EPSJDSetKrylovStart - Activates or deactivates starting the searching
   subspace with a Krylov basis. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  krylovstart - boolean flag

   Options Database Key:
.  -eps_jd_krylov_start - Activates starting the searching subspace with a
    Krylov basis
   
   Level: advanced

.seealso: EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDSetKrylovStart(EPS eps,PetscBool krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSJDSetKrylovStart_C",(EPS,PetscBool),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDGetKrylovStart"
/*@
   EPSJDGetKrylovStart - Gets if the searching subspace is started with a
   Krylov basis.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  krylovstart - boolean flag indicating if starting the searching subspace
   with a Krylov basis is enabled.

   Level: advanced

.seealso: EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDGetKrylovStart(EPS eps,PetscBool *krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSJDGetKrylovStart_C",(EPS,PetscBool*),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDSetBlockSize"
/*@
   EPSJDSetBlockSize - Sets the number of vectors added to the searching space
   every iteration.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  blocksize - non-zero positive integer

   Options Database Key:
.  -eps_jd_blocksize - integer indicating the number of vectors added to the
   searching space every iteration. 
   
   Level: advanced

.seealso: EPSJDSetKrylovStart()
@*/
PetscErrorCode EPSJDSetBlockSize(EPS eps,PetscInt blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,blocksize,2);
  ierr = PetscTryMethod(eps,"EPSJDSetBlockSize_C",(EPS,PetscInt),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDGetBlockSize"
/*@
   EPSJDGetBlockSize - Gets the number of vectors added to the searching space
   every iteration.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  blocksize - integer indicating the number of vectors added to the searching
   space every iteration.

   Level: advanced

.seealso: EPSJDSetBlockSize()
@*/
PetscErrorCode EPSJDGetBlockSize(EPS eps,PetscInt *blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(blocksize,2);
  ierr = PetscTryMethod(eps,"EPSJDGetBlockSize_C",(EPS,PetscInt*),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDGetRestart"
/*@
   EPSJDGetRestart - Gets the number of vectors of the searching space after
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

.seealso: EPSJDSetRestart()
@*/
PetscErrorCode EPSJDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(minv,2);
  PetscValidIntPointer(plusk,3);
  ierr = PetscTryMethod(eps,"EPSJDGetRestart_C",(EPS,PetscInt*,PetscInt*),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDSetRestart"
/*@
   EPSJDSetRestart - Sets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  minv - non-zero positive integer indicating the number of vectors of the
   searching subspace after restarting
-  plusk - positive integer indicating the number of vectors saved from the
   previous iteration   

   Options Database Key:
+  -eps_jd_minv - non-zero positive integer indicating the number of vectors
    of the searching subspace after restarting
-  -eps_jd_plusk - positive integer indicating the number of vectors saved
    from the previous iteration   
   
   Level: advanced

.seealso: EPSJDGetRestart()
@*/
PetscErrorCode EPSJDSetRestart(EPS eps,PetscInt minv,PetscInt plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,minv,2);
  PetscValidLogicalCollectiveInt(eps,plusk,3);
  ierr = PetscTryMethod(eps,"EPSJDSetRestart_C",(EPS,PetscInt,PetscInt),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDGetInitialSize"
/*@
   EPSJDGetInitialSize - Gets the initial size of the searching space. In the 
   case of EPSJDGetKrylovStart is PETSC_FALSE and the user provides vectors by
   EPSSetInitialSpace, up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSJDGetKrylovStart is PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  initialsize - non-zero positive integer indicating the number of vectors of
   the initial searching subspace

   Level: advanced

.seealso: EPSJDSetInitialSize(), EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDGetInitialSize(EPS eps,PetscInt *initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(initialsize,2);
  ierr = PetscTryMethod(eps,"EPSJDGetInitialSize_C",(EPS,PetscInt*),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDSetInitialSize"
/*@
   EPSJDSetInitialSize - Sets the initial size of the searching space. In the 
   case of EPSJDGetKrylovStart is PETSC_FALSE and the user provides vectors by
   EPSSetInitialSpace, up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSJDGetKrylovStart is PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  initialsize - non-zero positive integer indicating the number of vectors of
   the initial searching subspace

   Options Database Key:
.  -eps_jd_initial_size - non-zero positive integer indicating the number of
    vectors of the initial searching subspace
   
   Level: advanced

.seealso: EPSJDGetInitialSize(), EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDSetInitialSize(EPS eps,PetscInt initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,initialsize,2);
  ierr = PetscTryMethod(eps,"EPSJDSetInitialSize_C",(EPS,PetscInt),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDGetFix"
/*@
   EPSJDGetFix - Gets the threshold for changing the target in the correction
   equation. The target in the correction equation is fixed at the first
   iterations. When the norm of the residual vector is lower than this value
   the target is set to the corresponding eigenvalue.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  fix - positive float number

   Level: advanced

.seealso: EPSJDSetFix()
@*/
PetscErrorCode EPSJDGetFix(EPS eps,PetscReal *fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(fix,2);
  ierr = PetscTryMethod(eps,"EPSJDGetFix_C",(EPS,PetscReal*),(eps,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSJDSetFix"
/*@
   EPSJDSetFix - Sets the threshold for changing the target in the correction
   equation. The target in the correction equation is fixed at the first
   iterations. When the norm of the residual vector is lower than this value
   the target is set to the corresponding eigenvalue.

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  fix - positive float number

   Options Database Key:
.  -eps_jd_fix
   
   Level: advanced

.seealso: EPSJDGetFix()
@*/
PetscErrorCode EPSJDSetFix(EPS eps,PetscReal fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,fix,2);
  ierr = PetscTryMethod(eps,"EPSJDSetFix_C",(EPS,PetscReal),(eps,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
