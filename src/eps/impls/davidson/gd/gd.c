/*
  SLEPc eigensolver: "gd"

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <../src/eps/impls/davidson/common/davidson.h>

PetscErrorCode EPSSetUp_GD(EPS eps);
PetscErrorCode EPSDestroy_GD(EPS eps);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_GD"
PetscErrorCode EPSSetFromOptions_GD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg,op;
  PetscInt       opi,opi0;
  KSP            ksp;
  EPSOrthType    orth;
  const char     *orth_list[3] = {"I","B","B_opt"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS Generalized Davidson (GD) Options");CHKERRQ(ierr);

  ierr = EPSGDGetKrylovStart(eps,&op);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_gd_krylov_start","Start the searching subspace with a krylov basis","EPSGDSetKrylovStart",op,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetKrylovStart(eps,op);CHKERRQ(ierr); }

  ierr = EPSGDGetBOrth(eps,&orth);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_gd_borth","orthogonalization used in the search subspace","EPSGDSetBOrth",orth_list,3,orth_list[orth-1],&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetBOrth(eps,(EPSOrthType)(opi+1));CHKERRQ(ierr); }
 
  ierr = EPSGDGetBlockSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_blocksize","Number vectors add to the searching subspace","EPSGDSetBlockSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetBlockSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSGDGetRestart(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_minv","Set the size of the searching subspace after restarting","EPSGDSetRestart",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_gd_plusk","Set the number of saved eigenvectors from the previous iteration when restarting","EPSGDSetRestart",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = EPSGDGetInitialSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_initial_size","Set the initial size of the searching subspace","EPSGDSetInitialSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetInitialSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSGDGetWindowSizes(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_pwindow","(Experimental!) Set the number of converged vectors in the projector","EPSGDSetWindowSizes",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_gd_qwindow","(Experimental!) Set the number of converged vectors in the projected problem","EPSGDSetWindowSizes",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsBool("-eps_gd_double_expansion","use the doble-expansion variant of GD","EPSGDSetDoubleExpansion",PETSC_FALSE,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetDoubleExpansion(eps,op);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_FALSE);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}  
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_GD"
PetscErrorCode EPSSetUp_GD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      t;
  KSP            ksp;

  PetscFunctionBegin;
  /* Set KSPPREONLY as default */ 
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }

  /* Setup common for all davidson solvers */
  ierr = EPSSetUp_Davidson(eps);CHKERRQ(ierr);

  /* Check some constraints */ 
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&t);CHKERRQ(ierr);
  if (!t) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"EPSGD only works with KSPPREONLY");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_GD"
PetscErrorCode EPSCreate_GD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Load the Davidson solver */
  ierr = EPSCreate_Davidson(eps);CHKERRQ(ierr);
  ierr = EPSDavidsonSetFix_Davidson(eps,0.0);CHKERRQ(ierr);
  ierr = EPSDavidsonSetMethod_Davidson(eps,DVD_METH_GD);CHKERRQ(ierr);

  /* Overload the GD properties */
  eps->ops->setfromoptions       = EPSSetFromOptions_GD;
  eps->ops->setup                = EPSSetUp_GD;
  eps->ops->destroy              = EPSDestroy_GD;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetKrylovStart_C","EPSDavidsonSetKrylovStart_Davidson",EPSDavidsonSetKrylovStart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetKrylovStart_C","EPSDavidsonGetKrylovStart_Davidson",EPSDavidsonGetKrylovStart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBOrth_C","EPSDavidsonSetBOrth_Davidson",EPSDavidsonSetBOrth_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBOrth_C","EPSDavidsonGetBOrth_Davidson",EPSDavidsonGetBOrth_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBlockSize_C","EPSDavidsonSetBlockSize_Davidson",EPSDavidsonSetBlockSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBlockSize_C","EPSDavidsonGetBlockSize_Davidson",EPSDavidsonGetBlockSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetRestart_C","EPSDavidsonSetRestart_Davidson",EPSDavidsonSetRestart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetRestart_C","EPSDavidsonGetRestart_Davidson",EPSDavidsonGetRestart_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetInitialSize_C","EPSDavidsonSetInitialSize_Davidson",EPSDavidsonSetInitialSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetInitialSize_C","EPSDavidsonGetInitialSize_Davidson",EPSDavidsonGetInitialSize_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetWindowSizes_C","EPSDavidsonSetWindowSizes_Davidson",EPSDavidsonSetWindowSizes_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetWindowSizes_C","EPSDavidsonGetWindowSizes_Davidson",EPSDavidsonGetWindowSizes_Davidson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetDoubleExpansion_C","EPSGDSetDoubleExpansion_GD",EPSGDSetDoubleExpansion_GD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetDoubleExpansion_C","EPSGDGetDoubleExpansion_GD",EPSGDGetDoubleExpansion_GD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_GD"
PetscErrorCode EPSDestroy_GD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetKrylovStart_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetKrylovStart_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBOrth_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBOrth_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetBlockSize_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetBlockSize_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetRestart_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetRestart_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetInitialSize_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetInitialSize_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetWindowSizes_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetWindowSizes_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDSetDoubleExpansion_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSGDGetDoubleExpansion_C","",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetKrylovStart"
/*@
   EPSGDSetKrylovStart - Activates or deactivates starting the searching
   subspace with a Krylov basis. 

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  krylovstart - boolean flag

   Options Database Key:
.  -eps_gd_krylov_start - Activates starting the searching subspace with a
    Krylov basis
   
   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDSetKrylovStart(EPS eps,PetscBool krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSGDSetKrylovStart_C",(EPS,PetscBool),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetKrylovStart"
/*@
   EPSGDGetKrylovStart - Returns a flag indicating if the search subspace is started with a
   Krylov basis.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  krylovstart - boolean flag indicating if the search subspace is started
   with a Krylov basis

   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDGetKrylovStart(EPS eps,PetscBool *krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSGDGetKrylovStart_C",(EPS,PetscBool*),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetBlockSize"
/*@
   EPSGDSetBlockSize - Sets the number of vectors to be added to the searching space
   in every iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  blocksize - number of vectors added to the search space in every iteration

   Options Database Key:
.  -eps_gd_blocksize - number of vectors added to the search space in every iteration
   
   Level: advanced

.seealso: EPSGDSetKrylovStart()
@*/
PetscErrorCode EPSGDSetBlockSize(EPS eps,PetscInt blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,blocksize,2);
  ierr = PetscTryMethod(eps,"EPSGDSetBlockSize_C",(EPS,PetscInt),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetBlockSize"
/*@
   EPSGDGetBlockSize - Returns the number of vectors to be added to the searching space
   in every iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  blocksize - number of vectors added to the search space in every iteration

   Level: advanced

.seealso: EPSGDSetBlockSize()
@*/
PetscErrorCode EPSGDGetBlockSize(EPS eps,PetscInt *blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(blocksize,2);
  ierr = PetscTryMethod(eps,"EPSGDGetBlockSize_C",(EPS,PetscInt*),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetRestart"
/*@
   EPSGDGetRestart - Gets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration   

   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSGDGetRestart_C",(EPS,PetscInt*,PetscInt*),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetRestart"
/*@
   EPSGDSetRestart - Sets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration   

   Options Database Keys:
+  -eps_gd_minv - number of vectors of the searching subspace after restarting
-  -eps_gd_plusk - number of vectors saved from the previous iteration   
   
   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDSetRestart(EPS eps,PetscInt minv,PetscInt plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,minv,2);
  PetscValidLogicalCollectiveInt(eps,plusk,2);
  ierr = PetscTryMethod(eps,"EPSGDSetRestart_C",(EPS,PetscInt,PetscInt),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetInitialSize"
/*@
   EPSGDGetInitialSize - Returns the initial size of the searching space.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  initialsize - number of vectors of the initial searching subspace

   Notes:
   If EPSGDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSGDSetInitialSize(), EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDGetInitialSize(EPS eps,PetscInt *initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(initialsize,2);
  ierr = PetscTryMethod(eps,"EPSGDGetInitialSize_C",(EPS,PetscInt*),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetInitialSize"
/*@
   EPSGDSetInitialSize - Sets the initial size of the searching space.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  initialsize - number of vectors of the initial searching subspace

   Options Database Key:
.  -eps_gd_initial_size - number of vectors of the initial searching subspace
   
   Notes:
   If EPSGDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSGDGetInitialSize(), EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDSetInitialSize(EPS eps,PetscInt initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,initialsize,2);
  ierr = PetscTryMethod(eps,"EPSGDSetInitialSize_C",(EPS,PetscInt),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetBOrth"
/*@
   EPSGDSetBOrth - Selects the orthogonalization that will be used in the search
   subspace in case of generalized Hermitian problems.

   Logically Collective on EPS

   Input Parameters:
+  eps   - the eigenproblem solver context
-  borth - the kind of orthogonalization

   Possible values:
   The parameter 'borth' can have one of these values

+   EPS_ORTH_I - orthogonalization of the search subspace
.   EPS_ORTH_B - B-orthogonalization of the search subspace
-   EPS_ORTH_BOPT - B-orthogonalization of the search subspace with an alternative method

   Options Database Key:
.  -eps_gd_borth - Set the orthogonalization used in the search subspace

   Notes:
   If borth is EPS_ORTH_B, the solver uses a variant of Gram-Schmidt (selected in
   IP associated to the EPS) with the inner product defined by the matrix problem B.
   If borth is EPS_ORTH_BOPT, it uses another variant of Gram-Schmidt that only performs
   one matrix-vector product although more than one reorthogonalization would be done.
   
   Level: advanced

.seealso: EPSGDGetBOrth()
@*/
PetscErrorCode EPSGDSetBOrth(EPS eps,EPSOrthType borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,borth,2);
  ierr = PetscTryMethod(eps,"EPSGDSetBOrth_C",(EPS,EPSOrthType),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetBOrth"
/*@
   EPSGDGetBOrth - Returns the orthogonalization used in the search
   subspace in case of generalized Hermitian problems.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  borth - the kind of orthogonalization

   Notes:
   See EPSGDSetBOrth() for possible values of 'borth'.

   Level: advanced

.seealso: EPSGDSetBOrth(), EPSOrthType
@*/
PetscErrorCode EPSGDGetBOrth(EPS eps,EPSOrthType *borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(borth,2);
  ierr = PetscTryMethod(eps,"EPSGDGetBOrth_C",(EPS,EPSOrthType*),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetWindowSizes"
/*@
   EPSGDGetWindowSizes - Gets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Level: advanced

.seealso: EPSGDSetWindowSizes()
@*/
PetscErrorCode EPSGDGetWindowSizes(EPS eps,PetscInt *pwindow,PetscInt *qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscTryMethod(eps,"EPSGDGetWindowSizes_C",(EPS,PetscInt*,PetscInt*),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetWindowSizes"
/*@
   EPSGDSetWindowSizes - Sets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Options Database Keys:
+  -eps_gd_pwindow - set the number of converged vectors in the projector
-  -eps_gd_qwindow - set the number of converged vectors in the projected problem  
   
   Level: advanced

.seealso: EPSGDGetWindowSizes()
@*/
PetscErrorCode EPSGDSetWindowSizes(EPS eps,PetscInt pwindow,PetscInt qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,pwindow,2);
  PetscValidLogicalCollectiveInt(eps,qwindow,3);
  ierr = PetscTryMethod(eps,"EPSGDSetWindowSizes_C",(EPS,PetscInt,PetscInt),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetDoubleExpansion_GD"
PetscErrorCode EPSGDSetDoubleExpansion_GD(EPS eps,PetscBool use_gd2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = EPSDavidsonSetMethod_Davidson(eps,use_gd2?DVD_METH_GD2:DVD_METH_GD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetDoubleExpansion_GD"
PetscErrorCode EPSGDGetDoubleExpansion_GD(EPS eps,PetscBool *flg)
{
  Method_t       meth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = EPSDavidsonGetMethod_Davidson(eps,&meth);CHKERRQ(ierr);
  if (meth==DVD_METH_GD2) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDGetDoubleExpansion"
/*@
   EPSGDGetDoubleExpansion - Gets a flag indicating whether the double 
   expansion variant has been activated or not.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  flg - the flag

   Level: advanced

.seealso: EPSGDSetDoubleExpansion()
@*/
PetscErrorCode EPSGDGetDoubleExpansion(EPS eps,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(flg,2);
  ierr = PetscTryMethod(eps,"EPSGDGetDoubleExpansion_C",(EPS,PetscBool*),(eps,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGDSetDoubleExpansion"
/*@
   EPSGDSetDoubleExpansion - Activate a variant where the search subspace is
   expanded with K*[A*x B*x] (double expansion) instead of the classic K*r,
   where K is the preconditioner, x the selected approximate eigenvector and
   r its associated residual vector.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  use_gd2 - the boolean flag

   Options Database Keys:
.  -eps_gd_double_expansion - activate the double-expansion variant of GD
   
   Level: advanced
@*/
PetscErrorCode EPSGDSetDoubleExpansion(EPS eps,PetscBool use_gd2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,use_gd2,2);
  ierr = PetscTryMethod(eps,"EPSGDSetDoubleExpansion_C",(EPS,PetscBool),(eps,use_gd2));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

