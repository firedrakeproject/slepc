/*
   Options of polynomial JD solver.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include "pjdp.h"

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart_JD"
PetscErrorCode PEPJDSetRestart_JD(PEP pep,PetscReal keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) pjd->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    pjd->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart"
/*@
   PEPJDSetRestart - Sets the restart parameter for the Jacobi-Davidson
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_jd_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPJDGetRestart()
@*/
PetscErrorCode PEPJDSetRestart(PEP pep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  ierr = PetscTryMethod(pep,"PEPJDSetRestart_C",(PEP,PetscReal),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart_JD"
PetscErrorCode PEPJDGetRestart_JD(PEP pep,PetscReal *keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *keep = pjd->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart"
/*@
   PEPJDGetRestart - Gets the restart parameter used in the Jacobi-Davidson method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPJDSetRestart()
@*/
PetscErrorCode PEPJDGetRestart(PEP pep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscUseMethod(pep,"PEPJDGetRestart_C",(PEP,PetscReal*),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_JD"
PetscErrorCode PEPSetFromOptions_JD(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscReal      r1;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP JD Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pep_jd_restart","Proportion of vectors kept after restart","PEPJDSetRestart",0.5,&r1,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPJDSetRestart(pep,r1);CHKERRQ(ierr);
  }
  /* Set STPRECOND as the default ST */
  if (!pep->st) { ierr = PEPGetST(pep,&pep->st);CHKERRQ(ierr); }
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STPRECOND);CHKERRQ(ierr);
  }

  /* Set the default options of the KSP */
  ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPBCGSL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1e-5,PETSC_DEFAULT,PETSC_DEFAULT,100);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_JD"
PetscErrorCode PEPView_JD(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: %d%% of basis vectors kept after restart\n",(int)(100*pjd->keep));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

