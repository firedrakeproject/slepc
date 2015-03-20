/*
   Phi_1 function  phi_1(x) = (exp(x)-1)/x

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/fnimpl.h>      /*I "slepcfn.h" I*/

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Phi"
PetscErrorCode FNEvaluateFunction_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  *y = (PetscExpScalar(x)-1.0)/x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Phi"
PetscErrorCode FNEvaluateDerivative_Phi(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscScalar expx,phi1;

  PetscFunctionBegin;
  expx = PetscExpScalar(x);
  phi1 = (expx-1.0)/x;
  *y = (expx-phi1)/x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Phi"
PetscErrorCode FNView_Phi(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Phi_1: (exp(x)-1)/x\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Phi_1: (exp(%s*x)-1)/(%s*x)\n",str,str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Phi_1: %s*(exp(x)-1)/x\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Phi_1: %s",str);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*(exp(%s*x)-1)/(%s*x)\n",str,str);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate_Phi"
PETSC_EXTERN PetscErrorCode FNCreate_Phi(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction    = FNEvaluateFunction_Phi;
  fn->ops->evaluatederivative  = FNEvaluateDerivative_Phi;
  fn->ops->view                = FNView_Phi;
  PetscFunctionReturn(0);
}

