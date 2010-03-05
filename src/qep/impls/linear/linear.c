/*                       

   Straightforward linearization for quadratic eigenproblems.

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

#include "private/qepimpl.h"                /*I "slepcqep.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  PetscInt dummy;
} QEP_LINEAR;

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp_LINEAR"
PetscErrorCode QEPSetUp_LINEAR(QEP qep)
{
  SETERRQ(1,"Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSolve_LINEAR"
PetscErrorCode QEPSolve_LINEAR(QEP qep)
{
  SETERRQ(1,"Not implemented yet");
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "QEPCreate_LINEAR"
PetscErrorCode QEPCreate_LINEAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_LINEAR     *linear;

  PetscFunctionBegin;
  ierr = PetscNew(QEP_LINEAR,&linear);CHKERRQ(ierr);
  PetscLogObjectMemory(qep,sizeof(QEP_LINEAR));
  qep->data                      = (void *) linear;
  qep->ops->solve                = QEPSolve_LINEAR;
  qep->ops->setup                = QEPSetUp_LINEAR;
  qep->ops->destroy              = QEPDestroy_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

