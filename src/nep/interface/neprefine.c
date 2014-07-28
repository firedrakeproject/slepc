/*
   Newton refinement for NEP, simple version.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "NEPNewtonRefinementSimple"
PetscErrorCode NEPNewtonRefinementSimple(NEP nep,PetscInt *maxits,PetscReal *tol,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(NEP_Refine,nep,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(NEP_Refine,nep,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

