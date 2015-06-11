/*
   Modification of the *temp* implementation of the BLOPEX multivector in order
   to wrap created PETSc vectors as multivectors.

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


#if !defined(SLEPC_INTERFACE_HEADER)
#define SLEPC_INTERFACE_HEADER

#include <blopex_lobpcg.h>
#include "petsc-interface.h"

#define SLEPC_BLOPEX_USECONSTR 1121

extern int
SLEPCSetupInterpreter(mv_InterfaceInterpreter *ii);

extern void
SLEPCSetupInterpreterForDignifiedDeath(mv_InterfaceInterpreter *i);

#endif

