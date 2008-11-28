/*
   Modification of the *temp* implementation of the BLOPEX multivector in order
   to wrap created PETSc vectors as multivectors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/


#ifndef SLEPC_INTERFACE_HEADER
#define SLEPC_INTERFACE_HEADER

#include "../src/contrib/blopex/petsc-interface/petsc-interface.h"

extern int
SLEPCSetupInterpreter( mv_InterfaceInterpreter *ii );

extern void
SLEPCSetupInterpreterForDignifiedDeath( mv_InterfaceInterpreter *i );

#endif

