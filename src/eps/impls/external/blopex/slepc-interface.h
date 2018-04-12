/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(SLEPC_INTERFACE_HEADER)
#define SLEPC_INTERFACE_HEADER

#include <blopex_lobpcg.h>
#include "petsc-interface.h"

PETSC_INTERN PetscInt slepc_blopex_useconstr;

extern int
SLEPCSetupInterpreter(mv_InterfaceInterpreter *ii);

extern void
SLEPCSetupInterpreterForDignifiedDeath(mv_InterfaceInterpreter *i);

#endif

