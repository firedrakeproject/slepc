/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test setting FN parameters from the command line.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  FN             fn1,fn2;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fn1));
  PetscCall(FNSetOptionsPrefix(fn1,"f1_"));
  PetscCall(FNSetFromOptions(fn1));
  PetscCall(FNView(fn1,NULL));
  PetscCall(FNDestroy(&fn1));
  PetscCall(FNCreate(PETSC_COMM_WORLD,&fn2));
  PetscCall(FNSetOptionsPrefix(fn2,"f2_"));
  PetscCall(FNSetFromOptions(fn2));
  PetscCall(FNView(fn2,NULL));
  PetscCall(FNDestroy(&fn2));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -f1_fn_type exp -f1_fn_scale -2.5 -f2_fn_type rational -f2_fn_rational_numerator -1,1 -f2_fn_rational_denominator 1,-6,4

TEST*/
