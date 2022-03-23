/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test EPS with different builds with a matrix loaded from a file.\n"
  "This test is based on ex4.c in tutorials.\n"
  "It loads test matrices available in PETSc's distribution.\n"
  "Add -symm or -herm to select the symmetric/Hermitian matrix.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  char           filename[PETSC_MAX_PATH_LEN];
  const char     *prefix,*scalar,*ints,*floats;
  PetscReal      tol=PETSC_SMALL;
  PetscViewer    viewer;
  PetscBool      flg,symm;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-symm",&symm));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-herm",&flg));
  if (flg) symm=PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
  prefix = symm? "hpd": "nh";
  scalar = "complex";
#else
  prefix = symm? "spd": "ns";
  scalar = "real";
#endif
#if defined(PETSC_USE_64BIT_INDICES)
  ints   = "int64";
#else
  ints   = "int32";
#endif
#if defined(PETSC_USE_REAL_DOUBLE)
  floats = "float64";
#elif defined(PETSC_USE_REAL_SINGLE)
  floats = "float32";
#endif

  CHKERRQ(PetscSNPrintf(filename,sizeof(filename),"%s/share/petsc/datafiles/matrices/%s-%s-%s-%s",PETSC_DIR,prefix,scalar,ints,floats));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nReading matrix from binary file...\n\n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  if (symm) CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  else CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !__float128

   testset:
      args: -eps_nev 3
      requires: !complex
      filter: sed -e "s/92073/92072/" | sed -e "s/80649/80648/" | sed -e "s/80647/80648/" | sed -e "s/45755/45756/"
      output_file: output/test5_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur subspace arnoldi gd}}
      test:
         suffix: 1_power
         args: -eps_type power -st_type sinvert -eps_target 7
      test:
         suffix: 1_jd
         args: -eps_type jd -eps_jd_minv 3 -eps_jd_plusk 1
      test:
         suffix: 1_gd
         args: -eps_type gd -eps_gd_minv 3 -eps_gd_plusk 1
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion

   testset:
      args: -eps_nev 3
      requires: double complex
      output_file: output/test5_1_complex.out
      test:
         suffix: 1_complex
         args: -eps_type {{krylovschur subspace arnoldi gd}}
      test:
         suffix: 1_power_complex
         args: -eps_type power -st_type sinvert -eps_target 7
      test:
         suffix: 1_jd_complex
         args: -eps_type jd -eps_jd_minv 3 -eps_jd_plusk 1
      test:
         suffix: 1_gd_complex
         args: -eps_type gd -eps_gd_minv 3 -eps_gd_plusk 1
      test:
         suffix: 1_gd2_complex
         args: -eps_type gd -eps_gd_double_expansion

   testset:
      args: -symm -eps_nev 4 -eps_smallest_real
      requires: double !complex
      output_file: output/test5_2.out
      test:
        suffix: 2_arpack
        args: -eps_type arpack
        requires: arpack
      test:
        suffix: 2_blopex
        args: -eps_type blopex
        requires: blopex
      test:
        suffix: 2_trlan
        args: -eps_type trlan
        requires: trlan

   testset:
      args: -symm -eps_nev 4 -eps_smallest_real
      requires: double complex
      output_file: output/test5_2_complex.out
      test:
        suffix: 2_blopex_complex
        args: -eps_type blopex
        requires: blopex

TEST*/
