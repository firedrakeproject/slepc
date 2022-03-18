/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a GNHEP problem with contour integral. "
  "Based on ex7.\n"
  "The command line options are:\n"
  "  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  EPS               eps;
  RG                rg;
  Mat               A,B;
  PetscBool         flg;
  EPSCISSExtraction extr;
  EPSCISSQuadRule   quad;
  char              filename[PETSC_MAX_PATH_LEN];
  PetscViewer       viewer;
  PetscErrorCode    ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGNHEP problem with contour integral\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
    CHKERRQ(MatSetFromOptions(B));
    CHKERRQ(MatLoad(B,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n"));
    B = NULL;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GNHEP));
  CHKERRQ(EPSSetTolerances(eps,1e-9,PETSC_DEFAULT));

  /* set CISS solver with various options */
  CHKERRQ(EPSSetType(eps,EPSCISS));
  CHKERRQ(EPSCISSSetExtraction(eps,EPS_CISS_EXTRACTION_HANKEL));
  CHKERRQ(EPSCISSSetQuadRule(eps,EPS_CISS_QUADRULE_CHEBYSHEV));
  CHKERRQ(EPSCISSSetUseST(eps,PETSC_TRUE));
  CHKERRQ(EPSGetRG(eps,&rg));
  CHKERRQ(RGSetType(rg,RGINTERVAL));
  CHKERRQ(RGIntervalSetEndpoints(rg,-3000.0,0.0,0.0,0.0));

  CHKERRQ(EPSSetFromOptions(eps));

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSCISS,&flg));
  if (flg) {
    CHKERRQ(EPSCISSGetExtraction(eps,&extr));
    CHKERRQ(EPSCISSGetQuadRule(eps,&quad));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solved with CISS using %s extraction with %s quadrature rule\n\n",EPSCISSExtractions[extr],EPSCISSQuadRules[quad]));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_BACKWARD,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc
      output_file: output/test25_1.out
      test:
         suffix: 1
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: 1_cuda
         args: -mat_type aijcusparse -st_pc_factor_mat_solver_type cusparse
         requires: cuda double !complex !defined(PETSC_USE_64BIT_INDICES)

   testset:
      args: -f1 ${DATAFILESPATH}/matrices/complex/qc324.petsc -rg_type ellipse -rg_ellipse_center 1-0.09i -rg_ellipse_radius 0.15 -rg_ellipse_vscale 0.1
      output_file: output/test25_2.out
      test:
         suffix: 2
         requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: 2_cuda
         args: -mat_type aijcusparse -st_pc_factor_mat_solver_type cusparse
         requires: cuda double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)

TEST*/
