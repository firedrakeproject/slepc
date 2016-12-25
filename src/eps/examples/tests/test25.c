/*
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGNHEP problem with contour integral\n\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscOptionsGetString(NULL,NULL,"-f1",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-f2",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
    ierr = MatSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatLoad(B,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n");CHKERRQ(ierr);
    B = NULL;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps,1e-9,PETSC_DEFAULT);CHKERRQ(ierr);

  /* set CISS solver with various options */
  ierr = EPSSetType(eps,EPSCISS);CHKERRQ(ierr);
  ierr = EPSCISSSetExtraction(eps,EPS_CISS_EXTRACTION_HANKEL);CHKERRQ(ierr);
  ierr = EPSCISSSetQuadRule(eps,EPS_CISS_QUADRULE_CHEBYSHEV);CHKERRQ(ierr);
  ierr = EPSCISSSetUseST(eps,PETSC_TRUE);CHKERRQ(ierr);
  ierr = EPSGetRG(eps,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
  ierr = RGIntervalSetEndpoints(rg,-3000.0,0.0,0.0,0.0);CHKERRQ(ierr);

  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps,EPSCISS,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSCISSGetExtraction(eps,&extr);CHKERRQ(ierr);
    ierr = EPSCISSGetQuadRule(eps,&quad);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Solved with CISS using %s extraction with %s quadrature rule\n\n",EPSCISSExtractions[extr],EPSCISSQuadRules[quad]);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSErrorView(eps,EPS_ERROR_BACKWARD,NULL);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

