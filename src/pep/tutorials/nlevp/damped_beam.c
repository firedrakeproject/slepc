/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This example implements one of the problems found at
       NLEVP: A Collection of Nonlinear Eigenvalue Problems,
       The University of Manchester.
   The details of the collection can be found at:
       [1] T. Betcke et al., "NLEVP: A Collection of Nonlinear Eigenvalue
           Problems", ACM Trans. Math. Software 39(2), Article 7, 2013.

   The damped_beam problem is a QEP from the vibrarion analysis of a beam
   simply supported at both ends and damped in the middle.
*/

static char help[] = "Quadratic eigenproblem from the vibrarion analysis of a beam.\n\n"
  "The command line options are:\n"
  "  -n <n> ... dimension of the matrices.\n\n";

#include <slepcpep.h>

int main(int argc,char **argv)
{
  Mat            M,Mo,C,K,Ko,A[3]; /* problem matrices */
  PEP            pep;              /* polynomial eigenproblem solver context */
  IS             isf,isbc,is;
  PetscInt       n=200,nele,Istart,Iend,i,j,mloc,nloc,bc[2];
  PetscReal      width=0.05,height=0.005,glength=1.0,dlen,EI,area,rho;
  PetscScalar    K1[4],K2[4],K2t[4],K3[4],M1[4],M2[4],M2t[4],M3[4],damp=5.0;
  PetscBool      terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  nele = n/2;
  n    = 2*nele;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSimply supported beam damped in the middle, n=%" PetscInt_FMT " (nele=%" PetscInt_FMT ")\n\n",n,nele));

  dlen = glength/nele;
  EI   = 7e10*width*height*height*height/12.0;
  area = width*height;
  rho  = 0.674/(area*glength);

  K1[0]  =  12;  K1[1]  =   6*dlen;  K1[2]  =   6*dlen;  K1[3]  =  4*dlen*dlen;
  K2[0]  = -12;  K2[1]  =   6*dlen;  K2[2]  =  -6*dlen;  K2[3]  =  2*dlen*dlen;
  K2t[0] = -12;  K2t[1] =  -6*dlen;  K2t[2] =   6*dlen;  K2t[3] =  2*dlen*dlen;
  K3[0]  =  12;  K3[1]  =  -6*dlen;  K3[2]  =  -6*dlen;  K3[3]  =  4*dlen*dlen;
  M1[0]  = 156;  M1[1]  =  22*dlen;  M1[2]  =  22*dlen;  M1[3]  =  4*dlen*dlen;
  M2[0]  =  54;  M2[1]  = -13*dlen;  M2[2]  =  13*dlen;  M2[3]  = -3*dlen*dlen;
  M2t[0] =  54;  M2t[1] =  13*dlen;  M2t[2] = -13*dlen;  M2t[3] = -3*dlen*dlen;
  M3[0]  = 156;  M3[1]  = -22*dlen;  M3[2]  = -22*dlen;  M3[3]  =  4*dlen*dlen;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is block-tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&Ko));
  PetscCall(MatSetSizes(Ko,PETSC_DECIDE,PETSC_DECIDE,n+2,n+2));
  PetscCall(MatSetBlockSize(Ko,2));
  PetscCall(MatSetFromOptions(Ko));
  PetscCall(MatSetUp(Ko));

  PetscCall(MatGetOwnershipRange(Ko,&Istart,&Iend));
  for (i=Istart/2;i<Iend/2;i++) {
    if (i>0) {
      j = i-1;
      PetscCall(MatSetValuesBlocked(Ko,1,&i,1,&j,K2t,ADD_VALUES));
      PetscCall(MatSetValuesBlocked(Ko,1,&i,1,&i,K3,ADD_VALUES));
    }
    if (i<nele) {
      j = i+1;
      PetscCall(MatSetValuesBlocked(Ko,1,&i,1,&j,K2,ADD_VALUES));
      PetscCall(MatSetValuesBlocked(Ko,1,&i,1,&i,K1,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Ko,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Ko,MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(Ko,EI/(dlen*dlen*dlen)));

  /* M is block-tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&Mo));
  PetscCall(MatSetSizes(Mo,PETSC_DECIDE,PETSC_DECIDE,n+2,n+2));
  PetscCall(MatSetBlockSize(Mo,2));
  PetscCall(MatSetFromOptions(Mo));
  PetscCall(MatSetUp(Mo));

  PetscCall(MatGetOwnershipRange(Mo,&Istart,&Iend));
  for (i=Istart/2;i<Iend/2;i++) {
    if (i>0) {
      j = i-1;
      PetscCall(MatSetValuesBlocked(Mo,1,&i,1,&j,M2t,ADD_VALUES));
      PetscCall(MatSetValuesBlocked(Mo,1,&i,1,&i,M3,ADD_VALUES));
    }
    if (i<nele) {
      j = i+1;
      PetscCall(MatSetValuesBlocked(Mo,1,&i,1,&j,M2,ADD_VALUES));
      PetscCall(MatSetValuesBlocked(Mo,1,&i,1,&i,M1,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Mo,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Mo,MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(Mo,rho*area*dlen/420));

  /* remove rows/columns from K and M corresponding to boundary conditions */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,Iend-Istart,Istart,1,&isf));
  bc[0] = 0; bc[1] = n;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,2,bc,PETSC_USE_POINTER,&isbc));
  PetscCall(ISDifference(isf,isbc,&is));
  PetscCall(MatCreateSubMatrix(Ko,is,is,MAT_INITIAL_MATRIX,&K));
  PetscCall(MatCreateSubMatrix(Mo,is,is,MAT_INITIAL_MATRIX,&M));
  PetscCall(MatGetLocalSize(M,&mloc,&nloc));

  /* C is zero except for the (nele,nele)-entry */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,mloc,nloc,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  if (nele-1>=Istart && nele-1<Iend) PetscCall(MatSetValue(C,nele-1,nele-1,damp,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetFromOptions(pep));
  PetscCall(PEPSolve(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(PEPConvergedReasonView(pep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(PEPDestroy(&pep));
  PetscCall(ISDestroy(&isf));
  PetscCall(ISDestroy(&isbc));
  PetscCall(ISDestroy(&is));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&Ko));
  PetscCall(MatDestroy(&Mo));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -pep_nev 2 -pep_ncv 12 -pep_target 0 -terse
      requires: !single
      output_file: output/damped_beam_1.out
      test:
         suffix: 1
         args: -pep_type {{toar linear}} -st_type sinvert
      test:
         suffix: 1_qarnoldi
         args: -pep_type qarnoldi -pep_qarnoldi_locking 0 -st_type sinvert
      test:
         suffix: 1_jd
         args: -pep_type jd

   testset:
      args: -pep_nev 2 -pep_ncv 12 -pep_target 1i -terse
      requires: complex !single
      output_file: output/damped_beam_1.out
      test:
         suffix: 1_complex
         args: -pep_type {{toar linear}} -st_type sinvert
      test:
         suffix: 1_qarnoldi_complex
         args: -pep_type qarnoldi -pep_qarnoldi_locking 0 -st_type sinvert
      test:
         suffix: 1_jd_complex
         args: -pep_type jd

TEST*/
