/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test ST with four matrices and split preconditioner.\n\n";

#include <slepcst.h>

int main(int argc,char **argv)
{
  Mat            A,B,C,D,Pa,Pb,Pc,Pd,Pmat,mat[4];
  ST             st;
  KSP            ksp;
  PC             pc;
  Vec            v,w;
  STType         type;
  PetscScalar    sigma;
  PetscInt       n=10,i,Istart,Iend;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTest ST with four matrices, n=%" PetscInt_FMT "\n\n",n);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&D);CHKERRQ(ierr);
  ierr = MatSetSizes(D,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(D);CHKERRQ(ierr);
  ierr = MatSetUp(D);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
    if (i>0) {
      ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(B,i,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValue(B,i,i,-1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i<n-1) {
      ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatSetValue(C,i,n-i-1,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(D,i,i,i*.1,INSERT_VALUES);CHKERRQ(ierr);
    if (i==0) {
      ierr = MatSetValue(D,0,n-1,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i==n-1) {
      ierr = MatSetValue(D,n-1,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&v,&w);CHKERRQ(ierr);
  ierr = VecSet(v,1.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the split preconditioner matrices (four diagonals)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&Pa);CHKERRQ(ierr);
  ierr = MatSetSizes(Pa,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pa);CHKERRQ(ierr);
  ierr = MatSetUp(Pa);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Pb);CHKERRQ(ierr);
  ierr = MatSetSizes(Pb,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pb);CHKERRQ(ierr);
  ierr = MatSetUp(Pb);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Pc);CHKERRQ(ierr);
  ierr = MatSetSizes(Pc,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pc);CHKERRQ(ierr);
  ierr = MatSetUp(Pc);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Pd);CHKERRQ(ierr);
  ierr = MatSetSizes(Pd,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pd);CHKERRQ(ierr);
  ierr = MatSetUp(Pd);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(Pa,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(Pa,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
    if (i>0) {
      ierr = MatSetValue(Pb,i,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatSetValue(Pb,i,i,-1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatSetValue(Pd,i,i,i*.1,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(Pa,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pa,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pc,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the spectral transformation object
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = STCreate(PETSC_COMM_WORLD,&st);CHKERRQ(ierr);
  mat[0] = A;
  mat[1] = B;
  mat[2] = C;
  mat[3] = D;
  ierr = STSetMatrices(st,4,mat);CHKERRQ(ierr);
  mat[0] = Pa;
  mat[1] = Pb;
  mat[2] = Pc;
  mat[3] = Pd;
  ierr = STSetSplitPreconditioner(st,4,mat,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = STSetTransform(st,PETSC_TRUE);CHKERRQ(ierr);
  ierr = STSetFromOptions(st);CHKERRQ(ierr);
  ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Apply the operator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* sigma=0.0 */
  ierr = STSetUp(st);CHKERRQ(ierr);
  ierr = STGetType(st,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"ST type %s\n",type);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&Pmat);CHKERRQ(ierr);
  ierr = MatView(Pmat,NULL);CHKERRQ(ierr);
  ierr = STMatSolve(st,v,w);
  ierr = VecView(w,NULL);CHKERRQ(ierr);

  /* sigma=0.1 */
  sigma = 0.1;
  ierr = STSetShift(st,sigma);CHKERRQ(ierr);
  ierr = STGetShift(st,&sigma);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"With shift=%g\n",(double)PetscRealPart(sigma));CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&Pmat);CHKERRQ(ierr);
  ierr = MatView(Pmat,NULL);CHKERRQ(ierr);
  ierr = STMatSolve(st,v,w);
  ierr = VecView(w,NULL);CHKERRQ(ierr);

  ierr = STDestroy(&st);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&Pa);CHKERRQ(ierr);
  ierr = MatDestroy(&Pb);CHKERRQ(ierr);
  ierr = MatDestroy(&Pc);CHKERRQ(ierr);
  ierr = MatDestroy(&Pd);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -st_type {{shift sinvert}separate output} -st_pc_type jacobi
      requires: !single

TEST*/
