/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test VecComp.\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  Vec            v,w,x,y,vc,wc,xc,yc,vparent,vchild[2],vecs[2];
  const Vec      *varray;
  PetscMPIInt    size,rank;
  PetscInt       i,n,k,Nx[2];
  PetscReal      norm,normc,norm12[2],norm12c[2],vmax,vmin;
  PetscScalar    dot[2],dotc[2];

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheck(size<=2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test needs one or two processes");
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"VecComp test\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create standard vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
  CHKERRQ(VecSetSizes(v,8/size,8));
  CHKERRQ(VecSetFromOptions(v));

  if (!rank) {
    CHKERRQ(VecSetValue(v,0,2.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,1,-1.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,2,3.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,3,3.5,INSERT_VALUES));
  }
  if ((!rank && size==1) || (rank && size==2)) {
    CHKERRQ(VecSetValue(v,4,1.2,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,5,1.8,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,6,-2.2,INSERT_VALUES));
    CHKERRQ(VecSetValue(v,7,2.0,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecDuplicate(v,&w));
  CHKERRQ(VecSet(w,1.0));
  CHKERRQ(VecDuplicate(v,&x));
  CHKERRQ(VecDuplicate(v,&y));
  if (!rank) CHKERRQ(VecSetValue(y,0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create veccomp vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&vparent));
  CHKERRQ(VecSetSizes(vparent,4/size,4));
  CHKERRQ(VecSetFromOptions(vparent));

  /* create a veccomp vector with two subvectors */
  CHKERRQ(VecDuplicate(vparent,&vchild[0]));
  CHKERRQ(VecDuplicate(vparent,&vchild[1]));
  if (!rank) {
    CHKERRQ(VecSetValue(vchild[0],0,2.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[0],1,-1.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[1],0,1.2,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[1],1,1.8,INSERT_VALUES));
  }
  if ((!rank && size==1) || (rank && size==2)) {
    CHKERRQ(VecSetValue(vchild[0],2,3.0,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[0],3,3.5,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[1],2,-2.2,INSERT_VALUES));
    CHKERRQ(VecSetValue(vchild[1],3,2.0,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(vchild[0]));
  CHKERRQ(VecAssemblyBegin(vchild[1]));
  CHKERRQ(VecAssemblyEnd(vchild[0]));
  CHKERRQ(VecAssemblyEnd(vchild[1]));
  CHKERRQ(VecCreateCompWithVecs(vchild,2,vparent,&vc));
  CHKERRQ(VecDestroy(&vchild[0]));
  CHKERRQ(VecDestroy(&vchild[1]));
  CHKERRQ(VecView(vc,NULL));

  CHKERRQ(VecGetSize(vc,&k));
  PetscCheck(k==8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vector global length should be 8");

  /* create an empty veccomp vector with two subvectors */
  Nx[0] = 4;
  Nx[1] = 4;
  CHKERRQ(VecCreateComp(PETSC_COMM_WORLD,Nx,2,VECSTANDARD,vparent,&wc));
  CHKERRQ(VecCompGetSubVecs(wc,&n,&varray));
  PetscCheck(n==2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"n should be 2");
  for (i=0;i<2;i++) CHKERRQ(VecSet(varray[i],1.0));

  CHKERRQ(VecGetSize(wc,&k));
  PetscCheck(k==8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vector global length should be 8");

  /* duplicate a veccomp */
  CHKERRQ(VecDuplicate(vc,&xc));

  /* create a veccomp via VecSetType */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&yc));
  CHKERRQ(VecSetType(yc,VECCOMP));
  CHKERRQ(VecSetSizes(yc,8/size,8));
  CHKERRQ(VecCompSetSubVecs(yc,2,NULL));

  CHKERRQ(VecCompGetSubVecs(yc,&n,&varray));
  PetscCheck(n==2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"n should be 2");
  if (!rank) CHKERRQ(VecSetValue(varray[0],0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(varray[0]));
  CHKERRQ(VecAssemblyEnd(varray[0]));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Operate with vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(VecCopy(w,x));
  CHKERRQ(VecAXPBY(x,1.0,-2.0,v));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecCopy(wc,xc));
  CHKERRQ(VecAXPBY(xc,1.0,-2.0,vc));
  CHKERRQ(VecNorm(xc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecCopy(w,x));
  CHKERRQ(VecWAXPY(x,-2.0,w,v));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(VecCopy(wc,xc));
  CHKERRQ(VecWAXPY(xc,-2.0,wc,vc));
  CHKERRQ(VecNorm(xc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecAXPBYPCZ(y,3.0,-1.0,1.0,w,v));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(VecAXPBYPCZ(yc,3.0,-1.0,1.0,wc,vc));
  CHKERRQ(VecNorm(yc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecMax(xc,NULL,&vmax));
  CHKERRQ(VecMin(xc,NULL,&vmin));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"xc has max value %g min value %g\n",(double)vmax,(double)vmin));

  CHKERRQ(VecMaxPointwiseDivide(wc,xc,&vmax));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"wc/xc has max value %g\n",(double)vmax));

  CHKERRQ(VecDot(x,y,&dot[0]));
  CHKERRQ(VecDot(xc,yc,&dotc[0]));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  CHKERRQ(VecTDot(x,y,&dot[0]));
  CHKERRQ(VecTDot(xc,yc,&dotc[0]));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");

  vecs[0] = w; vecs[1] = y;
  CHKERRQ(VecMDot(x,2,vecs,dot));
  vecs[0] = wc; vecs[1] = yc;
  CHKERRQ(VecMDot(xc,2,vecs,dotc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON || PetscAbsScalar(dot[1]-dotc[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  vecs[0] = w; vecs[1] = y;
  CHKERRQ(VecMTDot(x,2,vecs,dot));
  vecs[0] = wc; vecs[1] = yc;
  CHKERRQ(VecMTDot(xc,2,vecs,dotc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON || PetscAbsScalar(dot[1]-dotc[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");

  CHKERRQ(VecDotNorm2(x,y,&dot[0],&norm));
  CHKERRQ(VecDotNorm2(xc,yc,&dotc[0],&normc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  PetscCheck(PetscAbsReal(norm-normc)<100*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecAbs(w));
  CHKERRQ(VecAbs(wc));
  CHKERRQ(VecConjugate(x));
  CHKERRQ(VecConjugate(xc));
  CHKERRQ(VecShift(y,0.5));
  CHKERRQ(VecShift(yc,0.5));
  CHKERRQ(VecReciprocal(y));
  CHKERRQ(VecReciprocal(yc));
  CHKERRQ(VecExp(y));
  CHKERRQ(VecExp(yc));
  CHKERRQ(VecLog(y));
  CHKERRQ(VecLog(yc));
  CHKERRQ(VecNorm(y,NORM_1,&norm));
  CHKERRQ(VecNorm(yc,NORM_1,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecPointwiseMult(w,x,y));
  CHKERRQ(VecPointwiseMult(wc,xc,yc));
  CHKERRQ(VecNorm(w,NORM_INFINITY,&norm));
  CHKERRQ(VecNorm(wc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecPointwiseMax(w,x,y));
  CHKERRQ(VecPointwiseMax(wc,xc,yc));
  CHKERRQ(VecNorm(w,NORM_INFINITY,&norm));
  CHKERRQ(VecNorm(wc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecSwap(x,y));
  CHKERRQ(VecSwap(xc,yc));
  CHKERRQ(VecPointwiseDivide(w,x,y));
  CHKERRQ(VecPointwiseDivide(wc,xc,yc));
  CHKERRQ(VecScale(w,0.3));
  CHKERRQ(VecScale(wc,0.3));
  CHKERRQ(VecSqrtAbs(w));
  CHKERRQ(VecSqrtAbs(wc));
  CHKERRQ(VecNorm(w,NORM_1_AND_2,norm12));
  CHKERRQ(VecNorm(wc,NORM_1_AND_2,norm12c));
  PetscCheck(PetscAbsReal(norm12[0]-norm12c[0])<10*PETSC_MACHINE_EPSILON || PetscAbsReal(norm12[1]-norm12c[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecPointwiseMin(w,x,y));
  CHKERRQ(VecPointwiseMin(wc,xc,yc));
  CHKERRQ(VecPointwiseMaxAbs(x,y,w));
  CHKERRQ(VecPointwiseMaxAbs(xc,yc,wc));
  CHKERRQ(VecNorm(x,NORM_INFINITY,&norm));
  CHKERRQ(VecNorm(xc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  CHKERRQ(VecSetRandom(wc,NULL));

  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&vparent));
  CHKERRQ(VecDestroy(&vc));
  CHKERRQ(VecDestroy(&wc));
  CHKERRQ(VecDestroy(&xc));
  CHKERRQ(VecDestroy(&yc));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
