/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCheck(size<=2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This test needs one or two processes");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"VecComp test\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create standard vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecSetSizes(v,8/size,8));
  PetscCall(VecSetFromOptions(v));

  if (!rank) {
    PetscCall(VecSetValue(v,0,2.0,INSERT_VALUES));
    PetscCall(VecSetValue(v,1,-1.0,INSERT_VALUES));
    PetscCall(VecSetValue(v,2,3.0,INSERT_VALUES));
    PetscCall(VecSetValue(v,3,3.5,INSERT_VALUES));
  }
  if ((!rank && size==1) || (rank && size==2)) {
    PetscCall(VecSetValue(v,4,1.2,INSERT_VALUES));
    PetscCall(VecSetValue(v,5,1.8,INSERT_VALUES));
    PetscCall(VecSetValue(v,6,-2.2,INSERT_VALUES));
    PetscCall(VecSetValue(v,7,2.0,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecDuplicate(v,&w));
  PetscCall(VecSet(w,1.0));
  PetscCall(VecDuplicate(v,&x));
  PetscCall(VecDuplicate(v,&y));
  if (!rank) PetscCall(VecSetValue(y,0,1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create veccomp vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecCreate(PETSC_COMM_WORLD,&vparent));
  PetscCall(VecSetSizes(vparent,4/size,4));
  PetscCall(VecSetFromOptions(vparent));

  /* create a veccomp vector with two subvectors */
  PetscCall(VecDuplicate(vparent,&vchild[0]));
  PetscCall(VecDuplicate(vparent,&vchild[1]));
  if (!rank) {
    PetscCall(VecSetValue(vchild[0],0,2.0,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[0],1,-1.0,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[1],0,1.2,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[1],1,1.8,INSERT_VALUES));
  }
  if ((!rank && size==1) || (rank && size==2)) {
    PetscCall(VecSetValue(vchild[0],2,3.0,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[0],3,3.5,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[1],2,-2.2,INSERT_VALUES));
    PetscCall(VecSetValue(vchild[1],3,2.0,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(vchild[0]));
  PetscCall(VecAssemblyBegin(vchild[1]));
  PetscCall(VecAssemblyEnd(vchild[0]));
  PetscCall(VecAssemblyEnd(vchild[1]));
  PetscCall(VecCreateCompWithVecs(vchild,2,vparent,&vc));
  PetscCall(VecDestroy(&vchild[0]));
  PetscCall(VecDestroy(&vchild[1]));
  PetscCall(VecView(vc,NULL));

  PetscCall(VecGetSize(vc,&k));
  PetscCheck(k==8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vector global length should be 8");

  /* create an empty veccomp vector with two subvectors */
  Nx[0] = 4;
  Nx[1] = 4;
  PetscCall(VecCreateComp(PETSC_COMM_WORLD,Nx,2,VECSTANDARD,vparent,&wc));
  PetscCall(VecCompGetSubVecs(wc,&n,&varray));
  PetscCheck(n==2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"n should be 2");
  for (i=0;i<2;i++) PetscCall(VecSet(varray[i],1.0));

  PetscCall(VecGetSize(wc,&k));
  PetscCheck(k==8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vector global length should be 8");

  /* duplicate a veccomp */
  PetscCall(VecDuplicate(vc,&xc));

  /* create a veccomp via VecSetType */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&yc));
  PetscCall(VecSetType(yc,VECCOMP));
  PetscCall(VecSetSizes(yc,8/size,8));
  PetscCall(VecCompSetSubVecs(yc,2,NULL));

  PetscCall(VecCompGetSubVecs(yc,&n,&varray));
  PetscCheck(n==2,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"n should be 2");
  if (!rank) PetscCall(VecSetValue(varray[0],0,1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(varray[0]));
  PetscCall(VecAssemblyEnd(varray[0]));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Operate with vectors
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecCopy(w,x));
  PetscCall(VecAXPBY(x,1.0,-2.0,v));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecCopy(wc,xc));
  PetscCall(VecAXPBY(xc,1.0,-2.0,vc));
  PetscCall(VecNorm(xc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecCopy(w,x));
  PetscCall(VecWAXPY(x,-2.0,w,v));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(VecCopy(wc,xc));
  PetscCall(VecWAXPY(xc,-2.0,wc,vc));
  PetscCall(VecNorm(xc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecAXPBYPCZ(y,3.0,-1.0,1.0,w,v));
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(VecAXPBYPCZ(yc,3.0,-1.0,1.0,wc,vc));
  PetscCall(VecNorm(yc,NORM_2,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecMax(xc,NULL,&vmax));
  PetscCall(VecMin(xc,NULL,&vmin));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"xc has max value %g min value %g\n",(double)vmax,(double)vmin));

  PetscCall(VecMaxPointwiseDivide(wc,xc,&vmax));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"wc/xc has max value %g\n",(double)vmax));

  PetscCall(VecDot(x,y,&dot[0]));
  PetscCall(VecDot(xc,yc,&dotc[0]));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  PetscCall(VecTDot(x,y,&dot[0]));
  PetscCall(VecTDot(xc,yc,&dotc[0]));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");

  vecs[0] = w; vecs[1] = y;
  PetscCall(VecMDot(x,2,vecs,dot));
  vecs[0] = wc; vecs[1] = yc;
  PetscCall(VecMDot(xc,2,vecs,dotc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON || PetscAbsScalar(dot[1]-dotc[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  vecs[0] = w; vecs[1] = y;
  PetscCall(VecMTDot(x,2,vecs,dot));
  vecs[0] = wc; vecs[1] = yc;
  PetscCall(VecMTDot(xc,2,vecs,dotc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON || PetscAbsScalar(dot[1]-dotc[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");

  PetscCall(VecDotNorm2(x,y,&dot[0],&norm));
  PetscCall(VecDotNorm2(xc,yc,&dotc[0],&normc));
  PetscCheck(PetscAbsScalar(dot[0]-dotc[0])<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Dots are different");
  PetscCheck(PetscAbsReal(norm-normc)<100*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecAbs(w));
  PetscCall(VecAbs(wc));
  PetscCall(VecConjugate(x));
  PetscCall(VecConjugate(xc));
  PetscCall(VecShift(y,0.5));
  PetscCall(VecShift(yc,0.5));
  PetscCall(VecReciprocal(y));
  PetscCall(VecReciprocal(yc));
  PetscCall(VecExp(y));
  PetscCall(VecExp(yc));
  PetscCall(VecLog(y));
  PetscCall(VecLog(yc));
  PetscCall(VecNorm(y,NORM_1,&norm));
  PetscCall(VecNorm(yc,NORM_1,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecPointwiseMult(w,x,y));
  PetscCall(VecPointwiseMult(wc,xc,yc));
  PetscCall(VecNorm(w,NORM_INFINITY,&norm));
  PetscCall(VecNorm(wc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecPointwiseMax(w,x,y));
  PetscCall(VecPointwiseMax(wc,xc,yc));
  PetscCall(VecNorm(w,NORM_INFINITY,&norm));
  PetscCall(VecNorm(wc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecSwap(x,y));
  PetscCall(VecSwap(xc,yc));
  PetscCall(VecPointwiseDivide(w,x,y));
  PetscCall(VecPointwiseDivide(wc,xc,yc));
  PetscCall(VecScale(w,0.3));
  PetscCall(VecScale(wc,0.3));
  PetscCall(VecSqrtAbs(w));
  PetscCall(VecSqrtAbs(wc));
  PetscCall(VecNorm(w,NORM_1_AND_2,norm12));
  PetscCall(VecNorm(wc,NORM_1_AND_2,norm12c));
  PetscCheck(PetscAbsReal(norm12[0]-norm12c[0])<10*PETSC_MACHINE_EPSILON || PetscAbsReal(norm12[1]-norm12c[1])>10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecPointwiseMin(w,x,y));
  PetscCall(VecPointwiseMin(wc,xc,yc));
  PetscCall(VecPointwiseMaxAbs(x,y,w));
  PetscCall(VecPointwiseMaxAbs(xc,yc,wc));
  PetscCall(VecNorm(x,NORM_INFINITY,&norm));
  PetscCall(VecNorm(xc,NORM_INFINITY,&normc));
  PetscCheck(PetscAbsReal(norm-normc)<10*PETSC_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Norms are different");

  PetscCall(VecSetRandom(wc,NULL));

  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&vparent));
  PetscCall(VecDestroy(&vc));
  PetscCall(VecDestroy(&wc));
  PetscCall(VecDestroy(&xc));
  PetscCall(VecDestroy(&yc));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
