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

static char help[] = "Test RG interface functions.\n\n";

#include <slepcrg.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  RG             rg;
  PetscInt       i,inside;
  PetscReal      re,im;
  PetscScalar    ar,ai,cr[10],ci[10],vr[7],vi[7];

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = RGCreate(PETSC_COMM_WORLD,&rg);CHKERRQ(ierr);

  /* ellipse */
  ierr = RGSetType(rg,RGELLIPSE);CHKERRQ(ierr);
  ierr = RGEllipseSetParameters(rg,1.1,2,0.1);CHKERRQ(ierr);
  ierr = RGSetFromOptions(rg);CHKERRQ(ierr);
  ierr = RGView(rg,NULL);CHKERRQ(ierr);
  re = 0.1; im = 0.3;
#if defined(PETSC_USE_COMPLEX)
  ar = re+im*PETSC_i;
#else
  ar = re; ai = im;
#endif
  ierr = RGCheckInside(rg,1,&ar,&ai,&inside);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Point (%g,%g) is %s the region\n",(double)re,(double)im,(inside>=0)?"inside":"outside");

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Contour points: ");
  ierr = RGComputeContour(rg,10,cr,ci);CHKERRQ(ierr);
  for (i=0;i<10;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(cr[i]);
    im = PetscImaginaryPart(cr[i]);
#else
    re = cr[i];
    im = ci[i];
#endif
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(%.3g,%.3g) ",(double)re,(double)im);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");

  /* interval */
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
  ierr = RGIntervalSetEndpoints(rg,-1,1,-0.1,0.1);CHKERRQ(ierr);
  ierr = RGSetFromOptions(rg);CHKERRQ(ierr);
  ierr = RGView(rg,NULL);CHKERRQ(ierr);
  re = 0.2; im = 0;
#if defined(PETSC_USE_COMPLEX)
  ar = re+im*PETSC_i;
#else
  ar = re; ai = im;
#endif
  ierr = RGCheckInside(rg,1,&ar,&ai,&inside);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Point (%g,%g) is %s the region\n",(double)re,(double)im,(inside>=0)?"inside":"outside");

  /* polygon */
#if defined(PETSC_USE_COMPLEX)
  vr[0] = 0.0+2.0*PETSC_i;
  vr[1] = 1.0+4.0*PETSC_i;
  vr[2] = 2.0+5.0*PETSC_i;
  vr[3] = 4.0+3.0*PETSC_i;
  vr[4] = 5.0+4.0*PETSC_i;
  vr[5] = 6.0+1.0*PETSC_i;
  vr[6] = 2.0+0.0*PETSC_i;
#else
  vr[0] = 0.0; vi[0] = 2.0;
  vr[1] = 1.0; vi[1] = 4.0;
  vr[2] = 2.0; vi[2] = 5.0;
  vr[3] = 4.0; vi[3] = 3.0;
  vr[4] = 5.0; vi[4] = 4.0;
  vr[5] = 6.0; vi[5] = 1.0;
  vr[6] = 2.0; vi[6] = 0.0;
#endif
  ierr = RGSetType(rg,RGPOLYGON);CHKERRQ(ierr);
  ierr = RGPolygonSetVertices(rg,7,vr,vi);CHKERRQ(ierr);
  ierr = RGSetFromOptions(rg);CHKERRQ(ierr);
  ierr = RGView(rg,NULL);CHKERRQ(ierr);
  re = 5; im = 0.9;
#if defined(PETSC_USE_COMPLEX)
  ar = re+im*PETSC_i;
#else
  ar = re; ai = im;
#endif
  ierr = RGCheckInside(rg,1,&ar,&ai,&inside);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Point (%g,%g) is %s the region\n",(double)re,(double)im,(inside>=0)?"inside":"outside");

  ierr = RGDestroy(&rg);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
