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

static char help[] = "Test the ring region.\n\n";

#include <slepcrg.h>

PetscErrorCode CheckPoint(RG rg,PetscReal re,PetscReal im)
{
  PetscErrorCode ierr;
  PetscInt       inside;
  PetscScalar    ar,ai;

  PetscFunctionBeginUser;
#if defined(PETSC_USE_COMPLEX)
  ar = re+im*PETSC_i;
#else
  ar = re; ai = im;
#endif
  ierr = RGCheckInside(rg,1,&ar,&ai,&inside);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Point (%g,%g) is %s the region\n",(double)re,(double)im,(inside>=0)?"inside":"outside");
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  RG             rg;
  RGType         rtype;
  PetscInt       i;
  PetscBool      triv;
  PetscReal      re,im,radius,vscale,start_ang,end_ang,width;
  PetscScalar    center,cr[12],ci[12];

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = RGCreate(PETSC_COMM_WORLD,&rg);CHKERRQ(ierr);

  ierr = RGSetType(rg,RGRING);CHKERRQ(ierr);
  ierr = RGIsTrivial(rg,&triv);CHKERRQ(ierr);
  if (!triv) SETERRQ(PETSC_COMM_SELF,1,"Region should be trivial before setting parameters");
  ierr = RGRingSetParameters(rg,2,PETSC_DEFAULT,0.5,PETSC_DEFAULT,0.25,0.1);CHKERRQ(ierr);
  ierr = RGSetFromOptions(rg);CHKERRQ(ierr);
  ierr = RGIsTrivial(rg,&triv);CHKERRQ(ierr);
  if (triv) SETERRQ(PETSC_COMM_SELF,1,"Region should be non-trivial after setting parameters");
  ierr = RGView(rg,NULL);CHKERRQ(ierr);

  ierr = RGGetType(rg,&rtype);CHKERRQ(ierr);
  ierr = RGRingGetParameters(rg,&center,&radius,&vscale,&start_ang,&end_ang,&width);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s region: \n  center=%g, radius=%g, vscale=%g\n  start angle=%g, end angle=%g, width=%g\n\n",rtype,(double)PetscRealPart(center),(double)radius,(double)vscale,(double)start_ang,(double)end_ang,(double)width);

  ierr = CheckPoint(rg,3.0,0.3);CHKERRQ(ierr);
  ierr = CheckPoint(rg,2.8253,0.28253);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nContour points: ");
  ierr = RGComputeContour(rg,12,cr,ci);CHKERRQ(ierr);
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

  ierr = RGDestroy(&rg);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
