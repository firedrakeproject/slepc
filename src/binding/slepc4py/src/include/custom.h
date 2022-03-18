#ifndef SLEPC4PY_CUSTOM_H
#define SLEPC4PY_CUSTOM_H

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON ((PetscErrorCode)(-1))
#endif

#undef  __FUNCT__
#define __FUNCT__ "SlepcInitializePackageAll"
static PetscErrorCode SlepcInitializePackageAll(void)
{
  PetscFunctionBegin;
  CHKERRQ(EPSInitializePackage());
  CHKERRQ(SVDInitializePackage());
  CHKERRQ(PEPInitializePackage());
  CHKERRQ(NEPInitializePackage());
  CHKERRQ(MFNInitializePackage());
  CHKERRQ(STInitializePackage());
  CHKERRQ(BVInitializePackage());
  CHKERRQ(DSInitializePackage());
  CHKERRQ(FNInitializePackage());
  CHKERRQ(RGInitializePackage());
  PetscFunctionReturn(0);
}

#endif/*SLEPC4PY_CUSTOM_H*/
