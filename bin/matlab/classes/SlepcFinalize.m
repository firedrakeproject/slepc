function err = SlepcFinalize()
%
%
err = calllib('libslepc', 'SlepcFinalize');PetscCHKERRQ(err);

