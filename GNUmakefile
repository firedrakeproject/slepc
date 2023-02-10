#
# Target to build (update) the SLEPc libraries
all :
	+@$(MAKE) -f makefile --no-print-directory $@

ifeq ($(firstword $(sort 4.1.99 $(MAKE_VERSION))),4.1.99)
include gmakefile
endif

# Override PETSc reconfigure target
reconfigure:
	-@echo "The reconfigure target is not available in SLEPc"

# For any target that doesn't exist in gmakefile, use the legacy makefile (which has the logging features)
% :
	+@$(MAKE) -f makefile --no-print-directory $@
