AC_INIT([grid-lma], [1.0], [michaellynch628@gmail.com])
AC_PROG_CXX
AC_CONFIG_FILES([Makefile])

# Set installation prefix (default: /usr/local)
AC_PREFIX_DEFAULT([/usr/local])

# Check for Grid installation
AC_ARG_WITH([grid],
    [AS_HELP_STRING([--with-grid=PATH], [path to Grid installation])],
    [grid_prefix="$withval"],
    [grid_prefix=""])

# Require Grid installation path to be specified
if test "x$grid_prefix" = "x"; then
    AC_MSG_ERROR([
Grid installation path must be specified:
  ./configure --with-grid=PATH
])
fi

# Verify Grid installation
if ! test -x "$grid_prefix/bin/grid-config"; then
    AC_MSG_ERROR([
grid-config not found at: $grid_prefix/bin/grid-config
Please verify the Grid installation path is correct.
])
fi

AC_MSG_NOTICE([Using Grid installation at: $grid_prefix])

# Test that grid-config works
AC_MSG_CHECKING([if grid-config works])
if "$grid_prefix/bin/grid-config" --version >/dev/null 2>&1; then
    AC_MSG_RESULT([yes])
    grid_version=$("$grid_prefix/bin/grid-config" --version 2>/dev/null || echo "unknown")
    AC_MSG_NOTICE([Grid version: $grid_version])
else
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([grid-config at $grid_prefix/bin/grid-config is not working])
fi

# Add debug support
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [enable debug symbols (default: no)])],
    [debug="$enableval"],
    [debug="no"])

if test "x$debug" = "xyes"; then
    AC_SUBST([DEBUGFLAGS], [-g])
    AC_MSG_NOTICE([Debug symbols enabled (-g)])
else
    AC_SUBST([DEBUGFLAGS], [""])
fi

AC_SUBST([GRID_PREFIX], [$grid_prefix])
AC_SUBST([PREFIX], [$prefix])
AC_OUTPUT

echo ""
echo "Configuration summary:"
echo "  Installation prefix: $prefix"
echo "  Grid installation:   $grid_prefix"
echo "  Debug symbols:       $debug"
echo ""
echo "Now run 'make' to build the project."
