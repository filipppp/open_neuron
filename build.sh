#!/usr/bin/env bash

ROOTDIR=`pwd`

BUILDDIR=$ROOTDIR/build/
DISTDIR=$ROOTDIR/dist/

mkdir $BUILDDIR
cd $BUILDDIR
cmake $ROOTDIR || exit 1

make


exit 0
