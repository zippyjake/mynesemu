#!/bin/sh

source="nesemu1.cc"
asmfile="nesemu1.s"
objfile="nesemu1.o"

# These first few complicated looking lines simply invoke
# Intel's compiler. For some reason they made it complicated to use.

# The compilation flags inhibit most of the compiler's optimization features.

I=/opt/intel/Compiler/11.1/072/bin
. $I/iccvars.sh intel64

LD_LIBRARY_PATH=/opt/intel/Compiler/11.1/072/idb/lib/intel64 \
/opt/intel/Compiler/11.1/072/bin/intel64/icc $source \
	$(pkg-config sdl --libs --cflags) -std=c++0x \
	-O0 -gdwarf-2 -debug all -S -fno-omit-frame-pointer \
	-use-intel-optimized-headers -fp-model fast=2 -no-ip \
	-unroll0 -no-vec -fno-jump-tables

# Then, translate the line&column information within the COMMENTS
# of the verbose assembler listing into actual labels with a special
# tag so we can recognize them later.

cat >line-trans.php <<EOF
<?php

$files = Array();

$fp = fopen('php://stdin', 'r');
$n=0;
$cur_file = 0;
while( !feof($fp) && ($s = fgets($fp,4000)) !== null)
{
  if(preg_match('/\.file +([0-9]+) +\"?([^"]*)\"?/', $s, $mat))
  {
    $files[$mat[1]] = $mat[2];
  }
  elseif(preg_match('/\.loc +([0-9]+) .*/', $s, $mat))
  {
    $cur_file = $mat[1];
  }
  elseif(preg_match('/.*#([0-9]+)\.([0-9].*)/', $s, $mat))
  {
    printf("BisqLine_%02d_%06d_%d_%d:\n",
      $cur_file, $n++, $mat[1], $mat[2]);
  }
  print $s;
}

#print_r($files);
EOF

php -q line-trans.php < $asmfile | tee test.s | as - -o $objfile

# Finally, link the emulator (with the libraries required by Intel's compiler)

g++ nesemu1.o -o nesemu1-dbg -lSDL /opt/intel/Compiler/11.1/072/lib/intel64/libirc.a

# From the resulting executable, produce a line number listing in text format.

nm nesemu1-dbg |egrep 'T main|BisqLine' > line-listings.txt
