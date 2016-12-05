#bin/bash
#Andrew

f="starting";
a="artifact-1.2.3.zip"; a="${a#*-}"; echo "${a%.*}"
