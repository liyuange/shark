set -e
# get tname from argv
tname=$1
./test-$tname 2 &> /dev/null
./test-$tname 1 &> /dev/null &
./test-$tname 0 &> /dev/null
echo "passed"
