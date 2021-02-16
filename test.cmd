modelTests.exe -or (echo "Model Test Failed";  exit 1)
apiTests.exe -or (echo "API Test Failed";  exit 1)
linter.exe -or (echo "Linter Test Failed";  exit 1)
exit 0
