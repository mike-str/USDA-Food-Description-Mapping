# split input file into chunks of 200 lines
head -n 1 input_desc_list.csv > head
sed -n '2,200p' input_desc_list.csv > tmp ; cat head tmp > input_desc_list_1to200.csv
sed -n '201,400p' input_desc_list.csv > tmp ; cat head tmp >  input_desc_list_201to400.csv
sed -n '401,600p' input_desc_list.csv > tmp ; cat head tmp > input_desc_list_401to600.csv
sed -n '601,800p' input_desc_list.csv > tmp ; cat head tmp > input_desc_list_601to800.csv
sed -n '801,1000p' input_desc_list.csv > tmp ; cat head tmp > input_desc_list_801to1000.csv
sed -n '1001,1200p' input_desc_list.csv > tmp ; cat head tmp > input_desc_list_1001to1200.csv
rm tmp head
