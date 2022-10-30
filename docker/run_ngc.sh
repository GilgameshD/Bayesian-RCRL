###
 # @Author: Wenhao Ding
 # @Email: wenhaod@andrew.cmu.edu
 # @Date: 2022-08-18 03:14:33
 # @LastEditTime: 2022-10-30 15:10:36
 # @Description: 
### 

JSON_PATH='./json'
FILES=$(ls $JSON_PATH)

for files in $FILES
do  
echo 'runing file:' $files
ngc batch run -f files
done  

