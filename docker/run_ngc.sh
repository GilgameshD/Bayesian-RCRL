###
 # @Author: Wenhao Ding
 # @Email: wenhaod@andrew.cmu.edu
 # @Date: 2022-08-18 03:14:33
 # @LastEditTime: 2023-01-15 01:56:48
 # @Description: 
### 

JSON_PATH='./json'
FILES=$(ls $JSON_PATH)

for files in $FILES
do  
echo 'runing file:' $JSON_PATH'/'$files
ngc batch run -f $JSON_PATH'/'$files
sleep 30
done  

