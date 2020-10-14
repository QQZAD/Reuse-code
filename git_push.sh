#!/bin/bash
# 非维护人员请不要运行此脚本
for file in `ls $1`
do
    if [ "${file#*.}" != "sh" ] && [ "${file#*.}" != "md" ] && [ "${file#*.}" != "cpp" ] && [ "${file#*.}" != "cuda" ] && [ "${file#*.}" != "dpdk" ] && [ "${file#*.}" != "python" ] && [ "${file#*.}" != "ptm_data" ]
    then
        echo "发现未跟踪的文件$file，已终止！"
        exit
    fi
done
git config --global user.email "2298070398@qq.com"
git config --global user.name "QQZAD"
git config --global push.default matching
git status
git add .
git commit -m "reuse-code"
git push
