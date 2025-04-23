cd /home/jovyan/Desktop/mae

# 一、先把整個 .git 資料夾刪掉，清乾淨
rm -rf .git

# 二、重新初始化 repo 並建立 .gitignore
git init
cat > .gitignore << 'EOF'
# 忽略模型輸出資料夾
output_dir/
# 忽略擷取後的病患特徵
brain_extracted_features/
# 忽略所有 tensorboard log 檔及 tfevents 檔
log_dir/
*.tfevents.*
EOF

# 三、建立 README.md（如果需要）
[ -f README.md ] || echo "# mae" > README.md

# 四、把所有檔案（除了 .gitignore 裡忽略的）加入、commit
git add .
git commit -m "Clean initial commit: drop large log files"

# 五、設定遠端 (HTTPS 或 SSH 看你偏好)
git remote add origin git@github.com:WinterBelieve/mae.git
# 或者：git remote add origin https://github.com/WinterBelieve/mae.git

# 六、強制推送到 main
git branch -M main
git push -f origin main
