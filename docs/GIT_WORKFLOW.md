# GitHub使用与多人协作

仓库：https://github.com/Dundun123123/435

## 新电脑

```powershell
git clone https://github.com/Dundun123123/435.git
cd 435
git status
git log -1 --oneline
```

然后按README创建新环境和下载数据。不要复制第一台电脑的.venv，不需要把完整数据推到GitHub。

## 开发分支

在main工作区干净且没有待保留本地改动时更新：

```powershell
git switch main
git pull --ff-only
git switch -c feature/mpi-two-nodes
# GPU任务另开feature/gpu-tfidf
```

开发完成、测试通过后，只添加明确的代码和已审查文档：

```powershell
git add mpi_main.py docs tests
git diff --cached --stat
git diff --cached
git commit -m "Implement and validate two-node MPI TF-IDF"
git push -u origin feature/mpi-two-nodes
```

上面mpi_main.py是未来文件，当前尚不存在。不要照抄commit信息把未验证功能写成已验证。GitHub上创建PR，由组员查看后合并；不要force-push覆盖另一台电脑的提交。

## 首次认证

公开仓库clone通常不需登录；push需要写权限。Git for Windows若带Git Credential Manager，可以在用户本机终端执行：

```powershell
git credential-manager github login
```

按实际安装版本的帮助完成浏览器/设备授权。不要把密码或token粘贴到聊天、README或Git远程URL里。也可以使用用户已经配置的GitHub CLI或SSH认证。凭据交给Git凭据管理器保存。

第一次commit还需用户自己的署名：

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

以上只设置本仓库，不修改全局配置。自动准备的首次提交若使用Codex本地自动化署名，不等于为用户声明GitHub身份。

## 发布实验结果

新结果默认放results/local/并忽略。完成核对、删除公开副本中的敏感主机信息，再复制到如results/mpi_two_nodes/或results/gpu/。提交数据清单、原始时间CSV、环境版本、验证结果和图，不提交完整原始语料或下载缓存。

## 若当前准备电脑尚未成功push

完整本地仓库与离线备份见交付包。用户登录后，在仓库根目录：

```powershell
git remote -v
git status
git push -u origin main
```

只有push成功且GitHub显示对应提交，另一台电脑的git clone才能获取内容。若远程已被他人更新，先检查和正常合并，不force-push。

