# Pet问题反馈
这里进行两个方面的介绍：

+ 提问与贡献解决方案
+ pet安装常见问题的示例
## 提问与贡献解决方案
这里我们希望为使用者在安装和使用pet时提供一个解决问题的途径。
### 关于提问的注意内容
+ 安装出现问题时，请先核对环境要求、安装步骤和已存在的解决方案
+ 不要针对一个问题重复提问
+ 提问时，请将自己的环境、执行过程、描述清楚
+ 遇到不同于已有的问题时，创建新的Issue，以问题出现的环节命名

在此，我们也十分欢迎和感谢用户提供自己的解决方案
## 安装中常见问题的示例
这里用来介绍pet安装的常见问题及解决办法，按问题产生的执行步骤划分板块，来汇总错误。
### 执行“sudo pip3 install -r PytorchEveryThing/requirements.txt”
Issue 1：报错信息显示：“no module named gdbm”问题。

解决方案：使用apt-get安装“python3.5-gdbm”模块：

```
sudo apt-get install python3.5-gdbm
```

Issue 2：报错信息显示："python setup.py egg_info"。

解决方案：安装“libpq-dev“和”python3.5-dev”模块：

```
sudo apt-get install libpq-dev python3.5-dev
```

### 在“PytorchEveryThing/pet”下执行“./make”

Issue 1：“make: g++: Command not found”

解决方案：更新apt-get,安装与gcc对应版本的g++

```
sudo apt-get update
sudo apt-get install g++
```
Issue 2：ImportError: torch.utils.ffi requires the cffi package
解决方案：在python2.7下安装“cffi”模块

```
sudo pip install cffi
```

### 运行SSD时的错误

#### Issue ：RuntimeError: randperm is only implemented for CPU

```
File "./tools/ssd/train_net_multi_gpu.py", line 330, in <module>
main()
File "./tools/ssd/train_net_multi_gpu.py", line 299, in main
train_loss = train(model, train_loader, criterion, priors, optimizer, epoch)
File "./tools/ssd/train_net_multi_gpu.py", line 72, in train
for i, (inputs, targets) in enumerate(iterator):
File "/usr/local/lib/python3.5/dist-packages/torch/utils/data/dataloader.py", line 451, in __iter__
return _DataLoaderIter(self)
File "/usr/local/lib/python3.5/dist-packages/torch/utils/data/dataloader.py", line 247, in __init__
self._put_indices()
File "/usr/local/lib/python3.5/dist-packages/torch/utils/data/dataloader.py", line 295, in _put_indices
indices = next(self.sample_iter, None)
File "/usr/local/lib/python3.5/dist-packages/torch/utils/data/sampler.py", line 138, in __iter__
for idx in self.sampler:
File "/usr/local/lib/python3.5/dist-packages/torch/utils/data/sampler.py", line 51, in __iter__
return iter(torch.randperm(len(self.data_source)).tolist())
RuntimeError: randperm is only implemented for CPU
```

解决方案：修改“python3.6/site-packages/torch/utils/data“路径下的”sampler.py"文件

```
cd python3.6/site-packages/torch/utils/data
sudo vim sampler.py
```

(类名称：RandomSampler；函数：iter；51行)将原来的:

```Python
def __iter__(self):
    return iter(torch.randperm(len(self.data_source)).tolist())
```

修改为：

```Python
def __iter__(self):
    cpu = torch.device('cpu')
    return iter(torch.randperm(len(self.data_source), device=cpu).tolist())
```
