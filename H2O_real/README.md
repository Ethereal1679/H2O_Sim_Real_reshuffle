# G1-Sim2Real
policy deployment on Unitree G1

### Installation ###
```bash
cd unitree_sdk2_python-master
pip install -e .
```
安装其他部署网络需要的模块

### Usage ###
unitree_sdk2py使用pip安装的方式导入，大部分脚本都可以直接引用

sim2real和其他相关脚本/路径默认采用以g1_sim2real/为根目录的相对引用，可以通过executor.py间接执行来规避引用问题

提供了用executor.py执行g1_example.py的案例作为参考，g1的底层通信和控制接口可参考sim2real/example/g1_example.py