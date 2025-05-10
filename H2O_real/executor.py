import os
import sys
# 在根目录下执行一些脚本（主要是测试脚本），以处理python路径引用的问题

# 获取项目根目录的绝对路径
root_path = os.path.abspath(os.path.dirname(__file__))

# 将项目根目录添加到sys.path
sys.path.append(root_path)

# 指定要运行的脚本路径
script_path = os.path.join(root_path, 'sim2real/example/', 'g1_example.py')
# 运行指定脚本
with open(script_path, 'rb') as file:
    exec(compile(file.read(), script_path, 'exec'))