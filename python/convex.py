# 可导一定连续,但连续不一定可导(\/),极限0/0的定义
# 0<=t<=1
# x和y是f(x)定义域内任意两点
# convex 凸函数 下凸上凹 f"(x)>=0 f(tx+(1-t)y)<=tf(x)+(1-t)f(y)
# concave 凹函数 上凸下凹 f"(x)<=0 f(tx+(1-t)y)>=tf(x)+(1-t)f(y)
# 简单的说,凸优化要求优化问题中的目标函数和约束函数是凸函数
# 凸集的定义:对于集合内的每一对点,连接该对点的直线段上的每个点也在该集合内,封闭性
# 驻点的定义:一阶导数为0,可导函数的极值点一定是驻点,驻点不一定是极值点
# 不可导点的左右极限不相等,不可导点左右的单调性发生改变则是极值点
# 极小值点需要考察驻点和不可导点
# 拐点的定义:二阶导数为0,该点左右的凹凸性不同,且一阶导数的变化趋势相反
# 对于凸函数而言,极小值点就是最小值点,所以局部最优就是全局最优
