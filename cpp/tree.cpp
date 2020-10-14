#include <stdio.h>
/*
满二叉树:除叶子节点外的所有节点的度均为2 n=2^h-1
完全二叉树:从左到右,从上到下,节点连续分布 n<=2^h-1

BST二叉查找树:
若左子树不空，则左子树上所有节点的值均小于它的根节点的值
若右子树不空，则右子树上所有节点的值均大于或等于它的根节点的值
左、右子树也分别为二叉查找树

BST的树高为h，节点个数为n
查找、插入、删除时间复杂度:
平均情况：O(h)
最好情况：BST成为平衡二叉查找树，假设其包含最大满二叉树的节点数为m，其树高为h-1，则
m=2^(h-1)-1=>h=log(2)(m+1)+1
n>=m+1=>log(2)(n)>=log(2)(m+1)=>log(2)(m+1)=⌊log(2)(n)⌋
所以h=⌊log(2)(n)⌋+1=>O(h)=O(log(2)n)=O(logn)
这里的底数可以省略，他们都属于同一种时间复杂度/同阶无穷小，与n无关（分治法的思想，二分法的底数为2）
最坏情况：退化为单链表，O(n)

平衡的目的:避免BST的最坏情况
平衡二叉查找树:可以是空树,任意一个节点的左子树和右子树都是平衡二叉树，并且高度之差的绝对值不超过1
查找、插入、删除时间复杂度为O(logn)

红黑树:自平衡二叉查找树
1.每个节点都是红色和黑色
2.红黑树的根节点必须是黑色的
3.叶子节点NIL是黑色的
4.红色节点的两个子结点一定是黑色的
5.任意一个节点到它的每个叶子节点的路径都包含数量相同的黑色结点

考虑从根节点到叶子节点的可能路径
根据性质4路径上不存在两个连续红色节点，
最短可能路径都是黑色节点，
根节点和叶子节点NIL都是黑色，最长可能路径有交替的红色和黑色节点
根据性质5最短可能路径和最长可能路径均具有相同数量的黑色节点，
最长可能路径中红色节点比黑色节点少1
最长可能路径<最短可能路径*2
*/

enum COLOR
{
    RED,
    BLACK
};

struct rbTree
{
    COLOR color;
    int key;
    struct rbTree *left;
    struct rbTree *right;
    struct rbTree *p;
};

int main()
{
    return 0;
}