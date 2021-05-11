#include "rbtree.hpp"

static void rbRotateLeft(struct Node *node, struct Root *root)
{
	struct Node *right = node->right;
	struct Node *parent = RB_PARENT(node);
	if ((node->right = right->left))
	{
		rbSetParent(right->left, node);
	}
	right->left = node;
	rbSetParent(right, parent);
	if (parent)
	{
		if (node == parent->left)
		{
			parent->left = right;
		}
		else
		{
			parent->right = right;
		}
	}
	else
	{
		root->node = right;
	}
	rbSetParent(node, right);
}

static void rbRotateRight(struct Node *node, struct Root *root)
{
	struct Node *left = node->left;
	struct Node *parent = RB_PARENT(node);
	if ((node->left = left->right))
	{
		rbSetParent(left->right, node);
	}
	left->right = node;
	rbSetParent(left, parent);
	if (parent)
	{
		if (node == parent->right)
		{
			parent->right = left;
		}
		else
		{
			parent->left = left;
		}
	}
	else
	{
		root->node = left;
	}
	rbSetParent(node, left);
}

void rbInsertColor(struct Node *node, struct Root *root)
{
	struct Node *parent, *gparent;
	while ((parent = RB_PARENT(node)) && RB_IS_RED(parent))
	{
		gparent = RB_PARENT(parent);
		if (parent == gparent->left)
		{
			{
				register struct Node *uncle = gparent->right;
				if (uncle && RB_IS_RED(uncle))
				{
					RB_SET_BLACK(uncle);
					RB_SET_BLACK(parent);
					RB_SET_RED(gparent);
					node = gparent;
					continue;
				}
			}
			if (parent->right == node)
			{
				register struct Node *tmp;
				rbRotateLeft(parent, root);
				tmp = parent;
				parent = node;
				node = tmp;
			}
			RB_SET_BLACK(parent);
			RB_SET_RED(gparent);
			rbRotateRight(gparent, root);
		}
		else
		{
			{
				register struct Node *uncle = gparent->left;
				if (uncle && RB_IS_RED(uncle))
				{
					RB_SET_BLACK(uncle);
					RB_SET_BLACK(parent);
					RB_SET_RED(gparent);
					node = gparent;
					continue;
				}
			}
			if (parent->left == node)
			{
				register struct Node *tmp;
				rbRotateRight(parent, root);
				tmp = parent;
				parent = node;
				node = tmp;
			}
			RB_SET_BLACK(parent);
			RB_SET_RED(gparent);
			rbRotateLeft(gparent, root);
		}
	}
	RB_SET_BLACK(root->node);
}

static void rbEraseColor(struct Node *node, struct Node *parent, struct Root *root)
{
	struct Node *other;
	while ((!node || RB_IS_BLACK(node)) && node != root->node)
	{
		if (parent->left == node)
		{
			other = parent->right;
			if (RB_IS_RED(other))
			{
				RB_SET_BLACK(other);
				RB_SET_RED(parent);
				rbRotateLeft(parent, root);
				other = parent->right;
			}
			if ((!other->left || RB_IS_BLACK(other->left)) && (!other->right || RB_IS_BLACK(other->right)))
			{
				RB_SET_RED(other);
				node = parent;
				parent = RB_PARENT(node);
			}
			else
			{
				if (!other->right || RB_IS_BLACK(other->right))
				{
					RB_SET_BLACK(other->left);
					RB_SET_RED(other);
					rbRotateRight(other, root);
					other = parent->right;
				}
				rbSetColor(other, RB_COLOR(parent));
				RB_SET_BLACK(parent);
				RB_SET_BLACK(other->right);
				rbRotateLeft(parent, root);
				node = root->node;
				break;
			}
		}
		else
		{
			other = parent->left;
			if (RB_IS_RED(other))
			{
				RB_SET_BLACK(other);
				RB_SET_RED(parent);
				rbRotateRight(parent, root);
				other = parent->left;
			}
			if ((!other->left || RB_IS_BLACK(other->left)) && (!other->right || RB_IS_BLACK(other->right)))
			{
				RB_SET_RED(other);
				node = parent;
				parent = RB_PARENT(node);
			}
			else
			{
				if (!other->left || RB_IS_BLACK(other->left))
				{
					RB_SET_BLACK(other->right);
					RB_SET_RED(other);
					rbRotateLeft(other, root);
					other = parent->left;
				}
				rbSetColor(other, RB_COLOR(parent));
				RB_SET_BLACK(parent);
				RB_SET_BLACK(other->left);
				rbRotateRight(parent, root);
				node = root->node;
				break;
			}
		}
	}
	if (node)
	{
		RB_SET_BLACK(node);
	}
}

void rbErase(struct Node *node, struct Root *root)
{
	struct Node *child, *parent;
	int color;
	if (!node->left)
	{
		child = node->right;
	}
	else if (!node->right)
	{
		child = node->left;
	}
	else
	{
		struct Node *old = node, *left;
		node = node->right;
		while ((left = node->left) != nullptr)
		{
			node = left;
		}
		if (RB_PARENT(old))
		{
			if (RB_PARENT(old)->left == old)
			{
				RB_PARENT(old)->left = node;
			}
			else
			{
				RB_PARENT(old)->right = node;
			}
		}
		else
		{
			root->node = node;
		}
		child = node->right;
		parent = RB_PARENT(node);
		color = RB_COLOR(node);
		if (parent == old)
		{
			parent = node;
		}
		else
		{
			if (child)
			{
				rbSetParent(child, parent);
			}
			parent->left = child;
			node->right = old->right;
			rbSetParent(old->right, node);
		}
		node->parentColor = old->parentColor;
		node->left = old->left;
		rbSetParent(old->left, node);
		goto color;
	}
	parent = RB_PARENT(node);
	color = RB_COLOR(node);
	if (child)
	{
		rbSetParent(child, parent);
	}
	if (parent)
	{
		if (parent->left == node)
		{
			parent->left = child;
		}
		else
		{
			parent->right = child;
		}
	}
	else
	{
		root->node = child;
	}
color:
	if (color == RB_BLACK)
	{
		rbEraseColor(child, parent, root);
	}
}

static void rbAugmentPath(struct Node *node, rbAugmentF func, void *data)
{
	struct Node *parent;
up:
	func(node, data);
	parent = RB_PARENT(node);
	if (!parent)
	{
		return;
	}
	if (node == parent->left && parent->right)
	{
		func(parent->right, data);
	}
	else if (parent->left)
	{
		func(parent->left, data);
	}
	node = parent;
	goto up;
}

/*
 * 在向树中插入@node之后，更新树以考虑新条目和重新平衡造成的任何损害
 */
void rbAugmentInsert(struct Node *node, rbAugmentF func, void *data)
{
	if (node->left)
	{
		node = node->left;
	}
	else if (node->right)
	{
		node = node->right;
	}
	rbAugmentPath(node, func, data);
}

/*
 * 在删除节点之前，在rebalance路径上找到在@node被删除后仍然存在的最深的节点
 */
struct Node *rbAugmentEraseBegin(struct Node *node)
{
	struct Node *deepest;
	if (!node->right && !node->left)
	{
		deepest = RB_PARENT(node);
	}
	else if (!node->right)
	{
		deepest = node->left;
	}
	else if (!node->left)
	{
		deepest = node->right;
	}
	else
	{
		deepest = rbNext(node);
		if (deepest->right)
		{
			deepest = deepest->right;
		}
		else if (RB_PARENT(deepest) != node)
		{
			deepest = RB_PARENT(deepest);
		}
	}
	return deepest;
}

/*
 * 移除后，更新树来解释移除的条目和任何重新平衡的伤害
 */
void rbAugmentEraseEnd(struct Node *node, rbAugmentF func, void *data)
{
	if (node)
	{
		rbAugmentPath(node, func, data);
	}
}

/*
 * 这个函数返回树的第一个节点(按排序顺序)
 */
struct Node *rbFirst(const struct Root *root)
{
	struct Node *n = root->node;
	if (!n)
	{
		return nullptr;
	}
	while (n->left)
	{
		n = n->left;
	}
	return n;
}

struct Node *rbLast(const struct Root *root)
{
	struct Node *n = root->node;
	if (!n)
	{
		return nullptr;
	}
	while (n->right)
	{
		n = n->right;
	}
	return n;
}

struct Node *rbNext(const struct Node *node)
{
	struct Node *parent;
	if (RB_PARENT(node) == node)
	{
		return nullptr;
	}
	// 如果有一个右子结点，往下走，然后尽可能往左走
	if (node->right)
	{
		node = node->right;
		while (node->left)
		{
			node = node->left;
		}
		return (struct Node *)node;
	}
	// 没有右手的孩子。左边和下面的节点都比我们小，所以任何“next”节点都必须在父节点的大致方向上。
	// 爬上树;只要祖先是父结点的右子结点，就一直往上走。
	// 第一次它是父节点的左子节点时，说的父节点就是我们的next节点
	while ((parent = RB_PARENT(node)) && node == parent->right)
	{
		node = parent;
	}
	return parent;
}

struct Node *rbPrev(const struct Node *node)
{
	struct Node *parent;
	if (RB_PARENT(node) == node)
	{
		return nullptr;
	}
	// 如果有左子结点，就往下，然后往右，越远越好
	if (node->left)
	{
		node = node->left;
		while (node->right)
		{
			node = node->right;
		}
		return (struct Node *)node;
	}
	// 没有左手的孩子。向上走，直到我们找到一个祖先，它是其父的右子结点
	while ((parent = RB_PARENT(node)) && node == parent->left)
	{
		node = parent;
	}
	return parent;
}

void rbReplaceNode(struct Node *victim, struct Node *replace, struct Root *root)
{
	struct Node *parent = RB_PARENT(victim);
	// 将周围的节点设置为指向待更换节点
	if (parent)
	{
		if (victim == parent->left)
		{
			parent->left = replace;
		}
		else
		{
			parent->right = replace;
		}
	}
	else
	{
		root->node = replace;
	}
	if (victim->left)
	{
		rbSetParent(victim->left, replace);
	}
	if (victim->right)
	{
		rbSetParent(victim->right, replace);
	}
	// 将指针/颜色从victim复制到replace
	*replace = *victim;
}