#pragma once
struct Node
{
	unsigned long parentColor;
#define RB_RED 0
#define RB_BLACK 1
	struct Node *right;
	struct Node *left;
};

struct Root
{
	struct Node *node;
};

#define RB_PARENT(r) ((struct Node *)((r)->parentColor & ~3))
#define RB_COLOR(r) ((r)->parentColor & RB_BLACK)
#define RB_IS_RED(r) (!RB_COLOR(r))
#define RB_IS_BLACK(r) RB_COLOR(r)
#define RB_SET_RED(r)           \
	do                          \
	{                           \
		(r)->parentColor &= ~1; \
	} while (0)
#define RB_SET_BLACK(r)        \
	do                         \
	{                          \
		(r)->parentColor |= 1; \
	} while (0)

static inline void rbSetParent(struct Node *rb, struct Node *p)
{
	rb->parentColor = (rb->parentColor & 3) | (unsigned long)p;
}

static inline void rbSetColor(struct Node *rb, int color)
{
	rb->parentColor = (rb->parentColor & ~1) | color;
}

#define RB_ENTRY(ptr, type, member) container_of(ptr, type, member)

#define RB_EMPTY_ROOT(root) ((root)->Node == nullptr)
#define RB_EMPTY_NODE(node) (RB_PARENT(node) == node)
#define RB_CLEAR_NODE(node) (rbSetParent(node, node))

static inline void rbInitNode(struct Node *rb)
{
	rb->parentColor = RB_RED;
	rb->right = nullptr;
	rb->left = nullptr;
	RB_CLEAR_NODE(rb);
}

extern void rbInsertColor(struct Node *, struct Root *);
extern void rbErase(struct Node *, struct Root *);

typedef void (*rbAugmentF)(struct Node *node, void *data);

extern void rbAugmentInsert(struct Node *node, rbAugmentF func, void *data);
extern struct Node *rbAugmentEraseBegin(struct Node *node);
extern void rbAugmentEraseEnd(struct Node *node, rbAugmentF func, void *data);

// 在树中查找逻辑上的下一个和上一个节点
extern struct Node *rbNext(const struct Node *);
extern struct Node *rbPrev(const struct Node *);
extern struct Node *rbFirst(const struct Root *);
extern struct Node *rbLast(const struct Root *);

// 快速替换单个节点，无需删除/再平衡/添加/再平衡
extern void rbReplaceNode(struct Node *victim, struct Node *replace, struct Root *root);

static inline void rbLinkNode(struct Node *node, struct Node *parent, struct Node **rbLink)
{
	node->parentColor = (unsigned long)parent;
	node->left = node->right = nullptr;
	*rbLink = node;
}