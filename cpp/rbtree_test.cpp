#include "rbtree.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

struct mynode
{
	struct Node node;
	char *string;
};

struct Root mytree;

struct mynode *mySearch(struct Root *root, char *string)
{
	struct Node *node = root->node;

	while (node)
	{
		struct mynode *data = container_of(node, struct mynode, node);
		int result;

		result = strcmp(string, data->string);

		if (result < 0)
			node = node->left;
		else if (result > 0)
			node = node->right;
		else
			return data;
	}
	return nullptr;
}

int myInsert(struct Root *root, struct mynode *data)
{
	struct Node **newNode = &(root->node), *parent = nullptr;

	/* Figure out where to put newNode node */
	while (*newNode)
	{
		struct mynode *p = container_of(*newNode, struct mynode, node);
		int result = strcmp(data->string, p->string);

		parent = *newNode;
		if (result < 0)
			newNode = &((*newNode)->left);
		else if (result > 0)
			newNode = &((*newNode)->right);
		else
			return 0;
	}

	/* Add newNode node and rebalance tree. */
	rbLinkNode(&data->node, parent, newNode);
	rbInsertColor(&data->node, root);

	return 1;
}

void my_free(struct mynode *node)
{
	if (node != nullptr)
	{
		if (node->string != nullptr)
		{
			free(node->string);
			node->string = nullptr;
		}
		free(node);
		node = nullptr;
	}
}

#define NUM_NODES 32

int main()
{

	struct mynode *mn[NUM_NODES];

	/* *insert */
	int i = 0;
	printf("insert node from 1 to NUM_NODES(32): \n");
	for (; i < NUM_NODES; i++)
	{
		mn[i] = (struct mynode *)malloc(sizeof(struct mynode));
		mn[i]->string = (char *)malloc(sizeof(char) * 4);
		sprintf(mn[i]->string, "%d", i);
		myInsert(&mytree, mn[i]);
	}

	/* *search */
	struct Node *node;
	printf("search all nodes: \n");
	for (node = rbFirst(&mytree); node; node = rbNext(node))
		printf("key = %s\n", RB_ENTRY(node, struct mynode, node)->string);

	/* *delete */
	printf("delete node 20: \n");
	struct mynode *data = mySearch(&mytree, "20");
	if (data)
	{
		rbErase(&data->node, &mytree);
		my_free(data);
	}

	/* *delete again*/
	printf("delete node 10: \n");
	data = mySearch(&mytree, "10");
	if (data)
	{
		rbErase(&data->node, &mytree);
		my_free(data);
	}

	/* *delete once again*/
	printf("delete node 15: \n");
	data = mySearch(&mytree, "15");
	if (data)
	{
		rbErase(&data->node, &mytree);
		my_free(data);
	}

	/* *search again*/
	printf("search again:\n");
	for (node = rbFirst(&mytree); node; node = rbNext(node))
		printf("key = %s\n", RB_ENTRY(node, struct mynode, node)->string);
	return 0;
}
/*
cd cpp;g++ -g -std=c++17 sort.cpp -o sort;./sort;cd ..
cd cpp;rm -rf sort bubbleSort.txt selectionSort.txt quickSort.txt;cd ..
*/
