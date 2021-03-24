#include <stdio.h>

struct node
{
    int val;
    node *next;
    node(int _val, node *_next = NULL)
    {
        val = _val;
        next = _next;
    }
};

void create(node *&head, int nb)
{
    node **pdata = new node *[nb];
    for (int i = 0; i < nb; ++i)
    {
        pdata[i] = new node(nb - i);
        if (i >= 1)
            pdata[i]->next = pdata[i - 1];
    }
    head = pdata[nb - 1];
}

void print(node *head)
{
    for (node *p = head; p != NULL; p = p->next)
    {
        printf("%d", p->val);
        if (p->next != NULL)
        {
            printf("->");
        }
    }
    printf("\n");
}

int main()
{
    node *head;
    create(head, 15);
    print(head);
}
/*
cd cpp;g++ -g link.cpp -o link;./link;cd ..
cd cpp;rm -rf link;cd ..
*/