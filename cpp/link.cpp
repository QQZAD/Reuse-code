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

void reverse(node *&head)
{
    node *pre = NULL, *curr = head, *post = head->next;
    while (curr != NULL)
    {
        curr->next = pre;
        pre = curr;
        curr = post;
        if (post != NULL)
            post = post->next;
    }
    head = pre;
}

void lastkNode(node *head, int k)
{
    if (head != NULL && k != 0)
    {
        node *first = head, *last = head;
        for (int i = 0; i < k - 1; ++i)
        {
            if (first->next != NULL)
            {
                first = first->next;
            }
        }
        while (first->next != NULL)
        {
            first = first->next;
            last = last->next;
        }
        printf("%d\n", last->val);
    }
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

void createCycle(node *head)
{
    int i = 0;
    node *p, *q = NULL;
    for (p = head; p->next != NULL; p = p->next)
    {
        i++;
        if (i == 4)
        {
            q = p;
        }
    }
    p->next = q;
}

void detectCycle(node *head)
{
    node *slow = head, *fast = head;
    while (fast && fast->next)
    {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast)
        {
            fast = head;
            while (fast != slow)
            {
                slow = slow->next;
                fast = fast->next;
            }
            printf("cycle-%d\n", fast->val);
            break;
        }
    }
}

int main()
{
    node *head;
    create(head, 15);
    // print(head);
    // reverse(head);
    // print(head);
    // lastkNode(head, 6);
    createCycle(head);
    detectCycle(head);
}
/*
cd cpp;g++ -g link.cpp -o link;./link;cd ..
cd cpp;rm -rf link;cd ..
*/