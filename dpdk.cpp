#include <stdio.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

// #include "../config/flows.hpp"

#define FLOWS_NB 4
//队列大小
#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32 //批大小

struct SPar
{
    int argc;
    char **argv;
};

// //使用全局设置
// //使用作为参数传递的mbuf_pool的RX缓冲区初始化给定端口。
// static int port_init(uint16_t port, struct rte_mempool *mbuf_pool)
// {
//     struct rte_eth_conf port_conf;
//     port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;

//     const uint16_t rx_rings = 1, tx_rings = 1; //收发包队列
//     uint16_t nb_rxd = RX_RING_SIZE;
//     uint16_t nb_txd = TX_RING_SIZE;
//     int retval;
//     uint16_t q;
//     struct rte_eth_dev_info dev_info;
//     struct rte_eth_txconf txconf;

//     if (!rte_eth_dev_is_valid_port(port))
//         return -1;

//     retval = rte_eth_dev_info_get(port, &dev_info);
//     if (retval != 0)
//     {
//         printf("Error during getting device (port %u) info: %s\n",
//                port, strerror(-retval));
//         return retval;
//     }

//     if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
//         port_conf.txmode.offloads |=
//             DEV_TX_OFFLOAD_MBUF_FAST_FREE;

//     // 配置以太网设备
//     retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
//     if (retval != 0)
//     {
//         printf("rte_eth_dev_configure failed \n");
//         return retval;
//     }

//     retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
//     if (retval != 0)
//         return retval;

//     // 每个以太网端口分配并设置1个RX队列
//     for (q = 0; q < rx_rings; q++)
//     {
//         retval = rte_eth_rx_queue_setup(port, q, nb_rxd, rte_eth_dev_socket_id(port), NULL, mbuf_pool);
//         if (retval < 0)
//         {
//             printf("rte_eth_tx_queue_setup failed \n");
//             return retval;
//         }
//     }

//     txconf = dev_info.default_txconf;
//     txconf.offloads = port_conf.txmode.offloads;

//     // 每个以太网端口分配并设置1个TX队列
//     for (q = 0; q < tx_rings; q++)
//     {
//         retval = rte_eth_tx_queue_setup(port, q, nb_txd, rte_eth_dev_socket_id(port), &txconf);
//         if (retval < 0)
//         {
//             printf("rte_eth_tx_queue_setup failed \n");
//             return retval;
//         }
//     }

//     //开启以太网端口
//     retval = rte_eth_dev_start(port);
//     if (retval < 0)
//     {
//         printf("rte_eth_dev_start failed \n");
//         return retval;
//     }

//     // //打印端口的MAC地址
//     // struct rte_ether_addr addr;
//     // retval = rte_eth_macaddr_get(port, &addr);
//     // if (retval != 0)
//     // 	return retval;

//     // printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
//     // 		   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
//     // 		port,
//     // 		addr.addr_bytes[0], addr.addr_bytes[1],
//     // 		addr.addr_bytes[2], addr.addr_bytes[3],
//     // 		addr.addr_bytes[4], addr.addr_bytes[5]);

//     //在混杂模式下为以太网设备启用RX
//     retval = rte_eth_promiscuous_enable(port);
//     if (retval != 0)
//         return retval;

//     return 0;
// }

//主要的逻辑核
//完成工作的主线程，从输入端口读取并写入输出端口
static int lcore_main(__attribute__((unused)) void *arg)
{

    uint16_t port;
    unsigned int lcore_id = rte_lcore_id(); //线程（核）的ID

    // //检查端口是否与轮询线程在同一节点上，以实现最佳性能。
    // RTE_ETH_FOREACH_DEV(port)
    // if (rte_eth_dev_socket_id(port) > 0 && rte_eth_dev_socket_id(port) != (int)rte_socket_id())
    //     printf("WARNING, port %u is on remote NUMA node to "
    //            "polling thread.\n\tPerformance will "
    //            "not be optimal.\n",
    //            port);

    // printf("\nCore %u forwarding packets. [Ctrl+C to quit]\n", lcore_id);

    //运行，直到应用程序退出或终止
    while (1)
    {

        //在端口上接收数据包，并在配对的端口上转发它们。 映射为0-> 1，1-> 0，2-> 3，3-> 2，依此类推。
        RTE_ETH_FOREACH_DEV(port)
        {
            //从配对的第一个端口获取RX数据包
            struct rte_mbuf *bufs[BURST_SIZE];
            const uint16_t nb_rx = rte_eth_rx_burst(port, 0, bufs, BURST_SIZE);

            // if (unlikely(nb_rx == 0))
            //     continue;

            //nb_rx表示包的个数，bufs表示包的地址指针
            //将包拷贝到GPU显存

            // pFlowsPackets[lcore_id] = NULL;
            // flowsPacNb[lcore_id] = BURST_SIZE;

            // //将TX数据包发送到第二个端口
            // const uint16_t nb_tx = rte_eth_tx_burst(port ^ 1, 0, bufs, nb_rx);

            // //释放所有未发送的数据包
            // if (unlikely(nb_tx < nb_rx))
            // {
            //     uint16_t buf;
            //     for (buf = nb_tx; buf < nb_rx; buf++)
            //     {
            //         // rte_pktmbuf_free(bufs[buf]);
            //     }
            // }
        }

        /*暂时只考虑一个批处理过程*/
        break;
    }
}

//执行初始化并调用per-lcore函数。
void *launchDpdk(void *_argc)
{
    SPar *par = (SPar *)_argc;
    int argc = par->argc;
    char **argv = par->argv;

    struct rte_mempool *mbuf_pool;
    unsigned nb_ports;
    uint16_t portid;
    unsigned lcore_id;

    //初始化EAL层
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

    // //检查是否有偶数个端口要发送或者接收
    // nb_ports = rte_eth_dev_count_avail();
    // if (nb_ports < 2 || (nb_ports & 1))
    //     rte_exit(EXIT_FAILURE, "Error: number of ports must be even\n");

    // //在内存中创建一个新的内存池来保存mbufs
    // mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    // //判断内存池是否申请成功
    // if (mbuf_pool == NULL)
    //     rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    // //初始化端口
    // RTE_ETH_FOREACH_DEV(portid)
    // if (port_init(portid, mbuf_pool) != 0)
    //     rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu16 "\n", portid);

    // 多线程绑定，循环处理数据包
    // rte_eal_mp_remote_launch(lcore_main, NULL, SKIP_MASTER); //SKIP_MASTER = 0,  < lcore handler not executed by master core.

    //在每个slave线程上调用lcore_main
    RTE_LCORE_FOREACH_SLAVE(lcore_id)
    {
        rte_eal_remote_launch(lcore_main, NULL, lcore_id);
    }

    // for (int lcore_id = 0; lcore_id < FLOWS_NB; lcore_id++)
    // {
    //     rte_eal_remote_launch(lcore_main, NULL, lcore_id);
    // }

    //在主内核上调用lcore_main
    // lcore_main(NULL);

    rte_eal_mp_wait_lcore(); //等待所有的lcore完成，为每个lcore发出rte_eal_wait_lcore()

    return NULL;
}

int main(int argc, char *argv[])
{
    SPar par;
    par.argc = argc;
    par.argv = argv;
    launchDpdk((void *)&par);
    return 0;
}

/*
ls /sys/kernel/mm/hugepages
sudo sh -c "echo 512 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
sudo gedit /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

sudo modprobe uio
sudo insmod ~/dpdk-stable-19.11.3/build/kernel/linux/igb_uio/igb_uio.ko

python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --status
sudo ifconfig enp7s0 down
sudo python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --bind=igb_uio enp7s0

rm -rf dpdk;g++ -g dpdk.cpp -o dpdk -I /usr/local/include -lrte_eal -lrte_ethdev -lrte_mbuf;sudo ./dpdk
rm -rf dpdk
*/