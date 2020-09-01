#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_eal.h>
#include <pthread.h>
#include <signal.h>
#include <netinet/ether.h>

/*该宏定义用于对接到GRIGHT*/
// #define GRIGHT
/*流的条数*/
#define FLOWS_NB 3 //对应./dpdk -l 0-3
/*接收描述符的初始数量*/
#define RX_RING_SIZE 1024
#define BURST_SIZE 32
/*mbuf池中rte_mbuf的数量，内存池的最佳大小(根据内存使用情况):n = (2^q - 1)*/
#define NUM_MBUFS 8981 //2^13-1
/*每个核对象缓存的大小*/
#define MBUF_CACHE_SIZE 250

struct SPar
{
    int argc;
    char **argv;
    SPar(int _argc, char **_argv)
    {
        argc = _argc;
        argv = _argv;
    }
};

/*流的批处理大小*/
static uint32_t batch_size[FLOWS_NB] = {128, 384, 256};
/*使用第一个网络设备端口*/
static uint16_t portid = 0;
/*单队列网卡*/
static uint16_t queueid = 0;
static volatile bool force_quit;
static struct rte_mempool *mbuf_pool;
static pthread_mutex_t lock;

static void signal_handler(int signum)
{
    struct rte_eth_stats eth_stats;
    if (signum == SIGINT || signum == SIGTERM)
    {
        printf("\n\n收到信号%d，准备退出...\n", signum);
        rte_eth_stats_get(0, &eth_stats);
        printf("接收%lu个数据包/丢弃%lu个RX数据包/传输失败%lu个数据包/RX mbuf分配失败%lu个数据包\n", eth_stats.ipackets, eth_stats.imissed, eth_stats.ierrors, eth_stats.rx_nombuf);
        force_quit = true;
    }
}

static inline int port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
    if (!rte_eth_dev_is_valid_port(port))
    {
        rte_exit(EXIT_FAILURE, "网络设备端口%u无效\n", port);
    }
    /*使用默认参数配置网络设备端口*/
    struct rte_eth_conf port_conf;
    port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;
    struct rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(port, &dev_info);
    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    {
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    }
    /*运行lspci -vvv|grep "MSI-X"后有结果返回说明NIC支持多队列*/
    const uint16_t nb_rx_queue = 1;
    const uint16_t nb_tx_queue = 0;
    uint16_t nb_rxd = FLOWS_NB * RX_RING_SIZE;
    uint16_t nb_txd = 0;

    int ret = rte_eth_dev_configure(port, nb_rx_queue, nb_tx_queue, &port_conf);
    if (ret != 0)
    {
        rte_exit(EXIT_FAILURE, "无法配置网络设备端口%u\n", port);
    }
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (ret != 0)
    {
        rte_exit(EXIT_FAILURE, "网络设备端口%u的RX/TX描述符数量不合法\n", port);
    }
    ret = rte_eth_rx_queue_setup(port, queueid, nb_rxd, rte_eth_dev_socket_id(port), NULL, mbuf_pool);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE, "无法创建网络设备端口%u的RX队列\n", port);
    }
    ret = rte_eth_dev_start(port);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE, "无法启动网络设备端口%u\n", port);
    }
    /*为网络设备端口启用混杂模式的接收*/
    rte_eth_promiscuous_enable(port);
    return 0;
}

static int lcore_main(__attribute__((unused)) void *arg)
{
    unsigned lcore_id = rte_lcore_id();
    struct ether_hdr *eth_hdr;

    /*一条流的所有数据包的总长度之和是一个很大的数，需要用uint64_t类型*/
    uint64_t batch_len = 0;
    uint32_t batch_nb = 0;

    uint16_t flow_batch_size = batch_size[lcore_id - 1];
    uint8_t *pac_bytes[flow_batch_size] = {NULL};
    uint32_t pac_len[flow_batch_size] = {0};

    uint8_t *flow = NULL;
    uint8_t *bytes = NULL;
    printf("\nlcore %u 接收数据包 [用Ctrl+C终止]\n", lcore_id);
    while (1)
    {
        struct rte_mbuf *bufs[BURST_SIZE];
        pthread_mutex_lock(&lock);
        uint16_t nb_rx = rte_eth_rx_burst(portid, queueid, bufs, BURST_SIZE);
        pthread_mutex_unlock(&lock);
        if (nb_rx)
        {
            uint16_t _nb_rx = nb_rx;
            for (int i = 0; i < nb_rx; i++)
            {
                unsigned pac_id = batch_nb + i;
                if (pac_id >= flow_batch_size)
                {
                    _nb_rx = i;
                    break;
                }
                pac_len[pac_id] = bufs[i]->pkt_len;
                batch_len += pac_len[pac_id];
                pac_bytes[pac_id] = (uint8_t *)malloc(sizeof(uint8_t) * pac_len[pac_id]);
                eth_hdr = rte_pktmbuf_mtod(bufs[i], struct ether_hdr *);
                memcpy(pac_bytes[pac_id], (uint8_t *)eth_hdr, pac_len[pac_id]);
                rte_pktmbuf_free(bufs[i]);
            }
            batch_nb += _nb_rx;
            printf("lcore %u 已接收%u个数据包\n", lcore_id, batch_nb);
            if (batch_nb >= flow_batch_size)
            {
                printf("lcore %u 数据包的总长度为%lu\n", lcore_id, batch_len);
                flow = (uint8_t *)malloc(sizeof(uint8_t) * batch_len);
                bytes = flow;
                for (int i = 0; i < flow_batch_size; i++)
                {
                    memcpy(bytes, pac_bytes[i], pac_len[i]);
                    bytes += pac_len[i];
                    free(pac_bytes[i]);
                }
                /*flow*/
                //TODO 考虑多个批处理过程
                break;
            }
        }
        if (force_quit)
        {
            break;
        }
    }
    return 0;
}

void *launchDpdk(void *par)
{
    SPar *spar = (SPar *)par;
    int argc = spar->argc;
    char **argv = spar->argv;

    force_quit = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE, "EAL初始化错误，没有分配巨页内存? 没有加载驱动模块? 没有绑定网卡？运行命令没有加sudo？\n");
    }
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0)
    {
        rte_exit(EXIT_FAILURE, "没有找到网络设备端口，估计该网卡不支持DPDK\n");
    }
    else
    {
        printf("网络设备端口的数量为%u\n", nb_ports);
    }
    mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", nb_ports * FLOWS_NB * NUM_MBUFS, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (mbuf_pool == NULL)
    {
        rte_exit(EXIT_FAILURE, "无法创建mbuf池\n");
    }
    port_init(portid, mbuf_pool);
    pthread_mutex_init(&(lock), NULL);
    rte_eal_mp_remote_launch(lcore_main, NULL, SKIP_MASTER);
    rte_eal_mp_wait_lcore();
    pthread_mutex_destroy(&(lock));
    return NULL;
}

#ifndef GRIGHT
int main(int argc, char *argv[])
{
    SPar spar(argc, argv);
    launchDpdk((void *)&spar);
    return 0;
}
#endif
/*
[1]确保DPDK正确安装并配置
注意：对于虚拟机中的网卡，在编译安装之前
gedit ~/dpdk-stable-19.11.3/kernel/linux/igb_uio/igb_uio.c
将pci_intx_mask_supported(udev->pdev)改为pci_intx_mask_supported(udev->pdev)||true

[2]根据硬件配置和需求分配巨页内存
ls /sys/kernel/mm/hugepages
sudo sh -c "echo 512 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
sudo gedit /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

[3]加载PDDK支持的驱动模块
lsmod
sudo modprobe uio
sudo insmod ~/dpdk-stable-19.11.3/build/kernel/linux/igb_uio/igb_uio.ko

[4]卸载选定的网卡并将该网卡绑定至igb_uio驱动（选择一个网卡就可以）
python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --status
卸载网卡ens38(02:06.0)
sudo ifconfig ens38 down
将网卡ens38(02:06.0)绑定至DPDK驱动igb_uio
sudo python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --bind=igb_uio 02:06.0
将网卡ens38(02:06.0)绑定至内核驱动e1000
sudo python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --bind=e1000 02:06.0

[5]编译并运行可执行文件dpdk
cd dpdk;rm -rf dpdk;g++ -march=native -mno-avx512f -g dpdk.cpp -o dpdk -I /usr/local/include -lrte_eal -lrte_ethdev -lrte_mbuf -lrte_mempool;sudo ./dpdk -l 0-3;cd ..

[6]清除可执行文件dpdk
cd dpdk;rm -rf dpdk;cd ..
*/