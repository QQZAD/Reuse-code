#include <stdio.h>

#include <rte_eal.h>
#include <rte_common.h>
#include <rte_malloc.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_net.h>
#include <rte_flow.h>
#include <rte_cycles.h>

#define FLOWS_NB 4
#define BURST_SIZE 32

struct SPar
{
    int argc;
    char **argv;
};

static uint8_t *flowPac[FLOWS_NB];
static uint32_t flowPacLen[FLOWS_NB];

static uint16_t port_id;
struct rte_mempool *mbuf_pool;

static void init_port(void)
{
    int ret;
    struct rte_eth_conf port_conf;
    port_conf.rxmode.split_hdr_size = 0;
    port_conf.txmode.offloads = DEV_TX_OFFLOAD_VLAN_INSERT | DEV_TX_OFFLOAD_IPV4_CKSUM | DEV_TX_OFFLOAD_UDP_CKSUM | DEV_TX_OFFLOAD_TCP_CKSUM | DEV_TX_OFFLOAD_SCTP_CKSUM | DEV_TX_OFFLOAD_TCP_TSO;

    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_dev_info dev_info;

    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0)
        rte_exit(EXIT_FAILURE,
                 "Error during getting device (port %u) info: %s\n",
                 port_id, strerror(-ret));

    port_conf.txmode.offloads &= dev_info.tx_offload_capa;
    printf(":: initializing port: %d\n", port_id);
    ret = rte_eth_dev_configure(port_id,
                                FLOWS_NB, 0, &port_conf);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE,
                 ":: cannot configure device: err=%d, port=%u\n",
                 ret, port_id);
    }

    rxq_conf = dev_info.default_rxconf;
    rxq_conf.offloads = port_conf.rxmode.offloads;
    for (int i = 0; i < FLOWS_NB; i++)
    {
        ret = rte_eth_rx_queue_setup(port_id, i, 512,
                                     rte_eth_dev_socket_id(port_id),
                                     &rxq_conf,
                                     mbuf_pool);
        if (ret < 0)
        {
            rte_exit(EXIT_FAILURE,
                     ":: Rx queue setup failed: err=%d, port=%u\n",
                     ret, port_id);
        }
    }

    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0)
        rte_exit(EXIT_FAILURE,
                 ":: promiscuous mode enable failed: err=%s, port=%u\n",
                 rte_strerror(-ret), port_id);

    ret = rte_eth_dev_start(port_id);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE,
                 "rte_eth_dev_start:err=%d, port=%u\n",
                 ret, port_id);
    }

    printf(":: initializing port: %d done\n", port_id);
}

static void main_loop(void)
{
    struct rte_mbuf *mbufs[BURST_SIZE];
    struct rte_ether_hdr *eth_hdr;
    struct rte_mbuf *m;
    uint8_t *_flowPac;
    uint16_t total_len;

    while (1)
    {
        for (int i = 0; i < FLOWS_NB; i++)
        {
            uint16_t nb_rx = rte_eth_rx_burst(port_id, i, mbufs, BURST_SIZE);
            if (nb_rx)
            {
                printf("nb_rx-%d\n", nb_rx);

                total_len = 0;
                for (int j = 0; j < nb_rx; j++)
                {
                    m = mbufs[j];
                    total_len += m->pkt_len;
                }

                flowPacLen[i] = total_len;
                _flowPac = (uint8_t *)malloc(sizeof(uint8_t) * flowPacLen[i]);

                for (int j = 0; j < nb_rx; j++)
                {
                    m = mbufs[j];
                    eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
                    memcpy(_flowPac, (uint8_t *)eth_hdr, m->pkt_len);
                    // rte_pktmbuf_free(m);
                    _flowPac += m->pkt_len;
                }
            }
        }

        break;
    }

    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
}

void *launchDpdk(void *_argc)
{
    SPar *par = (SPar *)_argc;
    int argc = par->argc;
    char **argv = par->argv;

    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
    {
        rte_exit(EXIT_FAILURE, "EAL初始化失败\n");
    }

    uint16_t nr_ports = rte_eth_dev_count_avail();
    // if (nr_ports == 0)
    // {
    //     rte_exit(EXIT_FAILURE, "没有发现以太网设备端口\n");
    // }
    port_id = 0;

    mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", 4096, 128, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (mbuf_pool == NULL)
    {
        rte_exit(EXIT_FAILURE, "无法初始化mbuf_pool\n");
    }

    init_port();

    // main_loop();

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

lsmod
sudo modprobe uio
sudo insmod ~/dpdk-stable-19.11.3/build/kernel/linux/igb_uio/igb_uio.ko

python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --status
sudo ifconfig wlp0s20f3 down
sudo python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --bind=igb_uio 00:14.3
sudo python3 ~/dpdk-stable-19.11.3/usertools/dpdk-devbind.py --bind=iwlwifi 00:14.3

rm -rf dpdk;g++ -g dpdk.cpp -o dpdk -I /usr/local/include -lrte_eal -lrte_ethdev -lrte_mbuf;sudo ./dpdk
rm -rf dpdk
*/